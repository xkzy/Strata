# Runtime adaptation (Go server)

Strata adapts while it serves. The controller observes CPU, system RAM, GPU/VRAM, request latency and engine health,
then changes server-owned policy: admission concurrency, bounded queueing, large-prompt slots, the memory tier,
background optimization, and whether admission is open at all. It does not change sampling, numerics or a model's
tokens; every request gets the same result it would get without the controller.

The engine's own adaptive expert cache (`--adapt-every`) is unchanged and is documented in
[DETAILS.md](DETAILS.md). This document is the server loop above it.

The implementation is in [`pkg/resourcemanager/controller.go`](../pkg/resourcemanager/controller.go),
[`pkg/server/runtime.go`](../pkg/server/runtime.go) and
[`pkg/server/runtime_test.go`](../pkg/server/runtime_test.go).

## Control loops

| Loop | Interval | What it may change | Why it is safe |
|---|---|---|---|
| fast | 1 s | concurrency, admission open/closed | admission only; active requests keep their slot |
| medium | 5 s | queue limit, memory tier, large-prompt slots, background optimization | affects new admissions only |
| slow | 30 s / event | retune request after a persistent workload shift, health transitions | bounded measurement, idle-only, 10 min minimum gap |

The controller runs one 1 Hz loop; the fast/medium/slow deadlines are evaluated inside it. The loop is started by
`StartRuntime` from the shipped commands after the engine starts. `resource_adapt: false` disables the controller; a
config that omits the key keeps the default (on), so an older file is not silently read as "off".
`/metrics.resource.adapt.enabled` says which mode is active.

## What is observed

The sample is copied from the existing telemetry collector and request monitor. Missing metrics stay **unknown**;
they are never replaced by zero or by an optimistic assumption.

- CPU utilisation (valid only after the collector's second sample).
- RAM used, total and therefore available; the collector uses `MemAvailable`, not nominally free pages.
- VRAM used/total, GPU temperature, GPU utilisation where the driver reports them.
- Finished requests in the last 10 s: p95 prompt and output tokens, p95 TTFT, p95 latency and mean decode tok/s.
- In-flight requests, queue depth, admissions, rejections, cancellations and retry estimates.
- Engine state (`ready`, `failed`, `starting`, ...) and its reason. A failed engine closes admission immediately.
- Parked KV bytes from the engine's KV ledger are available for display; they are not used to migrate live cache.

Per-core CPU, NUMA placement, cgroup counters, per-device VRAM and allocation failures are not exposed by the
current collector. They remain unavailable at runtime and are handled conservatively at startup
([optimization/hardware-discovery.md](optimization/hardware-discovery.md)). On a host where the collector has no
system-wide memory reading (it falls back to the Go heap on non-Linux), `ram_system` is false and the controller
treats host RAM as unknown instead of using process memory as a substitute.

## Admission and queue

A request reserves one slot. `concurrency` slots run and up to `queue_limit` requests wait in a bounded broadcast
queue; there is no strict FIFO guarantee, so a waiter can be overtaken. If the queue is full, admission is
closed, or the engine/device failed, the request gets **429** with a `Retry-After` estimate derived from recent
completion rate. Both paths admit: `startGeneration` after tokenizing, and `startRuntimeGeneration` before the
runtime builds its prompt (its ticket is upgraded by `MarkLarge` once the prompt length is known). The API returns the same shape as the rest of Strata:

```json
{"error":{"message":"the server is at capacity: server_overloaded (retry after 3 s)",
          "type":"server_overloaded","code":"server_overloaded"}}
```

A waiting request is released when its HTTP context ends; it never holds a worker or engine slot while waiting.
Large prompts (at least 4096 tokens) are limited to `large_prompt_slots` while other work is active, so one huge
prefill cannot occupy every slot.

## Policy

`power_policy` selects bounds; it is not a performance claim.

| Policy | Start concurrency | Max concurrency | Queue (start/max) |
|---|---|---|---|
| `BALANCED` | 4 | 16 | 32 / 128 |
| `MAX_THROUGHPUT` | 8 | 64 | 128 / 512 |
| `LOW_LATENCY` | 2 | 8 | 8 / 32 |
| `ENERGY_SAVING` | 2 | 6 | 8 / 32 |

Memory tiers: `retain` below 70% RAM, `balanced` from 70%, `tight` from 85%, `aggressive` from 93% (and at 90%
VRAM, 95% VRAM critical, or 95 °C). `tight` halves the queue; `aggressive` quarters it and closes admission.
A tight tier also halves concurrency on the next fast evaluation. CPU saturation is 90%; a p95 TTFT 1.5× above the
EWMA is treated as a latency signal.

VRAM is read together with the engine's own plan. The engine's `INFO` line carries `vram_free_mib`: what its
ready-time plan left free after every expert tier, the KV cache and the prompt-path buffers. While that is at least
the engine's 256 MiB stall floor, the used percentage describes the plan, not pressure — a card that only just fits
the model reads 97% at rest — and it neither closes nor tightens admission. Below the floor, or with no plan
reported, the percentage keeps its old meaning; the engine prints its own "only just fits" warning there.
Measured 2026-10-19, RTX 5060 Ti alone: the engine reported 334 MiB free with everything loaded and the collector
read 97%; without this rule every request was refused with `429 server_overloaded: memory_critical` while the card
had 334 MiB to give. With it, the same server and config reported `vram_percent` 95.53, `engine.vram_free_mib` 334,
tier `retain`, admission open - and served three short chats (38.6-56.7 tok/s decode) and raw completions of 131,062
and 222,805 tokens (1,354 and 1,174 tok/s prefill, 82 °C, 154 W, no Xid).

One caveat: the plan is the engine's own ready-time count. Another program that takes VRAM after the start is not in
it; the engine's allocation failures remain the backstop there, not the percentage.

The numbers are policy constants, documented here, not measured universal optima. The controller labels every
decision with its reason in `/metrics`.

## Mathematical model

State `s_t` is the sample above; active policy `x_t` is `Knobs`. A proposal `a` is scored as

```text
J(a) = -G(a) + λ·R(a) + C(a)
```

- `G` is the expected gain: a concurrency increase pays `(new-old)/new`; a pressure response pays 0.5; a
  tier/queue adjustment pays 0.1 (it claims safety, not throughput); closing admission on a fault or critical
  pressure pays 1.0 and is never lost to the cost test.
- `R` is transition risk (0.02 for medium changes, 0.03–0.05 for growth, 0.2 for an emergency close).
- `C` is measured inside the commit under the controller lock (checks, validator, whole-`Knobs` swap, wake-up):
  microseconds here (0.00–0.05 ms in the tests), because these knobs allocate and migrate nothing. One
  `Knobs` representation also carries the (currently unused) caller estimate for a future transition that
  would prepare real resources.
- `λ = 0.5`.

A change commits only when `G > λR + C`. Constraints are hard checks before the score:

```text
1 ≤ concurrency ≤ policy max
min queue ≤ queue ≤ max queue
concurrency growth requires (n·256 MiB + 512 MiB) ≤ available RAM
unknown RAM is not permission to grow
validator(knobs) must pass (server/engine readiness)
```

The controller does **not** claim a mathematical optimum for a runtime decision. A full feasible space is not
enumerated, and the only transitions it can perform are server policy swaps whose measured cost is microseconds;
there is no model of a device migration to optimize. Decisions are therefore labelled heuristic/empirical in the
plan and in this document. The startup solver's `model_optimum` and `certified_bound` labels apply only to the
startup search.

## Transaction and rollback

Every change follows one path under one lock:

```text
OBSERVE -> PROPOSE -> CHECK -> PREPARE -> VALIDATE -> COMMIT -> MONITOR
```

- **CHECK**: policy bounds, memory headroom, queue bounds, admission state.
- **PREPARE**: for admission knobs, nothing is allocated; the counters are the reservation. Future device
  transitions must reserve destination resources here before commit.
- **VALIDATE**: an optional validator (engine ready, no fatal health) runs before the switch.
- **COMMIT**: the whole `Knobs` value is replaced atomically. Active requests keep their tickets; a shrink only
  affects new admissions, so no request is killed to satisfy a policy.
- **MONITOR**: the previous value is kept for 15 s. If p95 TTFT is 20% above the value captured at commit, mean
  decode throughput is 10% below it after a growth, RAM reaches critical, or rejections rose after a growth, the
  controller rolls back **only if** the previous state was open and the device is still healthy. A rollback never
  revives a closed admission or a failed device.

Conflicting loops cannot interleave: all mutations happen under the controller mutex, and only one pending
rollback exists. A failed check, validator or cost test leaves the active knobs untouched. There is exactly one
reconfiguration coordinator (the controller), not one per loop.

## Hysteresis and anti-oscillation

- Minimum dwell for growth: **10 s**; a shrink also delays growth by 10 s.
- Separate enter/leave thresholds: critical pressure (93% RAM, 95% VRAM) closes admission; reopening requires a
  later sample below the **tight** line (85% RAM, 90% VRAM), not merely below the critical one. That gap is what
  stops a value sitting on the critical line from flapping. If RAM telemetry disappears after a memory close, the
  missing reading is not treated as recovery.
- A recovery retune (failed -> ready) uses the same 10-minute minimum gap as a workload-shift retune.
- Monitor window: **15 s**; rollback is allowed only for a measured regression.
- Growth requires demand (a waiting request) and a complete low-pressure memory reading.
- No smoothing delays a fatal engine state: a `failed` sample closes admission on the same evaluation.
- Unknown CPU/RAM/VRAM never becomes a pressure signal or a growth permission.
- Change attribution: a prompt-length mix that moves by more than half during the monitor window makes the
  window **inconclusive** and is recorded as such; its latency change is not blamed on the reconfiguration.

## Failure containment

- Engine state `failed`: admission closes with reason `device_failure`; new requests get 429. Active requests end
  through the engine's normal error path. The controller does not probe the failed device or retry the operation.
- The engine orchestrator owns restart and GPU snapshots; the controller only observes the state. A later `ready`
  sample is the engine's validated start, and only then is admission reopened.
- Host RAM/VRAM critical: admission closes; active requests are not killed. Queue growth stops immediately.
- No buffer, page or model state is freed or migrated by the controller. The server owns no engine allocation.
- If the failure is partial across devices, the current telemetry aggregates GPUs and cannot attribute it; Strata
  treats the engine as failed and closes admission rather than guessing which device is safe. This is a documented
  limitation, not a recovery claim.

## Slow retuning

A persistent prompt-length shift (p95 more than 2× the earlier baseline, at least 1024 tokens) or a failed →
ready engine transition sets a retune request. Recovery is the one event that can invalidate a placement chosen
for the previous device, so it earns a re-measurement; the controller only consumes the orchestrator's `ready`
state and never probes the failed device. The server runs one bounded `autotune` engine pass only when the engine
is ready, the request monitor is idle, background optimization is allowed, and at least 10 minutes passed since
the last request. A request made while the server is busy stays pending (the medium loop disables background work
while requests run) and is consumed only when a pass actually starts; it is not dropped by one busy evaluation.
The pass uses a server lifecycle context canceled by `StopRuntime`, so shutdown aborts a pass that is still waiting
for an idle server (the same applies to the first startup refinement). It never changes the running engine's flags;
the result is a hint for the next start. No additional LLM is involved.

## Observability

`GET /metrics` → `resource.adapt`:

```json
{
  "enabled": true,
  "policy": "BALANCED",
  "knobs": {"concurrency": 4, "queue_limit": 32, "admission_open": true,
            "tier": "retain", "background_optimization": true, "large_prompt_slots": 4},
  "in_flight": 2, "queued": 0, "admitted": 42, "rejected": 1, "canceled": 0,
  "last_reject": "server_overloaded",
  "decisions": [{"from": {...}, "to": {...}, "reason": "...", "transition_cost_ms": 0,
                 "transition_risk": 0.03, "applied": true}],
  "pending": {"from": {...}, "to": {...}, "until": 0},
  "ewma_ttft_ms": 180.5, "ewma_decode_tok_s": 42.1,
  "sample": {"cpu_percent": 31, "ram_percent": 44, "engine_state": "ready"}
}
```

The legacy `resource` fields (`concurrency`, `queue_limit`, `active_work`, decisions) remain for compatibility and
are not the runtime controller's counters; `resource.adapt` is authoritative for this feature.

## Measured overhead

AMD Ryzen 9 9950X3D, Linux, Go 1.25, `go test -bench`:

| Operation | Time |
|---|---|
| `Controller.Observe` (one sample, loop checks) | **52.5–55.5 ns/op** |
| `Admit` + `Release` (disabled or available slot) | **214–225 ns/op** |
| Queue handoff (one waiter, one release) | **496–530 ns/op** |

The 1 Hz sampler is the existing telemetry collector; its `nvidia-smi` probes dominate a sample, not the
controller. Admission adds one ticket-map entry and removes it on release; there is no per-request goroutine or
buffer. Measurements are three `-count=3` runs of `-benchtime 200000x` on the current code; the spread between
those runs is a few percent.

### Sampler probe cost

Measured `nvidia-smi` calls on the same PC (20 calls each): the telemetry query 33.6 ms, `dmon -s t -c 1` 115.4 ms,
the startup `name,memory.total` query 20.4 ms. The sampler used to fork both probes on every tick, so one sample
cost 149 ms of its 1000 ms period - about 15% of one core, and every other collector in the process repeated it.

The probes in [`pkg/telemetry/nvprobe.go`](../pkg/telemetry/nvprobe.go) are now shared and rate-limited:

| Probe | Window | Why that window |
|---|---|---|
| `--query-gpu` (memory, utilisation, temperature, PCIe link) | 900 ms sharing, 5 s after a failure | one probe per window serves every collector; a missing or faulted driver is not re-probed every second |
| `dmon` PCIe throughput | 5 s, last good reading reused in between | it costs about a whole sampling period and only feeds the Monitor sparkline |
| static facts (`name,memory.total`, else AMD sysfs, else `lspci`) | once per process | the hardware does not change while the server runs |

One server process still reads VRAM once per tick: it is the input to the admission decision and should not be
older than a period. A driver that stops answering is bounded: each probe runs under a deadline (8 s query, 3 s
dmon), so a wedged `nvidia-smi` cannot freeze the sampler and leave the controller deciding on an old reading.

Measured with `go test -bench BenchmarkSampleOnce` (`-benchtime 100x`, this PC): one sample went from 148-150 ms
to 48-54 ms, and building a collector from 20.4 ms + 0.09 ms to 0.09 ms. The effect is largest where many
collectors exist at once: `go test -count=1 ./pkg/server/` went from 26.8 s to 2.0 s on this PC.

## Stress and regression tests

`go test -race ./pkg/resourcemanager/` and `go test -race ./pkg/server/`:

- sudden CPU contention (97%) shrinks concurrency;
- gradual RAM pressure steps through `balanced` and `tight` before closing at critical;
- RAM/VRAM critical closes admission and rejects with a reason, including a request already queued;
- a VRAM fill the engine's plan accounted for (334 MiB free, 97% used) keeps admission open; the same reading
  with no plan, or with a plan below the engine's 256 MiB floor, still closes it;
- eight concurrent admit/mark/release workers run against the loops without a race or deadlock;
- insufficient host memory blocks growth even under demand;
- unknown telemetry neither invents pressure nor permits growth;
- queue full returns 429 with `retry_after`; cancellation while queued returns promptly;
- device failure closes admission, rejects new work, recovers only on a healthy sample, and asks for one
  bounded retune after recovery;
- monitor-window rollback after growth with worse p95 TTFT or 10% worse decode throughput;
- a workload shift during the monitor window is recorded as inconclusive, not as a reconfiguration regression;
- 20 stable evaluations produce zero changes;
- policy bounds and a rejecting validator leave knobs unchanged;
- a long-prompt shift requests a retune;
- large-prompt slots, including the runtime's post-tokenization `MarkLarge` upgrade;
- `Configure` wakes queued requests instead of leaving them on the old channel;
- `StopRuntime` cancels a background refinement/startup experiment and is idempotent;
- server: 429/Retry-After for both APIs, metrics exposure, recent-request statistics.

These are simulated samples; no GPU was detached, overheated, OOM-killed or otherwise stressed. Faults are
injected through `resourcemanager.Sample` and the fake probe/test server.

## Unsupported runtime transitions

The engine exposes no IPC for changing its worker pool, expert cache, KV layout, model partitioning, VRAM pools
or stream count while it runs. Strata therefore **does not** pretend to change them at runtime. They remain
startup/next-start decisions from the auto-tuner. KV-cache park/restore is observed and displayed, not triggered by
the controller. Multi-device health is aggregated; a partial device fault closes admission rather than guessing a
safe split. No recovery from a fatal hardware or driver failure is claimed; the engine's own failure log and
orchestrator own that path.

## Testing

```sh
go test -race ./pkg/resourcemanager/ ./pkg/server/
go test ./pkg/resourcemanager/ -bench BenchmarkController -benchtime 200000x
go run ./cmd/strata tune --save=false    # startup planner, unchanged
```
