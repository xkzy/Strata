# Auto-adaptive resource use (server layer, #18)

Strata decides how much to do at once from what it measures at runtime, not from fixed assumptions
("GPU is always faster", "more threads are always better", "use 100% of everything"). The controller lives in the
Python server (`serve/resource_manager.py`); it never changes sampling or numerics - a request gets the same
tokens it would have got with the controller switched off.

The engine's own adaptive tier (moving experts between RAM and VRAM, `--adapt-every`) is unchanged and stays
documented in [DETAILS.md](DETAILS.md). This document is the server's loop on top of it.

## The loop

```text
ResourceMonitor -> WorkloadAnalyzer -> ResourceEstimator -> Scheduler -> admission / queue -> Telemetry
      ^____________________________________________________________________________| feedback
```

- **ResourceMonitor** reads what the server already samples - the 1 Hz hardware thread of
  `serve/telemetry.py` (CPU %, RAM, GPU load/VRAM/temperature/power where a driver reports them) and the queue
  counters. It never starts a new poll: every read is one dict copy of a sample that already exists. Anything
  the OS does not report is simply absent; nothing here can stop the server.
- **WorkloadAnalyzer** keeps the last 256 finished requests: which classes the window is, how fast requests
  arrive (last 30 s), and the p95 of queue wait, time-to-first-token and tokens/s.
- **ResourceEstimator** turns those numbers into a direction per knob (grow / hold / shrink), comparing p95
  latency against the workload's own earlier baseline (an EWMA), never against a hardcoded target: the loop
  says "p95 TTFT rose from 1.0 s to 3.0 s", not "TTFT is bad for this GPU".
- **Scheduler** applies a direction only after it held for 2 evaluations, sits behind a cooldown (2 s; growth
  waits twice as long right after a shrink), and holds growth back for 45 s when the grown level did not add
  3% throughput - the loop learns the operating point instead of oscillating increase/decrease every interval.
  Every change is recorded with its reason (see "Observability" below).

The numbers in that paragraph are policy constants in `serve/resource_manager.py`, not hardware facts.

## What it changes (and nothing else)

| Knob | What it does | Bounds |
| --- | --- | --- |
| `concurrency` | how many requests may be active at once | 1 .. the engine's own batch slots (learned from `INFO batch_slots`; 1 for a serial engine) |
| `queue_limit` | how many requests may wait before new ones get a structured 429 | per policy: `MAX_THROUGHPUT` 16-512, `LOW_LATENCY` 4-32, `BALANCED` 8-128, `ENERGY_SAVING` 4-64 |
| `cache_tier` | memory-pressure tier: `retain` / `balanced` / `tight` / `aggressive` | follows measured host RAM (tight from 85%, refuses new work at 93% with a floor of 2 in flight); a tight tier halves the admission limit |
| power policy | the targets above | `power_policy` in the config |

The knobs are the whole write surface. Sampling parameters, temperature, penalties, KV precision and every
engine setting are outside the loop; `resource_adapt` turns the whole thing off.

## Workload classes (inferred, never asked)

`SHORT_INTERACTIVE`, `LONG_CONTEXT`, `HIGH_THROUGHPUT`, `LOW_LATENCY`, `PREFILL_HEAVY`, `DECODE_HEAVY`,
`MEMORY_BOUND`, `COMPUTE_BOUND`, `TRANSFER_BOUND`, `MOE_SPARSE`, `MOE_DENSE` - a request is classified twice:
before it runs from its own characteristics (prompt size against the context, the asked answer length), and
when it finishes from what it measured (which half of the work dominated the engine's clock, the expert-cache
hit rate and the share of experts computed elsewhere from its DONE line). No client ever sends a class.

## Backpressure instead of exhaustion

When active + queued + reserved requests reach the capacity (`concurrency + queue_limit`), the next request is
refused with **429**, `"code": "server_overloaded"` and a `retry_after` in seconds - estimated from the
window's own completion rate (queue divided by observed completions per second, 1-60 s). Host memory at the
critical pressure refuses too, but never below the floor of 2 in flight: pressure must not block the last
request. An admitted request reserves its capacity until `run()` takes it over or the request ends, so a burst
of parallel posts cannot slip past the check; a reservation that never reaches the engine expires after 120 s.

## Loop protection (with `serve/anti_loop.py`, `serve/generation_loop.py`)

A session whose generation loop was suspected (the detector's `SUSPECTED` verdicts, `finish_reason=
generation_loop`, or the guard's `SUSPECTED`/`THROTTLED`/`BLOCKED` states from `/v1/strata/guard/check`) is
admitted only while the server is under half capacity, for 90 s from the last suspicion - a runaway generation
may not consume the whole server, and the next request from that session queues behind healthy ones. Recovery
is the TTL lapsing with no further suspicion. Dynamic temperature stays a decoding-layer mechanism: nothing
here changes logits, and no temperature change touches KV state.

## Config

Both keys live in `strata-<model>.json` (and in the web page's Model settings):

```json
"power_policy": "BALANCED",
"resource_adapt": true
```

`power_policy` is one of `MAX_THROUGHPUT`, `LOW_LATENCY`, `BALANCED`, `ENERGY_SAVING`. `resource_adapt: false`
keeps one fixed limit whatever the load (as before #18). On start the server prints which policy it runs with.

## Observability: GET /metrics, the `resource` section

- `workload`: dominant classes with their shares, arrival/completion rate, p95 TTFT and queue wait, samples.
- `knobs`: concurrency, ceiling from the engine's slots, queue limit, capacity, in flight, reservations,
  cache tier, any saturation hold left.
- `pressure`: host memory used/total, CPU %, GPU load, free VRAM - whatever the driver reported.
- `backpressure`: admitted, rejected per reason, the last rejection with its limit.
- `loop_protection`: suspected sessions, confirmed loops, admissions it limited.
- `decisions`: the last knob changes with `from`/`to` and a reason string - this is what answers "why is this
  request using this resource?" and "why is this resource idle?".

## What it does not do (yet)

Engine-layer knobs - CPU worker count per operation, per-operation device choice, heterogeneous splitting,
KV-cache growth per request, expert-placement rebalancing - are start-up flags today, and changing them at
runtime needs measurements on real hardware first; that is the engine phase of #18 and is not claimed here.
Tenant fairness and quotas stay in `serve/multi_tenant.py`. No GPU, energy or multi-device number is claimed
in this document: none was measured for this feature.

## Testing

```text
python -m unittest serve.test_resource_manager     # 24 tests: classification, grow/shrink, hysteresis,
                                                   # saturation, 429 + retry_after, cache tier, loop cap,
                                                   # Service wiring and the HTTP 429, mock engine only
python -m pytest serve/ -q                        # 468 passed, 8 skipped (Linux, Python 3.13, 2026-10-06)
```

The tests run without a GPU or a model. The issue's benchmark matrix (1/2/4/8+ concurrent, short/long/very
long context, CPU-only vs devices, low/high memory, energy) still needs hardware runs before any performance
number is written down.
