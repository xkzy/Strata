# Startup benchmark

The tuner runs automatically. There is no required flag, no GPU id, no batch size and no thread count to supply.
It measures a bounded amount, solves, validates, persists a record, and applies only the knobs it measured. It is
in [`pkg/autotune/plan.go`](../../pkg/autotune/plan.go) (workflow),
[`pkg/autotune/benchmark.go`](../../pkg/autotune/benchmark.go) (bounded runner and probes) and
[`pkg/autotune/startup.go`](../../pkg/autotune/startup.go) (the server/CLI entry point).

## Workflow

```text
discover -> load record (hint) -> bounded probe -> solve -> validate -> persist -> apply
```

1. **Discover** the machine; never fail.
2. **Load** `strata-tuning.json` (or `--tuning-record`, or `STRATA_TUNING_RECORD`). A record is used only if its
   schema, hardware fingerprint, workload hash and hard constraints still match, and it is younger than 30 days.
   It is a hint, not truth: its measurements are re-checked by the solver and a refresh always wins per key. A
   cached start does **not** take a fresh performance sample; it validates the stored facts and refreshes the
   numbers later from the background engine pass. A changed driver, backend or model invalidates it outright.
3. **Probe** only when there is no usable measurement or the caller explicitly asks (`strata tune`). The server
   starts with `Measure: false` and `FirstRunCalibration: true`, so the first start pays for the host probe and
   later starts read the record instead.
4. **Solve** the finite domain (see `solver-and-optimality.md`).
5. **Validate** the selected plan with a health check and a small request when a probe can serve. Validation is
   recorded; it does not change the plan the server already started with.
6. **Persist** atomically. A failed or unwritable record does not fail startup.
7. **Apply** only measured knobs (`--pool-workers`, `--pcie-frac`, `--spec-min-p`). Everything else stays as the
   engine config had it.

## Phases and limits

`Runner.Run` executes the phases in this order; the engine starts only after the tuner returns.

| Phase | Work | Typical bound |
|---|---|---|
| health | probe's own health gate; a non-`ok` device stops the run before any work | phase limit |
| warmup | one short request per candidate whose devices pass health | phase limit |
| memory | host copy bandwidth (host probe) | 200 ms |
| compute | scalar CPU GFLOPS for CPU candidates; device compute when a probe supports it | 200 ms per candidate |
| scaling | CPU copy bandwidth at 1, CPU/2, 3×CPU/4 and CPU workers | 100 ms per point (reduced by the phase limit) |
| transfer | directional pair transfers among healthy devices, only when ≥ 2 exist | phase limit per pair |
| serving | one short prefill/decode per candidate, repeated when the budget allows | phase limit per candidate |
| cleanup | always, on a fresh 5 s deadline, even after a timeout or abort | 5 s |

Default limits: **45 s total, 8 s per phase, 2 GiB host memory**, one repeat. The engine refinement pass uses
30 s total, 8 s per phase and two repeats. `strata tune` defaults to a 60 s `--timeout` on the whole run.
Memory limits are checked after warmup and after each served candidate through `Probe.Usage()`; a probe that
reports more than the limit aborts the run.

The host probe allocates at most 128 MiB for a copy/scale test (64 MiB per buffer, or `available/16` if smaller)
and never touches a device. The engine probe sends at most one streaming completion per candidate, with at most
8 output tokens, over a 30 s HTTP client timeout.

## Safety rules

- A device must be `health = ok` before warmup and again before serving. `unknown` gets no work.
- The first fatal error aborts the run and is preserved as the first causal error. There is **no retry**, no
  reboot, and no "try again" advice: a device that faults is left alone.
- `Cleanup` always runs and has its own deadline. Completed samples are kept; the failed candidate is not
  published.
- The tuner never allocates the reported free VRAM. The host probe stays in regular Go memory; device work is
  whatever the engine probe sends through the server's normal request path.
- No thermal, power or stress test is run. A card reporting a hardware brake or ≥ 90 °C is `failed`, not a test
  target.
- No numerical precision, quantization or model semantics change is attempted by the tuner.
- The last plan is recorded even when the run aborts, with the failure appended; the active engine configuration
  is never migrated mid-request. Applying a new plan happens before engine start (or, at the next start, from the
  record).

## Engine probe scope

A rate measured on the **running** engine cannot be attributed to the other candidates.The engine probe therefore implements `CandidateScoper`: it measures only the one configuration it can reconstruct from the
engine command line (a CPU-only server, or a single healthy GPU without an explicit `--gpu` list). Its worker
count, PCIe share and speculation threshold are read from `--pool-workers`, `--pcie-frac` and `--spec-min-p`, and
the covered configuration is added to the probe list even when the default domain did not contain it. With an
explicit `--gpu` list, several visible GPUs, or a server-enforced different layout, it measures nothing and says
so in `notes`; the run is used for validation only. This avoids the tempting bug where one engine rate makes
every candidate look measured and equally fast.

`CandidateScoper` is also the extension point for a future device-memory-bandwidth probe: implement `Probe`,
report `Covers` for the candidates it can represent, and the solver will use its rates.

## Validation

With a probe that can serve, validation:

- checks device health through the probe;
- runs an optional `CorrectnessProbe` (no built-in comparison exists yet; the engine's own start-up already
  rejects unsupported kernels);
- serves a 32-token prompt and 2 tokens, with a 10 s bound;
- records `Validation.Validated/Correct/Detail` in the tuning record.

A record that is not validated may still lend its CPU measurements, but engine-side knobs are applied only from
measurements, never from an unvalidated placement guess. A cached start that does not run a smoke test keeps the
validation the same configuration already earned, and keeps the last ten failures as diagnostic history.

## Startup latency

Measured on the development machine (AMD Ryzen 9 9950X3D, 64 GiB RAM, RTX 5060 Ti + RTX 4060, Linux, Go 1.25;
1 GiB test pack, manifest geometry):

| Path | Wall time | What ran |
|---|---|---|
| First run, host calibration | 1.1–1.4 s over repeated runs (`strata tune`, includes process start) | discovery + 4-phase CPU scaling + memory/compute probe |
| Cached record, no probe | 0.16 s | discovery + record validation + solve |
| Full `go run ./cmd/strata tune` | 1.1–1.4 s | compile cache warm, same host probe |

The probe phases themselves measured: memory 42.6 GB/s in 202 ms, scaling about 400 ms for four points (1 → 40.3,
16 → 43.3, 24 → 44.0, 32 → 44.4 GB/s, one sample each), compute 11.1 GFLOPS scalar. Selected workers: 16 (the
smallest point within 97% of the measured peak). These are one machine's numbers, not a target.

After the engine is ready, the server waits for an idle moment (at most 10 minutes for readiness and 30 s of
idling) and runs one 30 s-bounded engine pass in the background. If the server is still busy after that, the pass
is skipped rather than adding load. It does not change the running engine's arguments; the next start reads the
record. A real request always wins over refinement.

## CLI

```sh
go run ./cmd/strata tune --save=false                 # discover + plan + host probe, print JSON
go run ./cmd/strata tune --record /tmp/t.json         # persist
go run ./cmd/strata tune --measure=false --record /tmp/t.json   # cached path, no probe
go run ./cmd/strata tune --engine http://127.0.0.1:8080        # validate/measure a running server
go run ./cmd/strata tune --model model.gguf --context 32768
go run ./pkg/../bin/strata serve --no-autotune        # opt out
```

`/metrics` exposes the last pass under `tuning` (fingerprint, objective, classification, coverage, gap, selected
candidate, prediction basis, evidence, applied args, notes). It is `{"available": false}` until a pass runs, so a
tuner bug cannot break `/metrics`.

## Failure handling

| Situation | Result |
|---|---|
| no `/proc`, no `nvidia-smi`, no cgroup | capabilities partially unknown; conservative CPU plan |
| record missing | normal; probe or model-only plan |
| record corrupt / wrong fingerprint / stale | ignored with a reason; does not stop startup |
| record cannot be written | startup continues; note says why |
| health unknown or failed | that device gets no work; CPU fallback if no device is usable |
| probe timeout | aborted phase is recorded; completed samples kept; cleanup still runs |
| memory limit exceeded | abort; candidate not published |
| validation request fails | `Validated: false`, `Correct: false`; the plan is still reported |
| no candidate passes the hard constraints | no plan is selected and no record is written; the caller keeps its configuration; rejection reasons are reported |
| engine not ready for refinement | refinement returns without changing anything |

## Tested and untested

Tested with `go test ./pkg/autotune/`: health abort, first-error preservation, no-retry, timeout and cleanup,
unsupported transfer, candidate scoping, corrupted and stale records, solver certificates, noisy measurements.
The host probe was exercised on the development machine above. The engine probe was exercised against an
`httptest` SSE server, not a real model (no weights or built engine are present in this environment).
