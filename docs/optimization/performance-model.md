# Performance model

The model answers one question: given the discovered machine and the actual model shape, which configuration
should be tried, and which configurations cannot possibly be right? It is a roofline, not a simulator. Every
number is labelled `measured`, `modeled` or `unavailable`; no unmeasured rate is presented as a measurement.
The code is [`pkg/autotune/cost.go`](../../pkg/autotune/cost.go) and
[`pkg/autotune/workload.go`](../../pkg/autotune/workload.go).

## Workload

`Workload` is the model geometry plus the request envelope plus the physical context window.

- **Geometry** comes from the GGUF metadata (`strata.*` / `tokenizer.ggml.*` keys) when a native model is used.
  For a canonical pack the shape comes from `pkg/modelschema` when the model name is known; the pack itself only
  supplies its on-disk weight bytes. A field that is absent stays 0 and the affected prediction is
  `unavailable` rather than guessed.
- **`KVBytesPerToken`** = `Layers × KVHeads × HeadDim × 2 (K and V) × bytesPerElement`, with `KVHeads` falling
  back to `Heads` and 2 bytes (fp16) as the engine default, or 1 with `--kv int8`.
- **`ActiveParams`** counts every dense weight plus the routed experts (`ExpertsPerToken`), not the whole expert
  table. This is the decode roofline denominator.
- **`TotalParams`** counts the whole table, and is used only to turn measured on-disk bytes into bytes per
  parameter: `bytesPerParam = WeightBytes / TotalParams`.
- **Envelope** defaults to a 2,048-token prompt, a 256-token answer and one user, shrunk if it would exceed the
  physical window. When request statistics exist, `EnvelopeFromStats` takes the nearest-rank p90 of prompts and
  outputs; it does not interpolate. Concurrency is the observed safe floor (1 unless the caller supplies more).
- **Physical context** is `--max-context`, otherwise the model's trained context, otherwise 32,768. The workload
  fingerprint includes the context, geometry and KV format, so a change invalidates a persisted record.

## Roofline

For a candidate `x`, with active weight bytes `A` and one KV token `K`:

```text
F_prefill = 2 · P_active · L_prompt          FLOPs for the prompt
F_decode  = 2 · P_active                     FLOPs per generated token

T_prefill(x) ≥ max( F_prefill / P_eff(x),  A / BW_eff(x) )
T_decode(x)  ≥ max( F_decode  / P_eff(x), (A + K) / BW_eff(x) )
```

`P_eff` and `BW_eff` come from measurements when they exist for that candidate, otherwise from the calibrated
machine model below. The implementation then adds the launch and synchronization terms, which are currently 0
because no built-in probe sets them:

```text
T_prefill' = T_prefill + (LaunchOverheadMS + SyncOverheadMS)/1000
T_decode'  = T_decode  + (LaunchOverheadMS + SyncOverheadMS)/1000

PrefillTokS = L_prompt / T_prefill'
DecodeTokS  = 1 / T_decode'
TTFT        = 1000 · T_prefill'
Latency     = TTFT + 1000 · L_output / DecodeTokS
```

The prefill memory term reads the active weights once for the prompt. Real chunked prefill may stream them more
than once, so this term can only underestimate time, never overestimate it. The compute term scales with prompt
length. For decode the model is bandwidth-bound on the active experts, which is the engine's actual bottleneck.

For a multi-device split:

```text
T_distributed = T_compute + T_communication + T_synchronization + T_imbalance
```

The model does **not** invent the last three terms. A split candidate therefore has `Basis = unavailable` unless
a measured rate exists for that exact device pair. This is deliberate: assuming that two GPUs add, or that a
transfer overlaps, is how a tuner recommends a slower configuration.

## Effective rates

- **CPU memory bandwidth** is measured by the host probe: repeated copies of a 64 MiB buffer (or `available/16`,
  whichever is smaller) for 200 ms. Measured on the development machine: 40.3–44.4 GB/s at 1–32 workers.
- **CPU compute** is a scalar Go FMA loop, measured in GFLOPS, then scaled by the widest SIMD width discovery
  reported: ×16 for `avx512f`, ×8 for `avx2`, ×4 otherwise. This is a coarse model, not a measurement of the
  engine's AVX-512 expert kernels; it exists so CPU prefill is not treated as bandwidth-free. A modeled CPU
  result is always labelled `modeled`, never `measured`.
- **GPU rates** (`GPUComputeTFLOPS`, `GPUMemBandwidthGBs`) are only used when a measurement exists. The built-in
  host probe cannot measure a device and the engine probe cannot attribute a device rate to every candidate, so
  a first start has no GPU rate. In that case the solver **does not** compare the measured host against a
  theoretical device: it keeps the engine's default placement and labels the plan `heuristic_selection` with the
  reason in `notes`. An explicit `--gpu` list is not mapped to discovery indices, for the same reason.
- **Link bandwidth** (`GPULinkGBs`) is recorded with direction (`from->to`) but is not yet used by the cost
  model, which refuses to price a split without a measured serving rate for it.
- **Noise** is the largest coefficient of variation among repeated serving samples, floored at 2% (the
  repeatability floor of a short run) and capped at 25%. Decisions use `threshold = max(3%, 2 × noise)`.

## Hard constraints

These are checked before scoring; they are constraints, never weights in the objective.

| Constraint | Rule |
|---|---|
| Context | `prompt + output ≤ physical context` and `physical context ≤ trained context`. |
| Batch | `1 ≤ batch`, and `batch ≤ concurrency` when batch > 1. |
| CPU kernels | A canonical Q2_0 pack needs `avx512f`; a GGUF-native pack needs `avx2`, `neon` or `sve`. |
| Host memory | `weights + KV(context × batch) + concurrency × 128 MiB + HostReserve ≤ UsableBytes`. |
| Host pressure | `pressure = critical` rejects CPU placement; swap never counts as available memory. |
| Device memory | `KV × share + concurrency × 128 MiB + 3% × free + DeviceReserve ≤ free`; a split also budgets its `weights × share`. |
| Pinned staging | Requested pinned bytes may not exceed `PinnedLimitBytes`. |
| Device health | Only `health = ok` devices; `unknown` and `failed` are rejected. |
| Backend | The device's backend must be in discovered `backends`. |
| Mixed backends | A split needs one backend; different backends cannot share pointers, streams or events. |
| Numerical format | If a device reported data types, the model's format family must appear (`int4`/`fp16` for q4-family, `int8` for q8). No reported types means unknown, not unsupported. |
| Device index | The candidate's indices must exist in the discovered list. |
| Geometry completeness | Unknown on-disk weight bytes or unknown KV bytes/token reject every candidate (`memory_uncertain`): a budget cannot be proven from a missing number. |

`HostReserve` = round-up-to-64-MiB of `max(10% of total RAM, 1 GiB) + measured free-memory spread`.
`DeviceReserve` = 512 MiB rounded to 64 MiB. These are policy floors, not a claim of universal optimality; the
spread term is measured per machine and per run. `MemoryUncertainty` is 3%, and `WorkspacePerStream` is 128 MiB.

## Objectives

An objective is chosen automatically unless `Options.Objective` overrides it:

| Workload | Objective | Score |
|---|---|---|
| concurrency > 1 | `max_throughput` | `DecodeTokS × concurrency` |
| prompt > 4 × max(output, 64) | `min_ttft` | `1000 / TTFTms` |
| otherwise | `max_decode_tok_s` | `DecodeTokS` |

`min_latency` (`1000 / LatencyMS`) is available for an explicit request. When several users are present the
objective is aggregate throughput, not one request's speed, and the per-request latency prediction is still
reported so a latency regression is visible.

## What is measured and what is modeled

| Quantity | Source | Label |
|---|---|---|
| CPU memory bandwidth | host probe | measured |
| CPU scalar GFLOPS | host probe | measured |
| CPU vector GFLOPS | scalar × SIMD width | modeled |
| GPU compute / bandwidth | no built-in probe yet | unavailable unless a record or custom probe supplied it |
| Serving prefill/decode per candidate | host probe: unsupported; engine probe: only the running configuration | measured when scoped |
| KV bytes, weight bytes, peaks | geometry + memory | computed |
| Host/device reserve | discovery + constants | policy |
| TTFT, decode tok/s, latency | roofline | modeled (lower bound on time) |

## Known limitations and residual error

- **No end-to-end comparison against inference was possible in this environment**: the repository has no model
  weights and no built engine in `build/`, so predicted versus measured throughput could not be reported. The
  tests validate the model's algebra and the solver's decisions, not its absolute accuracy.
- The CPU prefill term is compute-bound with a scalar-probe-derived vector rate. On a real AVX-512 expert kernel
  this is conservative (it predicts too little throughput), which biases the plan away from CPU.
- GPU rates are absent on a first start. The policy keeps the default placement instead of guessing; the
  background refinement can supply a serving rate later, but only for a configuration it can identify.
- The prefill memory term assumes one pass over active weights. It can only make the bound looser (faster), not
  tighter. A fully warm cache could move fewer bytes than any step assumes; that is why model results are labelled
  `modeled`/`roofline`, not measurements.
- `LaunchOverheadMS` and `SyncOverheadMS` are part of the schema but no probe sets them yet; launch overhead is
  therefore not modeled.
- Batch and prefill-chunk candidates only enter the domain when a probe actually tested them. Neither is applied
  by `ApplyArgs`, because the engine has no validated argument for them.
- Split candidates are never priced by theory. Multi-GPU use is enabled only when its measured end-to-end rate
  exists; otherwise it is rejected as unavailable and the reason is recorded.
