# Solver and optimality

The solver chooses one configuration from a finite, explicit set. It never claims more than it can prove:
correctness, memory and health are hard constraints, and every result states whether it is a model optimum, a
certified bound, an empirical optimum or a heuristic selection. The code is
[`pkg/autotune/solver.go`](../../pkg/autotune/solver.go); the model it ranks with is
[`performance-model.md`](performance-model.md).

## The candidate

```text
x = (placement, devices, pool_workers, pcie_frac, spec_min_p, batch, prefill_chunk)
```

These map to real Strata controls: CPU / one GPU / a layer split; `--pool-workers`; `--pcie-frac`;
`--spec-min-p`; the server's parallel slots; `--prefill` chunk. `PCIeFrac = -1` and `PrefillChunk = 0` mean "use
the engine's own default". `Candidate.Key()` is stable across processes and is what measurements and records are
keyed by.

## The finite domain

`BuildDomain` builds the search space from what discovery and measurement establish:

| Dimension | Values |
|---|---|
| placement | `cpu`; every healthy accelerator as `gpu`; every same-backend pair as `split` (even if it will be rejected, so the rejection is counted) |
| pool workers | the engine default `0`, every count the CPU scaling probe ran, its measured best, and every worker count in a measured candidate |
| pcie frac | `-1` (engine default) plus values from measured candidates only |
| spec min p | `0` plus values from measured candidates only |
| batch | `1` plus measured values up to the envelope's concurrency |
| prefill chunk | `0` plus values from measured candidates only |

The default is deliberately small: with three accelerators, one worker set and no measured knobs, the development
machine produced 25 candidates. A sweep adds the values it actually ran; no knob is invented. The full product
is enumerated by `Enumerate()`; `EnumerateCapped()` is used only to bound the number of **probe** candidates
(the server caps that at 4), never to bound the solver's evaluation.

## Hard constraints, not weights

`Check` returns the first set of violations. A candidate with any violation is rejected before scoring and is
counted in `Plan.Rejected` by stable code (`device_memory`, `host_memory`, `device_health`,
`numerical_format_unsupported`, `mixed_backend_split`, ...). The complete table is in
[`performance-model.md`](performance-model.md#hard-constraints).

Consequences worth stating explicitly:

- A healthy device with unknown free memory is rejected (`memory_uncertain`). Filling "most" of the reported
  free VRAM is not allowed.
- A split across different backends is rejected; no pointer, stream or event is shared across them.
- An unfit or failed device is never used, even if it would be fastest.
- Correctness checks are never traded against speed, and the tuner never changes precision or quantization.

## Objective and selection

`Score` converts a prediction to "higher is better": decode tok/s, aggregate decode throughput × concurrency,
`1000/TTFT`, or `1000/latency`. `ChooseObjective` picks one from the workload (see
[`performance-model.md`](performance-model.md#objectives)).

Ordering is by score, then by a deterministic tie-break (fewer devices, lower device index, fewer workers, lower
PCIe fraction). The selected candidate can then be adjusted, and these adjustments are part of the claim:

1. **Noise-aware fallback.** With `threshold = max(3%, 2 × noise)`, a valid previous configuration or the
   engine's default is kept when it is within `threshold` of the best score. This avoids flip-flopping on
   insignificant differences. It also means the selected candidate may not be the fastest model candidate; when
   that happens the plan is classified `heuristic_selection`, never `model_optimum`.
   A default is only ever kept if it is in the feasible set. If no candidate passes the hard constraints, no
   candidate is selected (`selected.placement` is empty), the rejection counts and the reason are reported, and
   no record is written: the caller keeps its own configuration rather than publishing an unsafe plan.
2. **Worker count.** The smallest measured worker count that reaches 97% of the measured peak copy bandwidth is
   used. It is an empirical point, not an interpolation.
3. **No host-only placement override.** If no accelerator candidate has a measured serving rate, the default
   placement (GPU when one is healthy, otherwise CPU) is kept even if the host model predicts CPU to be faster.
   Comparing a measured host against a theoretical device is not evidence. The plan is heuristic and the reason
   is in `notes`.

## Exhaustive search

`Solve` evaluates every candidate in the domain, so its result is the optimum **under the cost model** over the
defined finite space. It is exhaustive by construction: `Plan.Evaluated` equals `len(domain.Enumerate())` in
the tests. The classification is:

- `empirical_optimum` when every feasible candidate has a measured prediction and the selected one is the best;
- `model_optimum` when the selected candidate is the best modeled one and it was not displaced by the noise or
  default fallback;
- `heuristic_selection` when no candidate had a usable prediction, or when a fallback displaced the best, or
  when only measured-but-unusable alternatives existed;
- `certified_bound` is produced by branch-and-bound (below).

`Coverage` is `exhaustive` when the whole domain was evaluated; `bounded` when branch-and-bound pruned part of
it or the caller supplied a truncated domain. A bounded search can never be called a proven global optimum.

## Branch and bound

`BranchAndBound` computes, for each feasible candidate, an upper bound on the score implied by the roofline with
**every overhead removed** and with the **best bandwidth and compute any device in the discovery offers** (not
the candidate's own):

```text
T_bound = max( 2·P_active·tokens / P_best ,  (A_active + K) / BW_best )
```

Under the model's own assumption (active weights are moved at least once), a configuration cannot be faster than
a bound that ignores all communication, launch, synchronization and allocator cost, and uses the best hardware
rate available; its model score therefore cannot exceed the score derived from `T_bound`. Candidates are sorted
by that bound. The best is evaluated first; a remaining candidate is pruned when its bound cannot beat the best
observed score. Measured candidates receive an infinite bound and are never pruned, so a measurement is never
discarded by a model estimate. A cache that already holds every weight could beat the cold-weight assumption;
the certificate is therefore about the model, not about a warm machine.

The reported `gap` is the **relative margin** between the selected score and the best upper bound among pruned
candidates,

```text
margin = (best − worst_pruned_bound) / best
```

It is not a distance to the true optimum. Since pruning only removes candidates whose bound cannot beat the
best, the selected candidate is the model optimum over the space that was evaluated; the remaining optimality
gap **under the model** is zero. The margin is what makes the certificate auditable: a small margin means the
next candidate was close, a large margin means the bound was decisive.

Limits of the certificate:

- It is a certificate for **model scores**, not for the physical machine. Measured candidates are excluded from
  pruning, and a modeled candidate's real speed can differ from the roofline.
- The bound is loose by design (best device for every candidate, no overhead), which can make pruning rare. A
  loose bound never prunes a contender; it only weakens the usefulness of the certificate.
- `Plan.Gap` is 0 when nothing was pruned.

## What is proven and what is claimed

| Claim | Evidence | Label |
|---|---|---|
| Memory fits | arithmetic over discovered memory, geometry, explicit reserves | proven under the discovered numbers (uncertainty term included) |
| Context/batch valid | prompt+output ≤ physical ≤ trained; batch ≤ concurrency | proven |
| Kernel/format supported | SIMD flags, discovered backends, reported data types | proven when reported; `unknown` is recorded, not assumed |
| Health safe | only `health = ok`; unknown/failed rejected | proven under discovery's report |
| Selected is best modeled candidate | all domain candidates evaluated | `model_optimum` (not a measured guarantee) |
| Selected cannot be beaten under the model after pruning | valid upper bound + gap margin | `certified_bound` |
| Selected is fastest measured | every feasible candidate has a measured rate | `empirical_optimum` |
| Selected is a safe default | no usable prediction or a noise/default fallback | `heuristic_selection` |

No result is ever described as a mathematically proven global optimum of the physical machine. The model's
assumptions and residuals are in [`performance-model.md`](performance-model.md#known-limitations-and-residual-error).

## Reproducible validation

```sh
go test ./pkg/autotune/ -run 'TestSolve|TestBranchAndBound|TestHardConstraints|TestNoise|TestHostOnly' -v
go vet ./pkg/autotune/
go run ./cmd/strata tune --save=false --pack /path/to/pack --name Qwen3.8-Flash-Next
```

- `TestSolveAgreesWithBruteForce` builds the domain, solves it, and independently enumerates and checks it with
  the same constraints and model; it requires identical selection, exhaustive coverage and matching evaluated
  count.
- `TestBranchAndBoundMatchesAndCertifies` requires the same selected candidate as exhaustive search, at least
  one pruned candidate, a margin in `[0,1]`, and the `certified_bound` label.
- `TestHardConstraintsReject` covers tight VRAM, failed health, context, SIMD, mixed backends and format
  mismatch.
- `TestNoiseKeepsTheDefault` requires a 1% win under 10% noise to keep the default and be labelled heuristic, and
  a 30% win to switch and be labelled empirical.
- `TestHostOnlyModelKeepsTheDefaultPlacement` requires the GPU default to survive when no accelerator rate was
  measured.

Example run on the development machine (1 GiB test pack, manifest geometry, three discovered accelerators):
25 evaluated, 20 feasible, 5 rejected for device memory, 0 pruned, selected the default GPU placement with the
measured 16-worker point, classification `heuristic_selection` because no accelerator throughput was measured.
This is exactly the intended behavior on a first start with no engine rate; it is not a performance claim.
