# Performance bottleneck analysis

Labels: **[M]** measured (source), **[V]** verified by reading code/headers, **[D]** derived arithmetic from [M]/[V] inputs, **[I]** inferred, **[U]** not measured. Nothing here is a student result; no student exists.

## 1. Headline finding

For this teacher on this PC, decode time per verify pass is set by **how many expert-weight bytes must be read from RAM instead of VRAM**, not by FLOPs, kernel launches, or the number of sequential levels. The model's stored size (29.6 GB resident + 28.8 GB n-gram table) exceeds the 24 GB of total VRAM, so part of the 10-of-256 expert traffic is served by CPU/RAM every pass. Any architecture that raises *positions per pass* raises *distinct experts per pass* and therefore the dominant cost; any architecture that raises *accepted tokens per pass* or *VRAM hit rate* lowers cost per token.

## 2. Evidence: where a decode round goes

### 2.1 Tuned production configuration [M]
44.8 tok/s decode at 2.4–3.2 tokens per pass (`docs/DETAILS.md:1294`) → **54–71 ms per pass** [D]. A single-token pass reads **5.10 GB** of weights (exact per-tensor bytes from the GGUF types, `bytes_model.py`: 4.11 GB dense/other + 0.98 GB for the 10 routed experts per layer). If every byte were in VRAM it would take 11.4 ms on the 5060 Ti at its vendor-spec 448 GB/s (9.2 ms for the dense part alone) [D, spec assumed]. Measured passes are **4.7–6.2× above the all-resident floor**, so that floor is not what binds. (An earlier draft of this file used 3.01 GB from an average bits-per-weight; that average hid the 16-bit hyper-connection tensors and ~6-bit trunk and was wrong.)

### 2.2 One instrumented round breakdown [M, different config]
Source: `Strata-data/ncm/cap1/run_00.out` (earlier session, capture build with `--dump-final-r`, 96 decoded tokens, 14-token prompt, 5060 Ti). **Unrepresentative in two ways:** it is a capture build, and draft acceptance was 0.077 (1.08 tokens/round) because the prompts were raw token-id lists. Use it only for the *shape* of the cost.

| Component | ms / round | share of ≈104 ms |
|---|---|---|
| CPU expert pool (gate/up 46.3 + down 24.6) | 71.5 | 69 % |
| waiting for expert rings (PCIe stream) | 22.2 | 21 % |
| adaptive-tier expert swap | 7.4 | 7 % |
| host overhead + commit | 0.1 | <1 % |
| activation quantise, dispatch plan | 0.4 | <1 % |
| (remainder: GPU layers, not itemised) | ~2 | ~2 % |

Round time = 9,258.8 ms / 89 rounds = 104 ms [D]. Window of 4 positions → "CPU experts 13.99 distinct / 15.27 routed per layer", i.e. **92 % of CPU-side routed slots in a 4-token window are distinct experts** — almost no reuse across positions on the missed (cold) experts. Pool throughput reported: 19.3 GB/s.

Consistency check [D]: 13.99 distinct experts × 48 layers × 2.046 MB per expert (exact: gate+up+down bytes ÷ 256 ÷ 48 from the GGUF types) = 1.374 GB; at the reported 19.3 GB/s = **71.2 ms vs 71.5 ms measured (0.4 % off)**. So the CPU-pool time is explained by **bytes of cold experts per round ÷ RAM bandwidth**, with no unexplained residual.

### 2.3 What this does and does not show
- It shows the cost structure when experts live in RAM. It does **not** show the tuned config's split (tuned config uses 2 GPUs, `--expert-cache auto`, `--mmap-experts`, `--pcie-frac 0.25`; its per-component timing was not recorded) **[U]**.
- Throughput in this capture, 10.37 tok/s, ×(2.8/1.08) ≈ 27 tok/s if acceptance matched the code-prompt figure [D, illustration, not a claim].

## 3. Proven and derived bounds

1. **RAM-served expert floor [D, proven given its premise].** If a fraction `m` of the routed-expert bytes `B_e(n)` touched by an n-position pass is read from RAM at bandwidth `W_r`, then `T_pass ≥ m·B_e(n)/W_r`. With near-zero reuse across positions on cold experts (§2.2, 92 % distinct), `B_e(n) ≈ n·B_e(1)` for the cold set, so cold-set time grows ~linearly in n.
2. **Speculation break-even [D, `latency_model.py`].** Speculation wins iff `E[tokens](α,γ) > ρ(γ+1)`, where `ρ(n) = T_pass(n)/T_pass(1)`. At α = 0.77, γ = 3 → E = 2.82. If the pass were fully expert-bound with *no* overlap, ρ(4) = 4 and speculation **loses (0.7×)**; at 50 % overlap ρ = 2.5 (wins 1.2×); at 80 % overlap ρ = 1.6 (wins 1.8×). Strata's measured wins therefore imply substantial expert reuse or a large VRAM-resident share, i.e. **overlap/hit-rate is the key unmeasured variable.**
3. **Uniform-routing coverage [D].** If routing were uniform and independent, a 64-position block touches 92 % of experts per layer (`budget_model.py`), a 128-position block 99.4 %. Real routing is skewed (hence the expert cache and profile), so this is a worst case, not a prediction.
4. **Measured overlap across adjacent positions [M, `routing_stats.py` → `data/routing_stats.json`].** Source: 12 captured prompts, 2,109 positions from the earlier session's routing dumps (ids only; weights were zero). Caveats: assumes in-layer record order is processing order; includes rejected draft positions from a low-acceptance run, which may bias overlap *down*; one prompt set.

   | adjacent positions n | distinct experts / layer (measured) | uniform-random model | ρ(n) if fully expert-bound |
   |---|---|---|---|
   | 1 | 10.0 | 10.0 | 1.00 |
   | 2 | 17.68 | – | 1.77 |
   | 3 | 22.21 | – | 2.22 |
   | 4 | 26.42 ± 0.45 | 37.7 | **2.64** (uniform: 3.77) |
   | 6 | 33.32 | – | 3.33 |
   | 8 | 39.18 | – | 3.92 |

   So routing has real locality (26.4 vs 37.7 distinct at n = 4) but it is far from the 80 % overlap that would make wide blocks cheap. At α = 0.77, a *fully* expert-bound pass gives only ≈ 1.00–1.07× from speculation at γ = 1/2/3 [D]; with the 0.69 CPU-pool share of §2.2 as the expert-bound fraction, ≈ 1.3× [D]. This matches why Strata's MTP gains are real but modest.
5. **Static expert profile [M].** A top-N profile built on 8 files covers on 4 held-out files: top-16 31.3 %, top-32 43.8 %, top-64 57.5 %, top-128 75.1 % of expert activations per layer (uniform: 6.2 / 12.5 / 25 / 50 %). Skew is real and transfers across files, so a VRAM-resident hot set helps; a 75 % hit rate needs half of all experts resident.

## 4. Separating algorithmic limits from implementation defects

| Item | Class | Evidence | Impact |
|---|---|---|---|
| Autotune KV estimate counts 48 layers; only 12 hold per-token K/V | **Implementation defect (planner), fixable** | `pkg/autotune/workload.go:214-266` sets `Layers = block_count = 48` and never reads `qwen4exp.full_attention_interval`; `KVBytesPerToken` = Layers × KVHeads × HeadDim × 2 × bytes → 98,304 B/token fp16 vs true 24,576 B (12 layers). 262 K context: planner 24.0 GiB vs true 6.0 GiB (fp16), 3.4 GiB at the engine's 8-bit default. Cross-check: `docs/DETAILS.md:72` quotes "~13.7 KB of RAM per context token" for 8-bit KV, matching 12 layers × 2 × 256 × 2 = 12,288 elements + scales. | Planner may withhold VRAM for KV that is not needed, leaving fewer expert-cache slots. **Checked in E1b (§8): the engine does not use this estimate for its VRAM split** (the planner only rewrites `--pool-workers`, `--pcie-frac`, `--spec-min-p`, and only when measured); it changes feasibility verdicts and predictions only, and a verdict only at 262 K context. |
| `ActiveParams()` / `TotalParams()` use a generic transformer formula | Implementation defect (planner), minor | They assume dense FFN (4×hidden default) and attention 4·h² every layer, no DeltaNet, hyper-connections, PLE or shared experts. For this geometry ActiveParams ≈ 8.7 B vs true 6.64 B [D]. | Roofline predictions ~30 % pessimistic; low severity. |
| Cold-expert reads from RAM/PCIe | **Algorithmic/placement limit** | §2 | Dominant. Not a bug. |
| Kernel efficiency, occupancy, launch latency, CPU↔GPU sync | **Measured in E1 (§7)**: GPU spends ≈ 65 % of kernel time in `wait_flag_ge` spin kernels waiting for the CPU expert pool / PCIe | `nsys` 2025.6.3 capture, 256-token request; `verify.cpp:1300–1327` | Confirms §2: the GPU is idle-waiting on cold-expert work, not slow at compute |
| Allocation churn, fragmentation, graph compilation | [U] | none recorded | E1 |
| Run-to-run variance | Measured, small | "several percent" (`DETAILS.md:57-59`) | Needs ≥ 5 repeats for any A/B (E0) |

**Rule applied:** no architecture is justified by §4's first two rows; those are fixed (or not) independently.

## 5. The brief's checklist (3.1) against evidence

| Candidate cause | Verdict | Evidence status |
|---|---|---|
| Too many sequential rounds | Not the binding constraint for the teacher: cost per round is bytes-bound, rounds are already cut to ≈ 1/2.4–1/3.2 by MTP | [M] |
| Poor block prediction accuracy | Teacher's own MTP: α ≈ 0.77 code, 0.5–0.63 prose | [M]. No block drafter of NCM type exists to measure. |
| Expensive coarse-to-fine reconstruction | Moot (no generator) | — |
| Number of hierarchy levels / codebook size | Hidden-state codes: 32 bits/position gave 12 % head agreement, 77 % residual NMSE | [M, earlier session] |
| Quantisation errors needing correction | Yes for hidden-state codes (above); unknown for text-token codes | [M]/[U] |
| Train/inference mismatch, exposure bias | Real for any student (§ mathematical_analysis 5.3) | [D] |
| Poor teacher→student transfer | Unmeasured: no student | [U] |
| Capacity allocation / router cost / aux parameters | Aux parameters are < 1 % of tier (P_M ≈ 0.3–0.6 B vs 116.5 B) [D]; router is 31.5 M | [D] |
| Low speculative acceptance | Possible, prompt-dependent: 0.077 in the capture run vs 0.77 on code | [M] |
| Repeated full-sequence processing | NCM re-processes ℓ_k positions per level: Σℓ_k = L(K+1)/2 positions for a linear schedule (e.g. 16.5·L for K = 32) vs L with a KV cache [D] | [D] |
| Context compression/retrieval overhead | Not in the path today | — |
| KV access, attention kernel | Attention reads ≤ 2048 selected tokens/step ≈ 50 MB per step regardless of context (top-2048 indexer, 12 layers) [D]; ≈ 0.7 ms at 70 GB/s-class paths, negligible vs 54–71 ms | [D] |
| Quant/dequant overhead, launch, occupancy, sync | Compute kernels (all non-wait kernels) are ≈ 35 % of the capture span; the largest single compute kernel family is the Q6_K mmvq at 3.6 % of kernel time. No kernel stands out as a defect | [M] §7 |
| Host↔device transfer | Present: ring-wait 21 % of a round | [M] |

## 6. What this means for candidate selection

1. Levers ranked by what the evidence says they move: **(a) accepted tokens per pass** (multiplies every byte read), **(b) VRAM hit rate of experts / cold-set size** (divides the dominant cost), **(c) bits per expert weight** (divides bytes), (d) everything else — only after profiling.
2. NCM-style wide blocks push the wrong way on (b): n positions per pass ≈ n× cold experts when reuse is low.
3. Filter-bank vs RVQ cannot be a deciding factor: it changes the representation, not the bytes read per pass (and showed no token-agreement benefit, `baseline_report.md` §5).

## 7. E1 profile: where the GPU time goes (measured 2026-10-25)

**Method [M].** `nsys` 2025.6.3 (already installed with CUDA 13.2; an earlier note here saying it was absent was wrong), CUDA trace with graph nodes expanded, clean config of E0 on the 5060 Ti, one warm-up request outside the capture, then one request of 1,019 prompt tokens and 256 generated tokens inside it (8.4 s, 30.6 tok/s under profiling, within the E0 range). Files: `data/e1_cuda_gpu_kern_sum.csv`, `e1_cuda_api_sum.csv`, `e1_cuda_gpu_mem_*.csv`. One request only; no repeats.

| Quantity | Value |
|---|---|
| Capture span / kernels launched | 8.34 s / 293,416 kernels, all on device 0 |
| Total kernel time (summed) | 8.35 s |
| of which `wait_flag_ge_kernel` (GPU spin-wait on a host-mapped flag) | **5.40 s = 64.6 %** (17,136 launches, median 1.7 µs, max 8.0 ms, mean 315 µs) |
| Everything else (real compute) | 2.96 s ≈ 35 % of the span |
| Decode part only (after the first 2.3 s) | 6.04 s kernel time, of which 4.02 s = **66.6 %** waiting |
| Prefill part (first 2.3 s) | 2.31 s, of which 1.37 s = 59.4 % waiting |
| Wait time per second of the capture | 0.55–0.79 s of every second |
| Largest compute kernels | Q6_K mmvq 3.6 %, `router_top10` 2.8 % (14,400 × 16 µs), multi-expert mmvq 2.4 / 1.8 / 1.7 %, expert down 1.5 %, attention chunk 1.2 % |
| Host→device copies | 120 MB total in 53 copies (one 118 MB, the KV checkpoint); expert traffic goes through mapped memory and the pool, not `cudaMemcpy` |
| CUDA API time | `cudaStreamSynchronize` 67 % (240 calls, mean 0.78 ms), `cudaGraphLaunch` 20 % (462 calls) |

**What the code says the flag is [V].** `src/core/verify.cpp:1300–1327`: before each layer's MoE the GPU waits on flag A ("the pool published this group's GPU plan"), flag B ("the PCIe share is in staging (DMA) or mapped") and `m_flag_` ("the CPU's share is in the mapped rows"). So the GPU is waiting for the CPU expert pool and the PCIe path to deliver their share of each layer's experts.

**Reading [D].** This is the same bottleneck §2 derived from the one instrumented round, now seen from the GPU side with the production build: about two thirds of GPU time during decode is waiting on cold-expert work done by the CPU and PCIe. Faster GPU kernels cannot help; only reducing or overlapping the cold-expert work can (higher hit rate, fewer distinct experts per pass, more overlap between GPU compute and the CPU pool). The wait kernel has a timeout (`max_ns`); timeout events were not examined **[U]**.

**Limits [U].** One request, 1 K prompt, 256 tokens; spin time may include launch gaps; I did not separate CPU-pool wait from PCIe wait (flags A/B vs `m_flag_`); no CPU-side profile of the pool threads; `ncu` not used because no compute kernel is large enough to matter; the planner-KV half of E1 is not done.

## 8. E1b planner KV check (measured 2026-10-25; no GPU used)

**Question.** Does the 4× KV overcount in `pkg/autotune` change what the engine does or what the plan says?

**Code path [V].** `GeometryFromArgs` reads the GGUF header (`workload.go:214`), `KVBytesPerToken` feeds `Check` (feasibility, `solver.go:271–341`), the cost model (`cost.go:211,254`) and the plan record. `ApplyArgs` (`apply.go:26–38`) is the only thing that edits engine arguments, and it can set only `--pool-workers`, `--pcie-frac` and `--spec-min-p`, each only if that value was measured. Device placement, `--kv-resident` and the expert cache are never set by the planner; the engine sizes its own expert cache at start ("expert cache auto: 9.32 GiB free, 700 MiB reserved … 4424 slots", E0 log) with its own KV geometry.

**Probe [M].** A throwaway Go test (kept as `scripts/autotune_kv_probe_test.go.txt`, not built) loads the real teacher header and runs `Check` for a single-GPU candidate on a synthetic 5060 Ti with 16,000 MiB free, as shipped vs with `Layers = 12`:

| Context | KV as shipped | Verdict as shipped | KV corrected | Verdict corrected |
|---|---|---|---|---|
| 32,768 | 3.00 GiB | OK | 0.75 GiB | OK |
| 131,072 | 12.00 GiB | OK | 3.00 GiB | OK |
| 262,144 | 24.00 GiB | **rejected: needs 25.1 GiB free, has 15.6 GiB** | 6.00 GiB | OK |

Planner values: KV 98,304 B/token (48 layers) vs hybrid-correct 24,576 B/token (12 attention layers); `ActiveParams()` 8.66 B vs 6.64 B true; weight bytes 4.50 GB (not the 58 GB file). The model's default context is 262,144 and the shipped config passes no `--max-context`, so the planner's single-GPU verdict for this model at the default context is "infeasible" although the engine runs fine on that card.

**Conclusion [M/D].**
- The defect does **not** reduce tok/s today: no engine VRAM split depends on it. The E2 hypothesis "fixing it frees VRAM for more cached experts" is **falsified** for the present code.
- It does give a wrong plan at 262 K context (GPU placement rejected), wrong cost-model inputs and a roughly 30 % high `ActiveParams`. The risk is that a future change that does apply the plan (placement, `--kv-resident`) would inherit the 4× error.
- Fix, if the owner wants it: read `qwen4exp.full_attention_interval` (every 4th layer here) in `GeometryFromGGUF` and count only attention layers for KV; add the DeltaNet state (≈ 108 MiB per sequence, inferred) and a test with the real layer pattern. Not implemented: production code is under the approval gate.
- Limits: synthetic capabilities (not live discovery); only single-GPU placement checked; `ctx` is the planner's `MaxContext`, which I took as the model default because the shipped config has no `--max-context`.

## 9. E1 wait split: CPU pool vs PCIe (measured 2026-10-25, from the E1 trace; no new GPU run)

**Method [M].** Each `wait_flag_ge_kernel` launch in the E1 nsys trace (`verify.cpp:1300–1327`) is labelled by the kernel that follows it: `copy_i32_from_mapped` = flag A (pool published the GPU plan), `fetch_blobs` = flag B (PCIe share staged), `copy_rows_from_mapped` = flag C (CPU share in the mapped rows). Every layer shows this A, B, C order in the trace. Script: `scripts/e1_wait_split.py`. Config: `--pool-workers 4`, `--pcie-frac 0.25`, spec 4.

| Phase | Flag | Waits | Sum | Share of all kernel time | Median | p90 |
|---|---|---|---|---|---|---|
| decode | A plan | 3,880 | 7 ms | 0.1 % | 1.6 µs | 2.1 µs |
| decode | B PCIe | 3,880 | 6 ms | 0.1 % | 1.6 µs | 1.8 µs |
| decode | **C CPU share** | 3,880 | **4,009 ms** | **66.4 %** | 807 µs | 2,318 µs |
| prefill | A / B | 1,832 each | 3 ms each | 0.1 % each | 1.6 µs | ≤ 2.1 µs |
| prefill | **C CPU share** | 1,832 | **1,367 ms** | **59.2 %** | 645 µs | 1,472 µs |

**Result.** Essentially all of the wait is flag C, the CPU's share of the cold experts. The PCIe share (B) and the plan hand-off (A) are already done when the GPU reaches them (about 1.6 µs, i.e. the kernel's own launch cost). The "CPU/PCIe" wording in §7 should read "CPU pool". Per MoE layer-group the GPU idles about 0.8 ms median (2.3 ms p90) waiting for the CPU rows, over 3,880 decode layer-groups.

**What this does and does not show.**
- PCIe is not the late party at `pcie_frac 0.25`. It does not show that raising `--pcie-frac` would help: that moves work from the CPU to PCIe and was not tried.
- Flag C also covers time the CPU spends after the GPU has finished its own share; the trace cannot separate CPU compute from CPU-side scheduling and wake-up. That needs a profile of the pool threads (`perf`), which was not run.
- One request, 256 tokens, 30.6 tok/s under profiling; no repeats, so no confidence interval.

**Next measurement (needs the owner's OK for the GPU).** A `--pcie-frac` sweep (0.25, 0.4, 0.5) and a `--pool-workers` sweep with ≥ 5 repeats each, plus `perf` on the pool threads. Gate and margins as in `experiment_plan.md`.

## 10. E1 sweep: `--pool-workers` x `--pcie-frac` (measured 2026-10-25, GPU 0 = RTX 5060 Ti, no Xid)

**Correction first [M].** A first version of this section reported that neither knob changes decode speed (31.3-31.9 tok/s for 2-8 workers). **That was wrong for workers.** `strata serve` runs an autotune layer (`pkg/autotune`, switched off by `-no-autotune`) that rewrote `--pool-workers 4` to `--pool-workers 1` before starting the engine, whatever the config said; the engine log said "1 pool workers" in every one of those runs and the engine command line (read from `/proc`) showed `--pool-workers 1` and an added `--max-context 262144`. So that sweep measured one worker ten times. Its data is kept as `data/e1_sweep_autotuned_1worker.json` (decode 31.3-31.9 tok/s, i.e. the 1-worker figure; only its pcie-frac column is informative). **The E0 baseline (31.7 tok/s), the E1 nsys profile (30.6 tok/s) and §9 were all taken in this 1-worker state.**

**Method [M].** `scripts/e1_sweep.py`, now with `-no-autotune`; after each server is up it reads the engine's real `--pool-workers` and `--pcie-frac` from `/proc` and aborts the config if they differ from the request (all ten matched). One fresh server per config, otherwise the E0 clean config, temperature 0, 256 tokens, ~913-token prompt, unique prefix per request, 1 warm-up + 5 repeats, decode tok/s from the server log, Xid checked after each config, the base config repeated at the end. Data: `data/e1_sweep_noautotune.json`.

| engine workers | pcie-frac | decode tok/s (mean ± sd, n=5) | prefill tok/s | accept |
|---|---|---|---|---|
| 1 (autotune default, from E0) | 0.25 | 31.7 ± 2.5 | 476 | 0.76 |
| 2 | 0.25 | 38.74 ± 1.38 | 443 | 0.78 |
| 3 | 0.25 | 42.94 ± 1.63 | 445 | 0.76 |
| 4 | 0.25 (base) | 43.90 ± 2.44 | 448 | 0.77 |
| 4 | 0 | 44.38 ± 2.25 | 451 | 0.77 |
| 4 | 0.15 | 44.44 ± 1.65 | 448 | 0.76 |
| 4 | 0.40 | 44.42 ± 1.62 | 448 | 0.76 |
| 4 | 0.50 | 44.42 ± 1.62 | 451 | 0.76 |
| 6 | 0.25 | 45.08 ± 1.65 | 451 | 0.76 |
| 8 | 0.25 | 44.70 ± 2.30 | 455 | 0.77 |
| 4 (repeat) | 0.25 | 44.40 ± 1.64 | 448 | 0.76 |

**Result.**
- **Workers matter up to about 4.** 1 -> 2 -> 3 -> 4 workers: 31.7 -> 38.7 -> 42.9 -> 43.9 tok/s. From 4 to 6 or 8 the means differ by under 1.2 tok/s with a standard error of about 0.7-1.0 per config, so no gain is shown; the knee is at 3-4 on this PC. Drift between the first and last base run: 0.5 tok/s.
- **`--pcie-frac` does not matter:** 0 to 0.5 gives 43.9-44.4 tok/s at 4 workers (spread within the noise).
- **The strongest honest baseline on this PC is therefore about 44 tok/s decode** (4+ workers, autotune off), which reproduces the recorded 44.8 tok/s (tuner "search-9", 2026-10-18) to within 2 %. The 31.7 tok/s of E0 was the *default startup path* with its 1-worker override, not the engine's capability.
- **Against the proposed 1.25x gate:** 44.4 / 31.7 = 1.40x, but that compares with the autotuned default and is a configuration defect, not a model improvement. Against the strongest baseline (44.4) the best config (6 workers, 45.1) is +1.5 %, inside the noise.

**Why the planner chose 1 [U].** `ApplyArgs` rewrites `--pool-workers` only if the stored measurement record has a "measured CPU scaling curve" and `BestPoolWorkers` picks "the smallest count reaching 97 % of the measured peak" (`pkg/autotune/cost.go:90`, `apply.go:28`). That 1 came from a stored record is my inference from this code; I did not read the record (`autotune.DefaultRecordPath()`) or check why its curve peaks at 1. This is a candidate bug or stale record to report to the owner; I did not change it.

**What this changes elsewhere.**
- E0 (baseline_report §9) is the 1-worker default path; the clean 4-worker baseline is this table's 43.9-44.4.
- The E1 profile (§7) and wait split (§9) were taken at 1 worker, where the GPU waited 66 % of kernel time for a single CPU worker. At 4 workers the wait is expected to be smaller, but **that is not measured**: the profile must be repeated with `-no-autotune --pool-workers 4` before any statement about where the 4-worker time goes.
- Limits: ~913-token prompts, batch 1, GPU 0, `--spec 4`; no `perf` data (hardware events are denied here: SELinux and `perf_event_paranoid` 2).

## 11. E1 re-profile at 4 pool workers (measured 2026-10-25, nsys, no Xid)

**Why.** §7 and §9 were profiled in the 1-worker state that the autotune layer forced (§10). This repeats the profile with `-no-autotune` and the engine's `--pool-workers 4` verified from `/proc` before profiling. Method otherwise as §7/§9: one warmed request (913-token prompt, 256 tokens, unique prefix) captured with `nsys start/stop` around it, kernels labelled A/B/C by the kernel that follows each wait. Scripts: `scripts/e2_nsys.py`, `scripts/e2_split.py`; output: `data/e2_split.txt`. The trace itself (`.nsys-rep`, 463,793 events) is not committed.

**Request.** 256 tokens in 5,678 ms = **45.1 tok/s under the profiler** (the unprofiled sweep gave 43.9-44.4), prefill 1,950 ms for 913 tokens (468 tok/s).

| Decode phase | 1 worker (§9) | **4 workers** |
|---|---|---|
| wait A (plan) | 0.1 % of kernel time | 0.1 % |
| wait B (PCIe share) | 0.1 % | 0.2 % |
| **wait C (CPU share)** | **66.4 %** (sum 4,009 ms, median 807 µs, p90 2,318 µs) | **51.3 %** (sum 2,929 ms, median 419 µs, p90 1,241 µs, max 3.5 ms) |
| GPU busy, waits excluded | about 35 % of span | **47 % of span** (2,650 of 5,677 ms) |

**Result [M].**
- Four workers roughly halve the median CPU-share wait per layer (807 -> 419 µs) and cut the waiting share from 66 % to 51 % of decode kernel time, which is consistent with the 31.7 -> 44 tok/s gain in §10.
- **The GPU still sits idle about half of the decode time (52 % of the span) waiting for the CPU's share of the cold experts.** The plan hand-off and the PCIe share are done when the GPU reaches them (about 1.5-1.7 µs each), as before.
- No compute kernel exceeds 8.5 % of kernel time (the largest are Q6_K matrix-vector kernels); prefill is dominated by the usual GEMM and dequantisation kernels, with waits only 5.8 % of kernel time.
- Together with §10 this means the critical path on this PC is the CPU expert share: 4 workers help over 1, but 6 and 8 add nothing (45.1 and 44.7 tok/s). So the remaining CPU wait is not removed by more threads. **Why** is not shown by this trace.

**What this does not show.**
- It does not say what the CPU workers are doing during the wait: memory bandwidth or page-cache reads of the mmapped cold experts, a serial part of the hand-off, or scheduling. `perf` is denied here (hardware events: SELinux and `perf_event_paranoid` 2, see §10); a per-thread view needs either that setting changed by the owner or a software-clock `perf` attempt, not done.
- One request of 256 tokens, one profile, no repeats; the wait percentages carry no confidence interval. ~913-token prompt, batch 1, GPU 0, `--spec 4`.
- The profiler slightly changes timing (45.1 under nsys against 43.9-44.4 without).

**Consequence for the plan.** The lever that remains is fewer or cheaper cold-expert reads per pass (higher resident hit rate, 56 % in E0; locality-aware routing, E5; better expert placement), not more threads or a different PCIe split. This is consistent with the bytes-per-pass argument of §3, but no change of this kind has been measured end to end.
