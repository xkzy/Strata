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
| Autotune KV estimate counts 48 layers; only 12 hold per-token K/V | **Implementation defect (planner), fixable** | `pkg/autotune/workload.go:214-266` sets `Layers = block_count = 48` and never reads `qwen4exp.full_attention_interval`; `KVBytesPerToken` = Layers × KVHeads × HeadDim × 2 × bytes → 98,304 B/token fp16 vs true 24,576 B (12 layers). 262 K context: planner 24.0 GiB vs true 6.0 GiB (fp16), 3.4 GiB at the engine's 8-bit default. Cross-check: `docs/DETAILS.md:72` quotes "~13.7 KB of RAM per context token" for 8-bit KV, matching 12 layers × 2 × 256 × 2 = 12,288 elements + scales. | Planner may withhold VRAM for KV that is not needed, leaving fewer expert-cache slots. **Whether the engine uses this estimate for its VRAM split is [U]**; the shipped config sets `--kv-resident` by hand. Test in E1. |
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
