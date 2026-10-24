# 03 — Complexity and hardware

Numbers come from `research/scripts/budget_model.py` and `research/scripts/moe_coverage.py` (rerun to check). Every ASSUMPTION is labelled; none of this is a benchmark.
Hardware: RTX 5060 Ti 16 GB + RTX 4060 8 GB, 60 GB RAM, measured host→device 2.9 GB/s.

## 3.1 Parameter budget (teacher, from the GGUF header) — [V]

| Class | Unique params | Share | Active/token |
|---|---|---|---|
| Routed experts (256/layer, top-10) | 60,397,977,600 | 51.8 % | 2,359,296,000 (10/256 of class) |
| PLE n-gram table (hash lookup) | 51,200,245,760 | 43.9 % | 0 matmul |
| Hyper-connections | 640,624,640 | 0.5 % | all |
| GatedDeltaNet projections | 2,086,510,464 | 1.8 % | all |
| Sparse-indexer attention | 617,358,336 | 0.5 % | all |
| Shared expert / router / PLE small | 236,052,480 / 31,457,280 / 32,839,680 | <0.3 % | all |
| token_embd (row lookup) / output.weight | 635,699,200 / 635,699,200 | 0.5 % each | lookup / all |
| **Total unique stored** | **116,514,464,640** | | |
| **Matmul-active per token** | | | **6,639,838,080 (5.70 %)** |

±2 % band on P_T: 114.18–118.85 B. A "same-budget student" must therefore hold ~114–119 B parameters if the lookup table counts, or ~63–66 B if only matmul'd weights count. Which definition applies is an owner decision (04, G1).

## 3.2 Per-token compute

Dense-equivalent forward cost ≈ 2 × P_active ≈ 13.3 GFLOP/token (ASSUMPTION: 2 FLOP per multiply-add; attention-score cost over context ignored). At 3.2 tok/s measured decode, the engine is bound by weight streaming, not FLOPs. [FLOPs: derived; 3.2 tok/s: measured in docs/distillation_plan.md]

## 3.3 Memory at decode, per sequence

| Item | Value | Status |
|---|---|---|
| KV cache, 12 attention layers, 2 KV heads × (256+256), fp16 | 24,576 B/token → 6.00 GiB at 262,144 ctx (3.00 GiB 8-bit) | Derived; whether `compress_ratio` 4 shrinks it is **Unresolved** (hypothetical 1.50 GiB) |
| GatedDeltaNet state | ≈ 108 MiB, context-independent | ASSUMPTION: 48 heads × 128 × 128 fp32 per layer × 36 |
| Weights | 58.4 GB file; shard1 29.6 GB must be resident, shard2 28.8 GB (n-gram table) can stream from disk | Model card |

## 3.4 Training a same-budget student (arithmetic, ASSUMPTIONS in the script)

- Mixed-precision Adam state ≈ 16 B/param: **1.86 TB** (116.5 B) or **1.03 TB** (65 B). Hardware has 24 GB VRAM + 60 GB RAM.
- Dense 20-tokens/param training ≈ 1.63×10²⁴ FLOP (116.5 B) or 5.0×10²³ (65 B) ≈ **8,600 / 2,700 GPU-years** at 15 TFLOP/s × 40 % on one 4060-class GPU. MoE sparsity reduces FLOPs ~10× but not optimizer state, so the conclusion survives with ~10× fewer years still in the hundreds.
- *Conclusion (conditional on these assumptions):* a same-budget student is not trainable on this machine by orders of magnitude, regardless of architecture. This is Proven-by-arithmetic only if the assumptions hold; relaxing tokens/param to 1 still leaves hundreds of GPU-years.

## 3.5 Multiscale (NCM-style) generation cost vs MoE sparsity

*Analytic, uniform routing assumed (real routing is skewed → lower coverage; not measured):*

| Parallel positions B | Distinct experts touched / layer | Fraction of 256 |
|---|---|---|
| 1 | 10.0 | 3.9 % |
| 16 | 120.7 | 47 % |
| 32 | 184.5 | 72 % |
| 64 | 236.0 | 92 % |
| ≥ 256 | 256 | 100 % |

NCM predicts ℓ_k codes per level in parallel, up to ℓ_K = L positions. For B ≥ 64 a level pass touches ~all experts, so the MoE's bandwidth advantage (reading ~4 % of expert weights per token) disappears: a pass streams ≈ 90 % of the routed weights = ~9.4 s at 2.9 GB/s. With K = 32 levels that is **~5 min per 1024-token block** as a lower bound (ignores compute) — vs. 1024 / 3.2 ≈ 5.3 min for autoregressive decode. **At this bandwidth the parallel scheme gives no speed-up for a MoE that streams from host memory.** [Conditional: needs the weights to be streamed; if all experts fit in VRAM, parallelism does help.] This is the main reason Candidate B (MoE) is not obviously preferable to A (dense) for NCM (05).

## 3.6 Hardware risks

- 5060 Ti has faulted twice with Xid 79 under sustained load (docs/distillation_plan.md); only the 4060 (8 GB) is treated as safe for long jobs.
- 8 GB VRAM cannot hold the teacher's resident shard (29.6 GB); the engine tiers experts through RAM/disk.
- Teacher capture rate (~3.2 tok/s) means 1 M captured tokens ≈ 3.6 days of continuous decode (derived: 1e6 / 3.2 / 86400), 2 MB/position for full layer dumps.

## 3.7 Cost of the two engineering routes

| Route | Work needed | Risk |
|---|---|---|
| Add a student architecture to Strata | New forward graph, new GGUF arch check, protocol changes for block output | High; engine is single-architecture (00 §0.2) |
| Keep teacher; add small drafter + exact verifier | Sampling-exact rejection verifier in `core/verify.cpp` (does not exist); drafter module | Medium |
