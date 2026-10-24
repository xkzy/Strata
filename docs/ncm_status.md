# Strata-NCM status

Tests: `cd ncm && python -m pytest -q test_ncm.py` (torch + pytest; 8 pass on CPU at the time of writing). Strata's own Go/C++ suites were **not** run.

## Done and measured

- **Audit:** [architecture_audit.md](architecture_audit.md). The brief's "8B teacher" premise is false for this checkpoint (116.5 B unique parameters, [parameter_budget.md](parameter_budget.md)).
- **Teacher capture works** on an RTX 4060 8 GB: engine patched so `--dump-final-r` writes the verify window's final residual rows on the `--spec` path
  (`src/program/generate.cpp`); [distillation_plan.md](distillation_plan.md). 160 prompts x 256 tokens = ~41k decode positions captured (Strata-data/ncm/cap2, not in git).
  Prompt positions, logits and per-layer states are **not** captured.
- **Stage 2 ablation (final-layer residual, PCA-128):** [ncm_results.md](ncm_results.md). Held-out NMSE 0.75–0.85 vs 1.0 for the train mean; filter bank + RVQ with two codebook sets is
  0.011–0.024 better than RVQ alone at equal code bits on three splits, but with a parameter-matched (shared) codebook the advantage disappears (−0.007 to +0.016, worse at the lowest rate). Both are far from the unquantized PCA floor (0.557).
- **Head-aware training** ([ncm_results.md](ncm_results.md)): KL through the frozen teacher head lifts an unquantized linear decoder from 30% to 55% next-token
  agreement; quantized codecs stay at 13–14% (3 seeds, 4060). Earlier RVQ runs trained their codebooks by the quantization loss only (fixed in `ncm/train_head.py`).
- Components with unit tests: lifting filter bank (exact reconstruction with random weights), residual VQ, sparse MoE with capacity and balance loss, provenance-keeping context memory, capture readers.

## Not done

- No coarse-to-fine generator, no MoE-vs-dense comparison on real data, no student, no export, no distillation to teacher logits (logits not available on this path).
- Downstream check done once (engine `--head-eval`, [ncm_results.md](ncm_results.md)): best quantized reconstruction keeps the teacher's next token at 12% of held-out positions (true stream mean alone: 80%). One seed, one split. Not yet run: sampled generation, per-layer states, logits KL.
- No speculative-verification study, no long-context memory evaluation on real data, no runtime integration, no throughput/VRAM/latency benchmarks of any student.
- Docs from the brief still missing: `model_architecture`, `context_memory`, `runtime_integration`, `benchmark_protocol`.
- The ±2% same-budget student is out of scope on this hardware (65.3 B weights without the PLE table).
- Engine build `build-ncm` is being rebuilt for sm_89 + sm_120 (the engine refuses PTX JIT on the 5060 Ti).
- Branch `worktree-strata-ncm` is pushed to `origin` (`xkzy/Strata`); `main` is untouched.
