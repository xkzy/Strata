# Strata-NCM status

Tests: `cd ncm && python -m pytest -q test_ncm.py` (torch + pytest; 8 pass on CPU at the time of writing). Strata's own Go/C++ suites were **not** run.

## Done and measured

- **Audit:** [architecture_audit.md](architecture_audit.md). The brief's "8B teacher" premise is false for this checkpoint (116.5 B unique parameters, [parameter_budget.md](parameter_budget.md)).
- **Teacher capture works** on an RTX 4060 8 GB: engine patched so `--dump-final-r` writes the verify window's final residual rows on the `--spec` path
  (`src/program/generate.cpp`); [distillation_plan.md](distillation_plan.md). 160 prompts x 256 tokens = ~41k decode positions captured (Strata-data/ncm/cap2, not in git).
  Prompt positions, logits and per-layer states are **not** captured.
- **Stage 2 ablation (final-layer residual, PCA-128):** [ncm_results.md](ncm_results.md). Held-out NMSE 0.75–0.85 vs 1.0 for the train mean; filter bank + RVQ with two codebook sets is
  0.015–0.024 better than RVQ alone at equal code bits, but with a parameter-matched (shared) codebook the advantage disappears (−0.003 to +0.010). Both are far from the unquantized PCA floor (0.557).
- Components with unit tests: lifting filter bank (exact reconstruction with random weights), residual VQ, sparse MoE with capacity and balance loss, provenance-keeping context memory, capture readers.

## Not done

- No coarse-to-fine generator, no MoE-vs-dense comparison on real data, no student, no export, no distillation to teacher logits (logits not available on this path).
- Downstream check done once (engine `--head-eval`, [ncm_results.md](ncm_results.md)): best quantized reconstruction keeps the teacher's next token at 12% of held-out positions (true stream mean alone: 80%). One seed, one split. Not yet run: sampled generation, per-layer states, logits KL.
- No speculative-verification study, no long-context memory evaluation on real data, no runtime integration, no throughput/VRAM/latency benchmarks of any student.
- Docs from the brief still missing: `model_architecture`, `context_memory`, `runtime_integration`, `benchmark_protocol`.
- The ±2% same-budget student is out of scope on this hardware (65.3 B weights without the PLE table).
- Engine build `build-ncm` is sm_89 only (the 4060); it will not run on the 5060 Ti.
- Branch is committed locally and **not pushed**.
