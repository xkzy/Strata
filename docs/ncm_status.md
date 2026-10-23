# Strata-NCM status

Run the tests: `cd ncm && python -m pytest -q test_ncm.py` (needs torch + pytest; 5 pass at the time of writing, CPU).

## Implemented and tested (`ncm/`)

- `param_count.py`: exact unique parameter count from GGUF headers. Run against the real teacher (numbers in `parameter_budget.md`). No unit test yet.
- `multiscale.LiftingFilterBank`: learned analysis/synthesis (lifting), even/odd boundary by edge replication. Tested: reconstruction error
  < 1e-10 in float64 with **random** weights, for even and odd lengths. Stability under training and on real hidden states: not tested.
- `multiscale.ResidualVQ`: K-level residual VQ with commitment loss and straight-through gradient. Tested: error is non-increasing with levels; `decode(codes)` equals the quantized output.
- `moe.SparseMoE`: top-k, normalized weights, hard capacity with counted drops, Switch-style balance loss, entropy/utilization stats. Tested: capacity never exceeded, deterministic, balance loss >= 1.
- `memory.ContextMemory`: per-conversation segments with source spans, compressed embeddings, exact text spilled to disk under a RAM budget, invalidation, isolation. Tested on synthetic embeddings only.

These are components. They have **not** been trained, wired together, or run on teacher activations.

## Not done

- Teacher baseline benchmark, dataset pipeline, teacher logit/hidden-state extraction (needs the Strata engine to export them; no such export path exists).
- Any distillation, any trained checkpoint, any student export, any benchmark, any speed or quality claim. **No evidence on whether the multiscale design helps.**
- Filter bank vs RVQ ablation on real data; coarse-to-fine generator; speculative verification; MoE-vs-dense comparison; P_active.
- Strata runtime integration (the engine accepts only `qwen4exp`; see audit), startup hardware probing for the student, GPU/CPU fallback.
- The remaining docs from the mission (`model_architecture`, `distillation_plan`, `context_memory`, `runtime_integration`, `benchmark_protocol`).
- Strata's existing test suite was not run.

## Decision needed from the owner

The brief assumes an 8B teacher and a same-size student. The real teacher is 116.5 B (65.3 B excluding the PLE table), quantized, and
cannot be a PyTorch teacher on this machine. Options: (a) pick a smaller bf16 Qwen3 checkpoint (e.g. 8B) as the research teacher so the
proof of concept is real, keeping the Strata coder as a serving target only; (b) keep the coder as teacher and distill only small replacement
modules against activations captured through the Strata engine (needs a new capture path). Recommendation: (a) for the PoC.
