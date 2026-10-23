# Distillation plan (option b: Strata coder as frozen teacher, small replacement modules)

Scope after the teacher-size finding ([parameter_budget.md](parameter_budget.md)): the 116.5 B teacher is never trained or copied.
Only small modules are trained against activations captured from the unmodified engine. A same-budget standalone student is **out of
scope** on this hardware; the ±2% criterion does not apply to these modules and is reported as unmet by design.

## Capture path (exists; no engine changes)

`build-fat/strata generate ... --dump-layers L.bin --dump-logits G.bin --dump-routing R.bin` (flags at `src/program/generate.cpp:721-726`).
- Layers: no header, per position `(48+1) x 4 x 2560` float32 = 1,998,848 B/position. Read with `ncm/capture.py` (tested on a synthetic file only).
- Logits: header then rows of `n_vocab` float32 (`:4799-4820`); *header exact layout not yet parsed here*.
- Routing: records of `layer, k, ids, weights`; *record layout not yet parsed here*.
- Constraint from source: `--dump-layers` is written by `session_loop`, so `--no-capture` cannot be used (`:4719`); `--dump-routing` needs the expert pool (`:4650`); `--logits-stride >1` needs `--max-new 1`.
- Cost: ~2 MB/position, so 1 M positions of layer dumps is ~2 TB. Capture a chosen subset of layers (e.g. 12, 24, 36, 48) by post-filtering, and use sequences of a few thousand tokens.

**Not yet run.** Loading the 58 GB teacher needs the engine's expert tiering. This machine's GPU reported
`Unable to determine the device handle for GPU0` and the engine config in the repo was tuned for a 5060 Ti + 4060 pair. A first capture
should be a short prompt (tens of tokens) to validate formats before any dataset run, and should be started by the owner, watching for GPU errors.

## Stages for the modules

1. **Format check.** One short capture; parse logits/routing headers; assert layer-dump size = positions x 1,998,848 B.
2. **Multiscale tokenizer on residuals.** Input: the hc-stream residual (mean over 4 streams first, or a learned stream projection) at one layer, `[T, 2560]`.
   Train `LiftingFilterBank` + `ResidualVQ` for reconstruction + commitment loss only. Metric: relative reconstruction error per level and bits per position vs RVQ-only (the required ablation).
3. **Coarse-to-fine predictor.** Predict next-block codes from the prompt state. Metric: top-1 code accuracy, and agreement of the decoded block with the teacher's greedy tokens. Only this agreement, not code accuracy, says anything about language quality.
4. **MoE head** (`ncm/moe.py`) vs a dense head at equal total and equal active parameters, trained on the same targets.
5. **Verification.** Teacher greedy-agreement of drafted blocks via the existing greedy-exact verify window (`core/verify.cpp`). This is greedy agreement, **not** exact speculative sampling.

Losses in use: reconstruction (MSE), commitment/codebook (in `ResidualVQ`), load balance (in `SparseMoE`), and KL to teacher logits only for stages that output token distributions. No placeholder terms.

## Known limitations

- The teacher is a 1–3 bit quantized checkpoint: captured activations carry quantization error and are not those of the original bf16 model.
- Activations come from a model with hyper-connection residual streams and GDN layers; a module trained on layer-L residuals only works when plugged in at that layer.
- Nothing here yields a standalone model. A real student that runs without the teacher requires a full architecture and a new engine forward graph (see the audit).
