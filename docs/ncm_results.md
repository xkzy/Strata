# Stage 2 result: filter bank + residual VQ vs residual VQ alone, on real teacher residuals

**Outcome: negative at this data scale. Neither method reconstructs held-out teacher residuals better than the training mean, and the
filter bank shows no reliable advantage on held-out data.** This does not show the approach cannot work; it shows 1,160 positions are far too few.

## What was measured

- **Teacher:** the unmodified Strata Coder GGUF (`qwen4exp`, IQ1_M mix), run on an RTX 4060 with the patched engine
  (`--dump-final-r` on the `--spec` path, see [distillation_plan.md](distillation_plan.md)).
- **Data:** 12 prompts (`ncm/make_prompts.py`, v1: code, math, prose, JSON, SQL, a chat turn, numeric recall, a table), tokenized with
  Strata's Go tokenizer (`ncm/tokenize`, round-trip checked), 96 new tokens each: **1,160 decode positions**, all finite, contiguous per sequence.
  Prompt positions are not captured. Raw files: `Strata-data/ncm/cap1/` (47 MB, not in git; SHA-256 prefixes in the JSON).
- **Target:** final-layer residual, mean over the 4 hyper-connection streams -> 2560-d, standardised per dimension with train statistics.
  The stream mean discards per-stream differences (their mean |x| differed 3.4x in the first capture).
- **Split:** sequences 00–08 train (864 vectors, first 96 positions each), 09–11 held out (288 vectors). Prompt types in the held-out set
  (chat turn, numeric recall, table) do not appear in train, so this also tests shift across content types.
- **Methods:** (A) `ResidualVQ`, per position. (B) `LiftingFilterBank` (one level, learned end-to-end through the quantizer) with separate
  `ResidualVQ`s on the coarse (T/2) and detail (T/2) coefficients. 64 codes per level, 400 Adam steps, data-dependent codebook init, seeds 0–2.
- **Metric:** NMSE = ‖x − x̂‖² / ‖x‖² in standardised space; 1.0 equals predicting the train mean. Bits/position = levels x 6 bits, amortised over positions (B's coefficients are half the positions).

## Numbers (mean ± std over 3 seeds; `bench/ncm/stage2_cap1.json`)

| Method | Config | Bits/pos | Train NMSE | **Held-out NMSE** |
|---|---|---|---|---|
| A rvq | K=1 | 6 | 0.733 ± 0.003 | 1.052 ± 0.015 |
| B fb+rvq | K1=K2=1 | 6 | 0.618 ± 0.014 | 1.079 ± 0.012 |
| A rvq | K=2 | 12 | 0.593 ± 0.010 | 1.044 ± 0.016 |
| B fb+rvq | K1=K2=2 | 12 | 0.437 ± 0.013 | 1.018 ± 0.011 |
| B fb+rvq | K1=3, K2=1 | 12 | 0.413 ± 0.003 | 1.045 ± 0.021 |
| A rvq | K=3 | 18 | 0.512 ± 0.015 | 1.044 ± 0.009 |
| B fb+rvq | K1=K2=3 | 18 | 0.348 ± 0.003 | 1.025 ± 0.014 |
| B fb+rvq | K1=4, K2=2 | 18 | 0.327 ± 0.009 | 1.031 ± 0.009 |

## Reading it

- **Held-out NMSE is ≥ 1.0 everywhere.** The codebooks fit the 864 training vectors and carry nothing to new sequences. More levels lower
  only the train error — memorisation.
- **B's lower train error is not evidence for the filter bank.** B has two codebook sets and extra filter weights, so more capacity, and
  it fits the training set better. On held-out data B(2,2) 1.018 vs A(2) 1.044 and B(3,3) 1.025 vs A(3) 1.044 are about one to two seed-stds
  apart, with B(1,1) worse than A(1) and the coarse-heavy variants no better. With 3 seeds that is not a result.
- Possible reasons, **not tested**: too few vectors for 2560 dimensions; per-dimension standardisation giving noise dimensions equal weight;
  the final-layer residual being dominated by prompt-specific content; no dimensionality reduction before quantisation.
- Equal-bits comparison counts only the code stream. Codebook/filter parameters are not counted in the bits and differ between A and B.

## Not claimed

No claim about language quality, generation speed or latency. Nothing here uses the filter bank or VQ in a generator; the student does not exist.
The ±2% parameter-budget goal is out of scope on this hardware ([parameter_budget.md](parameter_budget.md)).

## Next experiment that would be informative

~100x the data (tens of thousands of positions, 100+ varied prompts; at ~13 tok/s decode that is a few hours of capture), a held-out set
from the same content types, a PCA or learned projection to ~128–256 dims before quantisation, and a train-mean and PCA-only baseline.
