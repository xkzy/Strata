# Stage 2 results: filter bank + residual VQ vs residual VQ alone, on real teacher residuals

## v2 (160 prompts, ~41k positions, PCA to 128-d): the quantizers generalize, and the filter bank is consistently a little better

**Outcome:** held-out error is now below the train-mean baseline (0.75–0.85 vs 1.0), and at equal bits the filter bank + RVQ beats plain RVQ
by about 0.015–0.024 NMSE on held-out data at every level count with seed std ≤ 0.006. **But every quantized variant is far worse than the
unquantized PCA floor (0.557), and the comparison does not isolate the multiscale structure from extra capacity** (see caveats). No
claim about language quality.

Setup (changes from v1 below): 160 prompts, seeded (`ncm/build_prompts_v2.py`: 50 Dolly, 30 OpenAssistant first turns, 30 WikiText
paragraphs, 30 snippets from this repo's code, 20 arithmetic / number-recall prompts), 256 new tokens each, the same unmodified teacher and
patched engine. 40,960 decode positions cut into non-overlapping windows of T=128: **256 train / 64 held-out windows**; every 5th prompt of
each source is held out (split in `bench/ncm/prompts_v2_manifest.json`), so held-out prompts are the same content types as train, different prompts.
Target: stream-mean final-layer residual, standardised with train statistics, projected on the top-128 train PCA directions; NMSE is measured
back in the standardised 2560-d space (so it includes the PCA truncation). 256 codes/level (8 bits), 600 Adam steps, seeds 0–2 (mean ± std).
Raw data: `Strata-data/ncm/cap2/` (not in git); results: `bench/ncm/stage2_cap2.json`.

| Method | Config | Bits/pos | Train NMSE | **Held-out NMSE** |
|---|---|---|---|---|
| train mean | – | 0 | 1.000 | 1.000 |
| PCA only (no quantizer) | R=128 | 4096 | 0.491 | **0.557** |
| A rvq | K=1 | 8 | 0.802 ± 0.016 | 0.853 ± 0.016 |
| B fb+rvq | 1+1 | 8 | 0.773 ± 0.002 | 0.838 ± 0.003 |
| A rvq | K=2 | 16 | 0.745 ± 0.003 | 0.809 ± 0.001 |
| B fb+rvq | 2+2 | 16 | 0.712 ± 0.001 | 0.791 ± 0.002 |
| B fb+rvq | 3+1 | 16 | 0.705 ± 0.001 | 0.789 ± 0.001 |
| A rvq | K=3 | 24 | 0.723 ± 0.006 | 0.793 ± 0.006 |
| B fb+rvq | 3+3 | 24 | 0.684 ± 0.002 | 0.769 ± 0.001 |
| B fb+rvq | 4+2 | 24 | 0.679 ± 0.005 | 0.768 ± 0.003 |
| A rvq | K=4 | 32 | 0.700 ± 0.003 | 0.774 ± 0.003 |
| B fb+rvq | 4+4 | 32 | 0.665 ± 0.001 | 0.753 ± 0.001 |
| B fb+rvq | 5+3 | 32 | 0.660 ± 0.002 | 0.752 ± 0.001 |

What the data supports:
- **Generalization:** train-to-held-out gaps are ~0.05–0.09 (v1: held-out was *worse* than the mean). More data and PCA fixed the memorization.
- **Filter bank vs RVQ at equal code bits:** B is lower on held-out at all four budgets (−0.015, −0.018, −0.024, −0.021). At 8 bits/pos the gap
  is within A's seed spread; at 16–32 bits it is several seed-stds. Coarse-heavy and even splits of the bits are indistinguishable.
- **Both are far from the floor:** best quantized held-out NMSE is 0.752 at 32 bits/position vs 0.557 for PCA with no quantization, so the code
  discards most of what a 128-d projection keeps. The final-layer residual is high-entropy; about half of its standardised energy (49% train, 56% held-out) is outside the top 128 PCs.

What it does not support:
- **The gain is not attributable to multiscale structure.** B has two codebook sets plus filter weights, so about twice the codebook parameters
  of A at the same code bits; "equal bits" counts only the code stream. A parameter-matched A (or B with one shared codebook) was not run.
  The filter's temporal prediction/update may be exploiting correlation between neighbouring positions, which is the hypothesis, but this
  experiment does not separate it from capacity.
- **NMSE of the final residual is not model quality.** Nothing here feeds the reconstruction through the LM head to check that the teacher's
  next token survives. That is the metric that matters and has not been measured.
- Held-out windows come from 32 prompts of the same five sources as train; no unseen-domain test. Only decode positions (the teacher's own
  continuation, which sometimes includes thinking-style tokens); prompt positions are not captured. Stream-mean discards per-stream information.
- The 5th-prompt split leaves 64 windows; the seed std reflects optimiser variance, not data variance (one split only).

Next experiments, in order of value: (1) quantize then run the teacher's head on the reconstruction and report next-token agreement vs the
unquantized and PCA-only residual (needs the LM head weights in Python or an engine hook); (2) a parameter-matched baseline; (3) a second
random split to get data variance; (4) more levels / larger codebooks to see how far toward the PCA floor the code gets.

---

## v1 (12 prompts, 1,160 positions, no PCA): negative

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
