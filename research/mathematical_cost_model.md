# Mathematical cost model

Labels: **[M]** measured, **[D]** derived from measured numbers, **[U]** unresolved.

## 1. Measured inputs
Strata server, `-no-autotune`, 4 pool workers, `--spec-min-p 0` (every pass uses the full window), 5 timed runs per row, RTX 5060 Ti, no Xid. [M] (`research/data/e3_pass_time.json`)

| Window T | ms per pass | tokens per pass | decode tok/s |
|---|---|---|---|
| 2 | 42.1 ± 2.6 | 1.67 | 39.7 |
| 3 | 55.8 ± 3.4 | 2.32 | 41.5 |
| 4 | 64.9 ± 3.8 | 2.50 | 38.5 |
| 5 | 81.1 ± 4.8 | 2.76 | 33.9 |
| 6 | 94.9 ± 3.9 | 2.89 | 30.4 |

**Fit [D]:** `t(T) ≈ 15.4 + 13.1·T` ms (least squares on the five rows). So a single-token pass is about 28.5 ms and each extra position costs about 13 ms, 46% of a one-token pass. The window must be at least 2 for the native IQ pack; T = 1 is extrapolated.

Cause [D, consistent with the measured routing]: distinct experts touched per layer grow with positions (10.0, 17.7, 22.2, 26.4, 33.3, 39.2 for 1, 2, 3, 4, 6, 8 positions), and expert traffic dominates the marginal cost. Extrapolation beyond T = 6 is unvalidated [U]: it should flatten as the 256 experts saturate.

Strongest baseline: default adaptive speculation, 44 tok/s decode (~913-token prompt, 256 tokens, n = 5). [M]

## 2. Break-even for any window scheme [D]
A window of T positions beats 44 tok/s if it yields more than `0.044·t(T) = 0.68 + 0.576·T` tokens per pass:

| T | Tokens/pass needed |
|---|---|
| 2 | 1.8 |
| 4 | 3.0 |
| 9 | 5.9 |
| 16 | 9.9 |
| 33 | 19.7 |
Roughly: it must keep 58% of every position it adds. Observed at T = 4 with forced full windows: 2.5.

## 3. Costs for an NCM generator [D]
NCM with K levels of sizes ℓ_1..ℓ_K, each level one trunk pass over ℓ_k positions, producing L tokens:
`time = K·15.4 + 13.1·Σℓ_k ms`.
- Levels 1, 2, 4, 8, 16, 32 (L = 32): 6 × 15.4 + 13.1 × 63 = 918 ms → **34.9 tok/s**, below the 44 baseline even at zero quality loss.
- Two levels (e.g. 4 then 32): 2 × 15.4 + 13.1 × 36 = 502 ms → 63.7 tok/s upper bound.
- Asymptote for any scheme that sends every output position through the trunk once: 1000/13.1 ≈ 76 tok/s [U beyond T = 6].
Not counted: KV prefix reads, head and codebook costs (small next to the trunk), the cost of rolling back recurrent state, and quality loss per level. So these are **upper bounds**, not predictions.

## 4. Verified drafts from heads only [D + M]
From the ceiling pilot, 8 draft tokens give 1.82 tokens per pass; the pass costs 15.4 + 13.1 × 9 = 133 ms → 13.7 tok/s, against 44. The trunk cost of verifying a block makes small acceptance expensive.

## 5. Crossover
There is no crossover for the measured quality: the break-even needs ~58% acceptance per added position, and the best measured heads-only agreement is 40% at one token ahead, 20% at two. [D + M]
