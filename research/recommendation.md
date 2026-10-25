# Recommendation

## Verdict
**The current NCM implementation needs a different multiscale prediction objective and a correction, not a drop-in port.** In order:

1. **Correction (the prototype in this repo was the wrong object).** The earlier prototype quantised the teacher's final hidden state and decoded it through the head (0.06–0.12 top-1 agreement). A faithful NCM quantises text tokens. A rebuilt tokenizer on teacher tokens first reached 61–62% held-out reconstruction because its quantiser collapsed; after the fix it reaches 75% at the finest level (C = 4096), with coarse levels at only 7–21% and 23.6 bits per token (a raw token id is ~17.9). Doubling the PCA dimension did not help. Training on 20× more text (627k windows) removed the memorisation (train ≈ test) and lifted the unquantised decoder to 97%; a longer big-batch run then reached 0.784 (cap2 test) / 0.801 (new text) at the finest level, plateauing by step 5,000, with the coarse levels still at 8–16%. Finite scalar quantisation did not beat it at equal bits (0.735; 0.799 only with 64,000 codes at 31.4 bits). That is not the paper's 98.8–99.99%, and 23.6 bits per token is more than a raw token id. [M]
2. **Different multiscale prediction objective.** Predicting several future tokens from the final state does not work: 0.40 / 0.20 / 0.13 agreement at 1 / 2 / 3 tokens ahead, 1.82 tokens per pass at 8. Future tokens must be conditioned on the tokens before them, i.e. on trunk passes. [M]
3. **A different weight-derived mapping is not supported.** Wavelet bases of the dense matrices and weight-space sharing between experts both came out as a random basis would (see teacher_weight_mapping.md). Hidden-unit permutations untested. [M]
4. **Fundamental limit on this model, from the cost side.** On this MoE, every added position costs about 13 ms against about 15 ms fixed (`t = 15.4 + 13.1·T`), so a window of T positions must yield more than `0.68 + 0.576·T` tokens per pass to beat 44 tok/s. NCM's per-level pass cost makes a 6-level hierarchy ≤ 34.9 tok/s even at zero quality loss; a two-level scheme is bounded by 63.7. The 36 recurrent layers also block bidirectional levels. [D]

## Mathematical reason for the slowness
Distinct experts touched per layer grow with positions (10 → 39 from 1 to 8), and expert memory traffic dominates the marginal cost. NCM assumes extra positions are nearly free; here they are not. [M + D]

## What is and is not claimed
No NCM variant has been shown faster than the 44 tok/s baseline. No acceleration is claimed. All bounds are upper bounds at zero quality loss.

## Go / no-go (needs your decision; nothing below was done)
- **No-go:** merging anything into the production engine, or full-scale training. No reproducible advantage exists.
- **Cheap next experiments, if you want to continue NCM:**
  1. The finest level plateaued near 0.79 after quantiser fix, 20× text and 3× more windows seen. FSQ was tried: 0.735 at equal bits (4,096 codes) and 0.799 with 64,000 codes at 31.4 bits/token, so it does not beat VQ at equal bits. A recall@K check (v6) shows the level-5 decoder ranks the true token in its top 4,096 for 89% of positions and the level-6 decoder for 98%, against 79% exact now: a **rank-coded finest level** (code = rank of the true token, same 12 bits) is the next untested change. The unquantised decoder is already at 97%, so the quantised path is what to change.
  2. Run the hierarchical-vs-flat head ablation once (1) works.
  3. Measure expert saturation: pass time at T = 8–32 to test the extrapolation behind the 76 tok/s asymptote. Needs the engine with a window above 6.
  4. Compare experts after hidden-unit matching, or on activations, before ruling out sharing.
- **Alternative with measured headroom:** the baseline itself. Tuning the pool-worker count was worth +39% (31.7 → 44 tok/s) and the autotune default loses it (`performance_bottleneck_analysis.md` §7–§11). That is an engine setting, not NCM, and it still needs your approval to change.

## Unresolved
Owner decisions G1–G6 in `architecture_decision.md`; SHA-256 recompute and licence conflict; the NCM repo summary is secondhand; the data set is small (41k positions, one prompt source).
