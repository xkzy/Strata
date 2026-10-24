# NCM algorithm analysis

Labels: **[M]** measured here, **[D]** derived from measured numbers, **[S]** from the paper/repo as summarised by a fetcher (not verified line by line), **[U]** unresolved.

## 1. What NCM is [S]
Source: Kumbong et al., arXiv 2610.14162 (text kept untracked in `research/data/ncm_paper.txt`), and the HazyResearch/ncm repository.

**Stage 1, tokenizer.** An encoder maps L tokens to `z ∈ R^{L×d}` (repo default: `wte` + `wpe` lookup, no layers). Level k takes the residual, average-pools it to ℓ_k vectors, picks the nearest entry of a per-level codebook, and adds the linearly upsampled entry back: `ẑ^(k) = ẑ^(k-1) + R_k(e_k)`. A decoder (one block) maps `ẑ^(k)` to token logits. Loss: CE + commitment; EMA codebooks (decay 0.99); nested dropout so any prefix of levels decodes. Example: 16 levels, 1 → 128 codes, L = 512.

**Stage 2, generator.** A decoder-only transformer, frozen tokenizer. For level k it predicts all ℓ_k codes in one forward pass, conditioned on the cumulative reconstruction of levels < k. Targets attend to the full prefix and to their own level. Generation takes K passes (K ≪ L) and produces a fixed length L with no end-of-sequence token.

**Reported [S]:** tokenizer reconstruction 98.79–99.99% (~100M parameters); 4–11× fewer function evaluations and 10–31× faster than diffusion LMs. Not compared against an autoregressive model with a KV cache.

## 2. Why it does not drop onto the Strata teacher
1. **Different object.** NCM quantises text tokens. The earlier prototype in this repo quantised the teacher's final hidden state, which is a different object (its result: 0.06–0.12 top-1 agreement). Neither was a faithful NCM. [M, baseline_report §5]
2. **Recurrent trunk.** 36 of the teacher's 48 layers are causal Gated-DeltaNet. NCM's generator needs bidirectional attention inside a level; a recurrent state cannot take a block of unfinished codes without rollback. [M: arch from the GGUF]
3. **Fixed length, no EOS.** NCM generates exactly L tokens. Strata serves open-ended chat. [S]
4. **Positions are not free here.** NCM's speed argument is that a pass over ℓ positions costs about one pass. On this MoE the marginal position costs about 13 ms against about 15 ms fixed (mathematical_cost_model.md). [M]

## 3. What I tested
| Piece | Result |
|---|---|
| Tokenizer on teacher tokens, teacher-PCA initialised (this repo) | 72–75% held-out reconstruction at 32 codes (C = 4096; 62% at C = 256) after fixing quantiser collapse; coarse levels 7–22%; d = 256 and 20× more text no better |
| Heads-only multi-token prediction from the final state | 0.40 / 0.20 / 0.13 agreement at 1 / 2 / 3 tokens ahead |
| Measured pass time vs window | 15.4 + 13.1 ms × T |
Details: prototype_and_ablation_results.md.

## 4. Open
**[U]** whether a much longer or larger-batch tokenizer run (the quantiser fix, d = 256 and 20× more text did not close the gap) reaches the paper's reconstruction on a 248K vocabulary. **[U]** whether an NCM generator can sit on a recurrent trunk.
