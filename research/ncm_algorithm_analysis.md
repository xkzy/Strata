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
| Tokenizer on teacher tokens, teacher-PCA initialised (this repo) | 78–80% held-out reconstruction at 32 codes (C = 4096; 62% at C = 256) after fixing quantiser collapse, 20× text and a longer big-batch run; coarse levels 8–16%; d = 256 no better |
| Finite scalar quantisation instead of the VQ (v5) | 73.5% at equal bits (4,096 codes); 79.9% only with 64,000 codes at 31.4 bits/token; coarse levels still 6–27% |
| Rank-coded finest level on the saved v6 model (v7) | code = rank of the true token in the level-5 decoder's logits: 89.1% / 88.0% exact (cap2 / new text) at the same 23.6 bits, against 78.8% / 80.6%; lossless variants cost ≥ 19.2 bits/token (raw id 17.92, unigram entropy 10.65) |
| Direct ranking objective (v8, soft-rank loss λ = 1, matched λ = 0 control) | no effect: level-5 exact 42.9% vs 44.5%, lossless 19.42 vs 19.34 bits |
| Heads-only multi-token prediction from the final state | 0.40 / 0.20 / 0.13 agreement at 1 / 2 / 3 tokens ahead |
| Measured pass time vs window | 15.4 + 13.1 ms × T |
Details: prototype_and_ablation_results.md.

## 4. Open
**[U]** whether anything reaches the paper's reconstruction on a 248K vocabulary. FSQ, a 64,000-code grid, rank coding and a ranking loss all landed at or below 0.80 at ≤ 23.6 bits (0.89 with rank coding at 23.6 bits; 0.98 only at 35.6 bits); the coarse codes narrow the candidates little (median true-token rank 31–106 at levels 1–3), which is what limits the scheme. Single seeds, a small English-chat corpus. **[U]** whether an NCM generator can sit on a recurrent trunk.
