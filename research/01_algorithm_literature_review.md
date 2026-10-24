# 01 — Algorithm and literature review

Access date for every URL: 2026-10-24. Evidence grade: **Fetched** = I opened the page this session and the claim is on it; **Recalled** = cited from memory, not re-fetched (treat as unverified until checked).
Claim status: **Proven**, **Conditionally proven** (states the condition), **Empirical hypothesis**, **Unresolved**.

## 1.1 NCM (the method under evaluation)

Kumbong, Khare, … Ré, *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*, arXiv 2610.14162v1. https://arxiv.org/abs/2610.14162 — **Fetched** (full text read). Code: https://github.com/HazyResearch/ncm (MIT); checkpoints `hazyresearch/ncm-checkpoints`.

What it is: (1) a multiscale residual-VQ tokenizer maps a token sequence to K levels of discrete codes, level k having ℓ_k codes with ℓ_1 < … < ℓ_K = L; it is trained with nested dropout so each prefix of levels reconstructs the text, then frozen; (2) a generator transformer predicts *all* codes of level k in parallel, conditioned on the cumulative reconstruction ẑ^(k−1) (Eq. 4, 5); output text is the argmax decode of the final reconstruction.

What it demonstrates (as reported by the authors, not re-measured here): tokenizers of 15–144 M parameters, generators ~88–120 M (scaling study 97 M / 317 M / 725 M on LM1B only), sequence length ≤ 1024, K = 12–32, codebook width 384; datasets TinyStories, LM1B, OpenWebText, XSum, WMT14, Sudoku; reconstruction accuracy 98.8–100 %; claims 10–31× speed-up against diffusion LMs.

What it does **not** demonstrate (their App. I and my reading): no code-generation task; no distillation from a large pretrained LLM; no model above 725 M; no sequence above ~1024; fixed K and ℓ_k; separately trained tokenizer; artifacts at coarse transitions; **no claim that the output distribution equals an autoregressive model's.** These gaps are exactly where this project would have to extrapolate. [Fact]

## 1.2 Closest relatives

| Idea | Source | Grade | What it gives us / does not give |
|---|---|---|---|
| Next-scale prediction | VAR, arXiv 2404.02905, https://arxiv.org/abs/2404.02905 | Fetched (abstract) | Coarse-to-fine "autoregression over scales" works for images; text has no natural scale hierarchy, so the analogy is a hypothesis. |
| Residual vector quantization + quantizer dropout | SoundStream, arXiv 2107.03312, https://arxiv.org/abs/2107.03312 | Fetched (abstract) | Variable-bitrate codes by truncating residual stages; audio, not text. |
| VQ-VAE | van den Oord et al., arXiv 1711.00937 | Recalled | Straight-through estimator + codebook/commitment losses; codebook collapse is a known failure. |
| Speculative decoding | Leviathan et al., arXiv 2211.17192, https://arxiv.org/abs/2211.17192 | Fetched (abstract) | **Exact** target distribution is preserved by rejection sampling, given the draft's *conditional* probabilities. This is the one established route to "faster, same distribution". |
| Speculative sampling | Chen et al., arXiv 2302.01318 | Recalled | Same guarantee, independent derivation. |
| Multi-token prediction | Gloeckle et al., arXiv 2404.19737, https://arxiv.org/abs/2404.19737 | Fetched (abstract) | n parallel heads on a shared trunk; the abstract did not state a self-speculative decoding result, so I do not rely on one. |
| Gated DeltaNet | arXiv 2412.06464, https://arxiv.org/abs/2412.06464 | Fetched (abstract; state shape not on the page) | Source of the recurrent-layer family in the teacher. Exact state shape is [Unresolved]. |
| MoE gating | Shazeer et al., arXiv 1701.06538 | Recalled | Sparse top-k routing with load-balancing. |
| Switch Transformer | arXiv 2101.03961, https://arxiv.org/abs/2101.03961 | Fetched (abstract); top-1 routing, capacity factor and auxiliary loss **not** on the abstract → unverified | Same family. |
| Distillation | Hinton et al., arXiv 1503.02531 | Recalled | Soft-target KL. |
| On-policy / generalized KD | GKD, arXiv 2306.13649, https://arxiv.org/abs/2306.13649 | Fetched (abstract) | Training on student-generated sequences reduces train/test mismatch for autoregressive students. |
| MiniLLM | arXiv 2306.08543 | Recalled | Reverse-KL distillation for LLMs. |
| Lifting scheme / wavelets | Mallat 1989; Daubechies & Sweldens 1998 | Recalled | Source of the perfect-reconstruction construction in 02. |

## 1.3 Teacher-side sources (all Fetched)

- Model card, https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF — facts in 00.
- GSQ arXiv 2604.18556, RCO arXiv 2605.00649 — **named on the card; I did not open the papers.** What they claim is therefore not reviewed here.

## 1.4 Comparison of approaches to "a faster standalone model with teacher-level behaviour"

| Approach | Preserves teacher distribution? | Same-budget (±2 %) feasible here? | Evidence base | Status |
|---|---|---|---|---|
| Standalone NCM-style student, distilled from teacher | No guarantee (argmax decode, within-level independence) | No (02, 03) | NCM paper ≤ 725 M, no LLM teacher | Empirical hypothesis |
| Dense autoregressive distilled student | Approximate (KL-minimising); exact only if KL = 0 | No (training compute) | Broad distillation literature | Empirical |
| Speculative decoding with a small drafter + teacher verifier | **Yes, exactly, for sampling** (when the verifier uses rejection sampling) | Yes — the drafter is small; the teacher stays | Proven in the literature | Proven (guarantee), drafter speed-up empirical |
| MTP / self-drafting heads in the teacher | Same as above | Yes | Model family ships an MTP layer | Empirical speed-up |
| NCM-style block drafter + exact verification | Yes if verified with proposal conditionals | Yes | None for LLM scale | Empirical hypothesis |

## 1.5 Search limits

The literature was sampled through abstract pages and the NCM full text, not a systematic survey. I did not search for: text-diffusion LMs beyond what NCM cites, VQ-based text tokenizers other than NCM's, lossy KV-compression papers, or MoE-distillation papers. These are gaps, not negative findings.
