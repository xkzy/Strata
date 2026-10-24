# 02 — Mathematical feasibility

Each statement is labelled **Proven**, **Conditionally proven**, **Empirical hypothesis** or **Unresolved**. "Proven" means a short argument given here, not a formal machine-checked proof.
Notation: teacher T with distribution p_T(x_t | x_<t); student S; sequence length L; K levels, ℓ_1<…<ℓ_K=L; codebook sizes |V_k|.

## 2.1 Perfect reconstruction of the multiscale transform

**Lifting scheme.** Split a signal into even/odd samples; d = o − P(e); s = e + U(d). Inverse: e = s − U(d); o = d + P(e).
- *Proven:* inverse is exact for **any** functions P, U (even nonlinear, learned), in exact arithmetic. In floating point the inverse is exact to rounding because the same P(e), U(d) are recomputed, but not bit-exact in general across devices.
- *Proven:* it is critically sampled: L/2 + L/2 outputs for L inputs; no redundancy added.
- *Proven (counterexample):* a generic strided convolution followed by a different learned decoder is **not** perfect-reconstruction; PR is a property of the lifting structure, not of "multiscale".
- *Unresolved / empirical:* PR does not imply useful decorrelation. Wavelet-style energy compaction is a property of specific P, U on smooth signals; hidden states of a language model are not smooth in token order. Whether learned P, U compact them is an empirical question.

## 2.2 Residual vector quantization

- *Conditionally proven:* if every codebook contains the zero vector (or the nearest-code choice may be "no correction"), the residual norm is non-increasing across stages: ‖r_{k+1}‖ ≤ ‖r_k‖. Without that condition, adding a stage can *increase* error.
- *Proven:* capacity bound. K stages with codebooks of size |V_k| over ℓ_k positions carry at most Σ_k ℓ_k log₂|V_k| bits. Lossless coding of a source needs at least its entropy; text at ~1–2 bits/char (LM-scale entropy estimates, order of magnitude only) bounds the rate from below.
- *Empirical hypothesis:* NCM reports 98.8–100 % reconstruction on ≤ 1024-token sequences at 15–144 M tokenizer sizes. Reconstruction of arbitrary text is possible by that route at that scale; nothing here shows it for code, long contexts, or large vocabularies (teacher: 248,320 tokens).

## 2.3 Exact reconstruction ≠ exact generation

*Proven.* A tokenizer that reconstructs every sequence exactly says nothing about whether a generator over its codes reproduces p_T. Generation needs p_S(codes) whose pushforward through the decoder equals p_T(x). Two ways it fails even with lossless reconstruction:

1. **Within-level independence (NCM Eq. 5).** If level-k codes are predicted independently given ẑ^(k−1), the joint is a product. *Counterexample:* target joint over two positions is "AB" or "BA" with probability ½ each; any product distribution that matches both marginals (½,½) assigns ¼ to AA and BB, which the target gives zero. The failure is zero only if the codes within a level are conditionally independent given the prefix — an assumption on the data, not a property of the architecture. Status: **Proven** (counterexample); how often it bites on code is **Empirical**.
2. **Deterministic decode.** NCM's output is the argmax decode of ẑ^(K). A deterministic map from codes to text cannot reproduce sampling variance unless the variance lives in the codes. So the model is at best distributionally correct over codes, and the paper makes no such claim. Status: **Proven** (as a statement about what the paper defines); effect size **Unresolved**.

## 2.4 Greedy agreement ≠ distributional equivalence

*Proven.* Top-1 agreement on held-out positions does not bound total-variation or KL distance between the two distributions; two distributions can share argmax everywhere and differ arbitrarily in the tail. Conversely, small KL implies high top-1 agreement only if the teacher's top-1 margin is large. The only guarantee linking a student to the teacher's *sampling* distribution is verification by rejection sampling (2.5), or KL → 0.

## 2.5 Speculative verification: what is exactly guaranteed

- *Proven (Leviathan et al., 2211.17192):* sampling a draft token x ~ q and accepting with probability min(1, p_T(x)/q(x)), otherwise resampling from the normalised max(0, p_T − q), yields exactly p_T. Requires q(x) for the drafted token (the proposal's conditional probability).
- *Conditionally proven:* a block drafter whose joint is a product Π q_i(x_i | prefix) is a valid proposal for sequential verification, as long as the accept/reject steps use those q_i and reject-at-first-mismatch semantics. NCM's argmax decode is a point mass; it is valid only with q=1 on the drafted token (accept prob = p_T(x)), which preserves exactness but with a low acceptance rate when p_T is spread.
- *Gap (Strata):* the existing verify window is **greedy-exact only** (`core/verify.cpp`): it accepts a draft iff it equals the teacher's argmax. That preserves *greedy* output exactly, not sampled output. A sampling-exact verifier is not implemented. [V by reading, not run]

## 2.6 Memory and capacity limits

- *Proven (pigeonhole):* a student with B bits of parameters cannot store more than 2^B distinguishable arbitrary facts; "same-budget" (±2 % of P_T parameters) bounds capacity from above, and quantization to fewer bits lowers B. The teacher itself is already compressed (pruned 512→256 experts, ~3.5 bpw).
- *Proven:* lossy compression cannot recall arbitrary facts beyond its capacity; retrieval from the source restores exactness only when the right source is retrieved.
- *Conditionally proven:* a student of the same architecture class and the same budget *can represent* the teacher (copy it). Matching it through training is what is infeasible (03).

## 2.7 What distillation can and cannot guarantee

- *Proven:* if KL(p_T‖p_S) = 0 on all contexts, S = T in distribution. Reaching 0 is not guaranteed by any finite training; there is no generalisation guarantee off the training distribution.
- *Empirical hypothesis:* a student with fewer *active* parameters can approximate a MoE teacher on a distribution (compute-matched distillation). Evidence for this at this size and for code is not in hand.
- *Unresolved:* whether the teacher's quantization noise (IQ1_M mix) is a good or bad target. The student learns the quantized model's function, not the BF16 one.

## 2.8 Tied/untied output and the head (relevant to my experiments)

*Proven (by construction):* teacher `output.weight` is a separate 248,320×2560 matrix; recovering tokens from final-layer states is a fixed linear map + argmax, so any compression of the final state must be judged *through that head*, not by reconstruction error. My measurement (docs/ncm_results.md): a state estimate with normalized MSE ≈ 0.75–0.8 gives next-token agreement of only 12–30 % (true state: 100 %; true mean stream: 79.6 %). *Empirical:* MSE is a poor proxy for the quantity that matters.

## 2.9 Summary table

| Claim | Status |
|---|---|
| Lifting is perfectly reconstructing for any P, U | Proven (exact arithmetic) |
| Generic strided/learned-conv multiscale is PR | Disproven by counterexample |
| RVQ error non-increasing | Conditionally proven (zero in codebook) |
| Product-over-positions joint can represent correlated tokens | Disproven (counterexample) |
| NCM output matches p_T | Unresolved; paper makes no such claim |
| Greedy agreement ⇒ same distribution | Disproven |
| Speculative rejection sampling preserves p_T | Proven (literature) |
| Strata verifies sampling-exactly today | Not true (greedy-exact only) |
| Same-budget student trainable on this hardware | Disproven by arithmetic in 03 (assumptions labelled) |
| NCM-style tokenizer reconstructs code to 100 % at LLM vocab | Unresolved |
