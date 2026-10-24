# Mathematical analysis

Labels: **[P]** proof from stated assumptions (given here), **[S]** standard textbook/paper result (source named; not re-derived), **[D]** arithmetic from measured/verified inputs, **[M]** measured, **[U]** not established. A formula is not a result until its assumptions hold for this model; each section ends with the assumptions that matter.

## 1. Parameter accounting (same-tier question)

Definitions: P_T = unique learned parameters of the teacher; P_S of the exported student; P_A active per step; P_E experts; P_R routers; P_M multiscale transform + reconstruction; P_G backbone and other heads. P_S = P_G + P_E + P_R + P_M, with tied or shared weights counted once; tier condition |P_S − P_T| / P_T ≤ ε, ε = 0.02.

**Values for the teacher [V/D]**: P_T = **116.514 B** (band 114.2–118.8 B). Three ways to report it, because the answer depends on the convention:

| Convention | Value | Band ±2 % |
|---|---|---|
| All tensors in the GGUF files (incl. 51.2 B PLE gather table) | 116.514 B | 114.18–118.84 B |
| Excluding the PLE table (compute-bearing params) | 65.314 B | 64.01–66.62 B |
| Active per token (10 of 256 experts + shared + dense) | 6.640 B | 6.51–6.77 B |

P_E = 60.398 B (51.84 %), P_R = 0.031 B (0.03 %), PLE = 51.200 B (43.94 %). **Open owner decision G1: which convention defines the tier.** Under "all tensors" a student must carry a ~51 B lookup table of its own; under "excluding PLE" it needs ≈ 65 B of compute-bearing weights.

**Proposition 1 [P, conditional].** A standalone student with budget P_S ∈ [0.98, 1.02]·P_T and no teacher backbone must itself contain ≈ 114–119 B parameters that are trained. If P_M ≈ 0.3–0.6 B (the NCM-style tokenizer + reconstruction at the scale NCM reports, 36–144 M params each, scaled), then P_M/P_T < 0.6 % — **the multiscale machinery is not a capacity problem; the cost is that the remaining ≈ 99.4 % has to be learned or copied.** Copying from the teacher is a weight-derivation question (`weight_provenance.md`), not a new architecture.

**Proposition 2 [P].** Training state for AdamW with fp32 master weights and two moments is 16 B/param → 1.86 TB for 116.5 B (1.05 TB excluding PLE); the hardware has 24 GB of VRAM [V]. Training the full tier from scratch or full fine-tuning is therefore infeasible on this PC regardless of algorithm; only parameter-efficient or frozen-teacher updates of ≲ 10^8–10^9 params fit (≈ 1.6–16 GB state) [D].

## 2. Filter-bank reconstruction

Let A: R^{L×d} → subbands, S the synthesis operator.

**Result [S, Vetterli–Kovačević style, not re-verified here]**: a critically sampled M-channel FIR filter bank has perfect reconstruction (SAz = z up to delay) iff its polyphase analysis matrix H_p(z) has a polyphase inverse S_p(z)H_p(z) = z^{−m}I. It is orthogonal iff H_p is paraunitary. Boundary handling (periodic, symmetric extension) changes the matrix at the ends and must be specified.

**Consequences for this task [P/D]:**
1. A *generic learned* strided convolution is not a perfect-reconstruction filter bank; PR is a constraint (paraunitary or an explicit learned inverse), not a property of "wavelet-inspired".
2. For a PR bank, SAz = z holds for **all** z, including hidden states. **PR is not compression:** all subbands together carry the same number of coefficients as z (critical sampling). Compression comes only from dropping or coarsely quantising subbands, and then ‖z − ẑ‖² = energy of what was dropped plus quantisation error (Parseval for orthogonal banks).
3. So the question for hidden states is whether their energy is concentrated in few subbands. **Measured [M, earlier session, not re-run]**: PCA to 128 of 2,560 dims leaves NMSE 0.557 (44 % variance captured); the filter bank + RVQ at 32 bits/pos gives test NMSE 0.753 vs plain RVQ 0.774. The data do not show the concentration that wavelet compression needs.

Assumptions that matter: linear time-invariant bank, stationarity along the sequence axis. Hidden states of a transformer are not stationary signals in position; token IDs are not a continuous signal and no argument here treats them as one.

## 3. Residual quantisation

ẑ = Σ_k R_k E_k(c_k), codes c_k from codebooks of size C_k; r_0 = z, r_k = r_{k−1} − R_k E_k(c_k).

**Bound [P].** ‖z − ẑ‖² = ‖r_K‖². If each stage contracts the residual energy by factor γ_k ≤ 1 (‖r_k‖² ≤ γ_k‖r_{k−1}‖²), NMSE ≤ Π γ_k. A stage with 2^b codewords cannot contract an isotropic Gaussian residual in d dims below γ ≥ 2^{−2b/d} (rate–distortion for a Gaussian source, Cover & Thomas [S]; reverse water-filling for non-isotropic spectra). At b = 32 bits total and d = 2,560 the isotropic bound is 2^{−2·32/2560} = 0.983 [D]: **32 bits per position cannot describe a 2,560-dim hidden state unless its spectrum is concentrated in ≲ tens of directions.** Measured NMSE 0.77 [M] confirms it is not.

Relation to NCM: NCM's latent is learned jointly for reconstruction of text/data tokens (reconstruction accuracy 98.8–99.99 % as reported in the paper [L, arXiv 2610.14162]), not obtained by compressing a frozen teacher's hidden state. Its codes are discrete symbols of a small vocabulary with K = 8–32 levels and 4,096–8,192-entry codebooks (12–13 bits per code); the bits per original position depend on its level lengths ℓ_k and were not compared with the 8–32 bits/position tried here, so no like-for-like rate comparison is claimed.

Assumption: codebook entries are cluster centres (EMA k-means in NCM's `quant.py`); the contraction factor is empirical, not a guarantee.

## 4. Block prediction and exactness

**Within-level independence (NCM Eq 5)**: the generator predicts q(c_k | c_{<k}) = Π_i q_i(c_{k,i} | c_{<k}). **Proposition 3 [P].** For true block distribution p, KL(p ‖ Πq_i) = Σ_i KL(p_i ‖ q_i) + TC(p) where TC(p) ≥ 0 is the total correlation of the block; so a factorised proposal cannot be exact unless the positions are conditionally independent given the conditioning. Parallel decoding therefore approximates p; **exactness needs a verifier.**

**Speculative decoding [S, Leviathan et al. 2211.17192; Chen et al. 2302.01318].** With draft distribution q and target p, accept token x with probability min(1, p(x)/q(x)); on rejection sample from the residual max(0, p − q)/Z. The resulting marginal is exactly p. With a deterministic greedy draft (q = δ_y) the acceptance probability is p(y).
Expected tokens per verify pass with γ drafted tokens and i.i.d. acceptance α: **E[tok] = (1 − α^{γ+1}) / (1 − α)** [S, Leviathan Thm 3.8 form]. Check: α = 0.77, γ = 3 → 2.82 [D], inside the measured 2.4–3.2 [M].

**Break-even [P].** Let ρ(n) = T_pass(n) / T_pass(1). Speculation with γ drafts (n = γ + 1) beats plain decoding iff E[tok] > ρ(n). **Implication [D]:** with measured ρ(4) = 2.64 for a fully expert-bound pass, the ceiling at α = 0.77 is 2.82 / 2.64 = 1.07×; the observed gains imply the pass is only partly expert-bound.

**Standalone vs distribution-preserving is a structural conflict [P].** If no teacher is available at inference (standalone), no verifier exists; the output distribution is the student's. Distribution preservation therefore needs either the teacher at inference or a proof that the student equals the teacher on the deployed support. This is open owner decision G2.

## 5. Sparse MoE: parameter vs compute vs memory traffic

Per layer, top-k = 10 of E = 256, each expert 2.046 MB [V]. For n positions in one pass, distinct experts touched D(n):
- uniform independent routing: E[D(n)] = E(1 − (1 − k/E)^n) → 10.0, 37.7 at n = 1, 4 per layer; 14.7 % coverage at n = 4, 92.2 % at n = 64 [D].
- measured [M, 2,109 positions, 12 prompts]: D(1..8) = 10.0, 17.68, 22.21, 26.42, –, 33.32, –, 39.18.

Bytes per pass = Σ_layers D(n) × 2.046 MB (+ dense 4.114 GB). MoE is computationally sparse (6.64 B active of 116.5 B) but memory-traffic-sparse **only for small n**; wide blocks erase the sparsity. Cold-set time model validated within 0.4 % on the one instrumented round (71.2 vs 71.5 ms) [D/M].

## 6. Perturbation of derived weights

**Linear layer [P].** For y = Wx with W' = W + ΔW: ‖y' − y‖ ≤ ‖ΔW‖₂‖x‖ ≤ ‖ΔW‖_F‖x‖ = ε_F‖W‖_F‖x‖, with ε_F the relative Frobenius error. This is an upper bound and loose (it ignores that x lies in a low-dimensional subspace and that errors average out). The measured weight-space errors (0.6–0.8 % 8-bit, 2.5–2.9 % 6-bit-like, 11–12 % 4-bit on hc tensors) are inputs to this bound, **not function-space results.**

**hc bottleneck [P].** y = U · σ(D x) with σ = SiLU (Lipschitz constant ≈ 1.0998). ‖Δy‖ ≤ ‖ΔU‖₂ ‖σ(Dx)‖ + ‖U'‖₂ · 1.0998 · ‖ΔD‖₂ ‖x‖. Down and up cannot be merged into one matrix because σ is nonlinear, so rank-r truncation of the product is not function-preserving.

**Function-preserving transforms [P].** Permuting experts together with their router rows, permuting hidden units of a bottleneck consistently across down/up, or folding a diagonal scale into the next linear layer are exact. Changing quantisation block layout is exact only if values are unchanged.

## 7. Cost of training a student (order of magnitude) [D]

6·N_act·D FLOP with N_act = 6.64 B: 10^9 tokens = 3.98 × 10^19 FLOP (11.5 GPU-days at an assumed 40 TFLOP/s sustained), 10^10 = 115 d, 10^11 = 1,153 d. NCM's own budgets are 13.4 B (LM1B) – 127 B (OWT) positions for ~100 M-param models, i.e. 134–1,272 positions per parameter [M, paper]; applying those ratios to 116.5 B parameters would be ≈ 1.6 × 10^13 – 1.5 × 10^14 positions (naive scaling, not a claim about what the tier needs). The 40 TFLOP/s figure is an assumption, not a measurement of this PC.

## 8. What is proven, conditionally proven, empirical, unresolved

| Claim | Status |
|---|---|
| PR filter banks exist and are lossless; compression needs energy concentration | Proven (standard theory) [S] |
| Generic learned conv ⇒ wavelet | Rejected (not implied) |
| 32 bits/position cannot describe a flat-spectrum 2,560-dim vector | Proven for isotropic Gaussian [P]; measured NMSE 0.77 on this model's hidden states [M] |
| Factorised block proposals cannot be exact | Proven [P] |
| Speculative decoding is distribution-preserving | Proven [S] |
| Fully expert-bound pass ⇒ speculation ≤ 1.07× at α = 0.77 | Conditionally proven on measured ρ(4) = 2.64, 12 prompts [D] |
| Same-tier standalone student trainable on this PC | Disproved by state size [P] |
| Quality of any weight-derived variant | Unresolved (needs GPU + calibration) |
| NCM at LLM scale / code / long context | Unresolved (no evidence) |
