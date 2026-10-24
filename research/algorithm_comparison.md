# Algorithm comparison

Labels: **[M]** measured (source), **[V]** verified from code/headers, **[D]** derived arithmetic, **[L]** literature claim (cited in `baseline_report.md`), **[U]** not measured. Nothing here is a student result; no student exists. Families A–F are the brief's (§4 of the mission); weight-derivation options (third message) are in `weight_provenance.md` §4 and are cross-referenced where they change a row.

## 1. The seven questions, per family

The brief asks seven questions of every candidate: (1) keeps the tokenizer, (2) distillable from this teacher, (3) standalone, (4) meets the ±2 % tier, (5) reduces sequential work or actual computation, (6) efficient on the target hardware, (7) credible evidence. "Credible evidence" below means evidence on this model or one close to it; literature evidence at another scale is marked.

| Family | (1) tokenizer | (2) distillable | (3) standalone | (4) ±2 % tier | (5) less sequential work / compute | (6) target hardware | (7) evidence |
|---|---|---|---|---|---|---|---|
| **A** Improved AR student (distill into a more efficient arch at the same budget; QAT; KV layout; fused kernels) | yes | yes, but the tier forces ≈ 65–117 B trained params → infeasible to train here (1.05–1.86 TB optimizer state [D]) | yes | only if it copies teacher weights (S0–S2, `weight_provenance.md`) | no change to rounds; **bytes/pass can fall** (hc 25 % of bytes) | decode is bandwidth-bound: 5.10 GB/pass vs 13.3 GFLOP [D] | none for a *new* arch; kernel/placement work has measured headroom (4.7–6.2× above all-resident floor [M/D]) |
| **B** MTP + speculative decoding | yes | yes (teacher is its own target) | teacher or student in the loop; exact variants need the teacher as verifier | head is < 1 % of tier | **yes**: 2.4–3.2 tokens/pass [M] | yes, already in Strata | strong on this model: α ≈ 0.77 code, 0.5–0.63 prose [M]; exactness theorems [L] |
| **C** NCM hierarchical generation | no (new discrete codes + tokenizer) | partly: codes are learned from text, teacher gives soft labels only | yes, if trained | tokenizer 36–144 M + generator ≈ P_S−P_M must be trained | fewer *rounds* but each level re-processes Σℓ_k = L(K+1)/2 positions [D] | wide blocks raise distinct experts per pass: ρ(2..8) = 1.77–3.92 [M] | NCM paper at ~100 M scale vs diffusion LMs [L]; hidden-state codes on this teacher recover the token at ≤ 12 % [M, earlier session] |
| **D** Wavelet / filter bank | no | as C | as C | negligible params | none by itself (representation only) | no change to bytes/pass | **no measured benefit over RVQ**: 8 bits/pos plain RVQ 7.8 % vs filter bank + RVQ 6.1 %; 32 bits 12.1 % vs 11.9 % head agreement, 8,192 positions, SE ≈ 0.4 pts [M, earlier session]; NMSE 0.753 vs 0.774 at 32 bits |
| **E** Sparse MoE | yes | the teacher *is* a sparse MoE | yes | experts are 51.8 % of tier | compute-sparse (6.64 of 116.5 B active) but **traffic-sparse only for small n**: 26.4 distinct experts/layer at n = 4 vs 37.7 uniform [M] | cold-expert RAM reads are 69 % of a captured round [M] | measured on this model; no dense-vs-MoE comparison at this budget exists **[U]** |
| **F** Other (semi-AR, SSM, hybrid attention, better distillation) | varies | varies | varies | varies | The teacher is already a hybrid (36 DeltaNet + 12 sparse attention); KV = 24 KiB/token, 6 GiB at 262 K [V] | – | no candidate has evidence on this checkpoint; EAGLE-family drafters (B) are the credible part of F |

## 2. Scoring against the measured bottleneck

The dominant cost of a decode pass is cold-expert bytes read from RAM (`performance_bottleneck_analysis.md` §2; 71.2 ms predicted vs 71.5 ms measured on one captured round, 0.4 % off [D]). Levers that move it: (a) accepted tokens per pass, (b) VRAM hit rate / cold-set size, (c) bytes per weight.

| Family | (a) accepted tokens / pass | (b) cold-set / hit rate | (c) bytes per weight | Training | Quality risk |
|---|---|---|---|---|---|
| A (systems only) | unchanged (2.4–3.2) | planner KV fix may free VRAM **[U]** | unchanged | none | none |
| A (re-quantised hc / trunk) | unchanged | frees VRAM for experts (see §3.2) | **lowers** | none | measurable, nonzero |
| B | **raises** | wider verify raises distinct experts (ρ(4) = 2.64) | unchanged | head only | none (exact) |
| C | unmeasured; no generator exists | works against it | n/a | ≥ 10^19 FLOP [D] | unknown |
| D | no effect | no effect | no effect | as C | unknown; not better than RVQ [M] |
| E (locality-aware router) | unchanged | **raises hit rate** (top-64 profile covers 57.5 % vs 25 % uniform [M]) | unchanged | router (31.5 M params) | measurable KL |
| F | – | – | – | – | – |

## 3. Evidence per family

### 3.1 A. Improved autoregressive student
- Recorded baseline: 44.8 tok/s decode, 335 tok/s prefill on this PC (tuner "search-9", 256 decode tokens, 3,001-token prompt, 2026-10-18) [M, recorded, **not re-run**]. Pass time 54–71 ms vs an 11.4 ms all-resident floor on the 5060 Ti at the vendor-spec 448 GB/s [D, spec assumed].
- Fixable defect: `pkg/autotune/workload.go` counts 48 KV layers instead of 12 → 4× too high KV estimate (24 GiB vs 6 GiB at 262 K, fp16) [V]. Effect on the engine's VRAM split **[U]**.
- Never profiled (`nsys`/`ncu` absent) **[U]**.
- Quantisation-aware options: hyper-connection tensors are BF16 at source, 0.641 B params and **1.283 GB = 25.2 % of bytes read per token** [V/D]. Weight-space error of re-quantising them (12 of 48 layers, symmetric absmax per 32 values with fp16 scale; an approximation of Q8_0/Q4_0, not the engine's code) [M, `hc_requant_probe.py`]: 8-bit 0.61–0.76 %, 6-bit-like 2.5–2.9 %, 4-bit 11.1–12.3 % relative Frobenius error. Function-space error **[U]**.
- Rank truncation of the hc bottleneck: 90 % of Σs_j² needs ≈ 234 of 320 units, 99 % needs ≈ 307 [M, weight-only proxy] — ≈ 25 % reduction at best; the SiLU between down and up blocks any exact merge [V].

### 3.2 Why a byte reduction can pay even if the bytes are "dense and resident"
hc tensors are dense and probably VRAM-resident, so the direct saving is small: 1.283 GB → 0.68 / 0.53 / 0.36 GB at 8 / 6 / 4 bits is 0.60 / 0.76 / 0.92 GB per pass, i.e. 1.3 / 1.7 / 2.1 ms of a 54–71 ms pass at 448 GB/s [D]. The channel that can matter is **VRAM freed for experts**: 0.60–0.92 GB ÷ 2.046 MB per expert ≈ 290–450 experts, i.e. 6–9 of 256 per layer (×48 layers) [D]. The current per-layer resident count is unknown **[U]**; assuming about 64 resident experts per layer, on the top-N coverage curve [M, static profile trained on 8 files, tested on 4] going from top-64 (57.5 %) to top-72 or top-73 adds roughly (75.1 − 57.5)/64 ≈ 0.28 percentage points of activation coverage per added expert → ≈ +2 pts of hit rate for 8 experts, i.e. the cold set shrinks by ≈ 5 % of its bytes [D, linear interpolation of a 4-point curve]. Expected effect on a 54–71 ms pass: ≈ 69 % cold share × 5 % ≈ 3–4 % — **within the run-to-run variance ("several percent"), below the proposed 1.25× gate.** So hc re-quantisation is a candidate for a flag, not a headline win; the brief's rule "reject if smaller but not faster" applies and it stays unproven until E3 measures end-to-end tok/s.

### 3.3 B. Multi-token prediction and speculative decoding
- The teacher ships a 1-block MTP head (116 MB dense + 708 MB experts, 512-expert router, ≈ 2.25 bpw experts) [V]. Measured 2.4–3.2 tokens/pass [M]; the model E[tok] = (1 − α^(γ+1))/(1 − α) at α = 0.77, γ = 3 gives 2.82 [D], inside that range.
- Break-even: speculation wins iff E[tok] > ρ(γ+1). With measured ρ(4) = 2.64 for a fully expert-bound pass, 2.82 / 2.64 ≈ 1.07× [D]; with the CPU-pool share 0.69 as the expert-bound fraction ≈ 1.3× [D]. Raising α is worth more than raising γ: ρ rises ≈ 0.3–0.45 per added position (1.77, 2.22, 2.64 for n = 2, 3, 4; 3.33, 3.92 for n = 6, 8 [M]) while E[tok] saturates at 1/(1 − α) = 4.3.
- Exactness: with draft q and teacher p, accept with min(1, p/q), resample from the residual — output distribution = p (Leviathan et al. 2211.17192; Chen et al. 2302.01318) [L]. Greedy deterministic draft is the case q = δ.
- Published drafters: EAGLE (2401.15077), EAGLE-3 (2503.01840), Medusa (2401.10774), SpecInfer tree verification (2305.09781), Gloeckle MTP (2404.19737), LayerSkip self-speculation (2404.16710) [L]. None measured on this checkpoint **[U]**. Exact vs greedy-equivalent vs approximate are separated in `mathematical_analysis.md` §4.

### 3.4 C. NCM hierarchical generation
- NCM (Kumbong et al., 2610.14162; code HazyResearch/ncm, MIT) [L]: per-dataset tokenizers 36–144 M params, generators ~100 M (725 M on LM1B only), fixed length L 48–1,024, K 8–32 levels, codebooks 4,096–8,192, reconstruction accuracy 98.8–99.99 %, speedups 10–31× **versus diffusion language models** on an H100 at ~100 M scale; Limitations (App. I): fixed K and ℓ_k, separately trained tokenizer, scaling untested. Why advantages may not transfer: baseline (diffusion LMs, not AR+cache or MTP), scale (~100 M vs 116.5 B), sequence length (≤ 1,024), hardware (H100 vs 16 GB + 8 GB consumer GPUs), and no quality-vs-teacher comparison.
- Hierarchy variants the brief lists (fewer levels, unequal sizes, adaptive depth, learned codebook allocation, joint prediction, better conditioning, confidence-aware reconstruction): none can be tested without a trained tokenizer and generator; they are listed in the experiment plan as E6 and not scored here. The number of sequential levels does not predict speed: each level re-processes ℓ_k positions, and the cost per round on this teacher is bytes, not rounds [D].
- Hidden-state compression of *this* teacher [M, earlier session, not re-run]: 32 bits/position leaves 77 % residual NMSE and the teacher LM head recovers the right token at ≈ 12 %.

### 3.5 D. Wavelet / filter bank
- Verdict on the brief's test ("must demonstrate a measurable benefit beyond RVQ"): not met. Mapping verified from the earlier session's code (`train_head.py`: A = RVQ, B = filter bank + RVQ). Plain RVQ is better at 8 bits (7.8 % vs 6.1 % head agreement, ≈ 5 SE) and equal at 32 bits (12.1 % vs 11.9 %). PCA to 128 dims alone gives 29.7 % but leaves NMSE 0.557.
- PR theory and what survives on hidden states: `mathematical_analysis.md` §2.

### 3.6 E. Sparse MoE
- Measured on this teacher: 256 experts, top-10 + 1 shared, expert 2.046 MB; routing skew is real (static top-N profile from 8 files covers 31.3 / 43.8 / 57.5 / 75.1 % of activations for top-16/32/64/128 on 4 held-out files vs 6.2 / 12.5 / 25 / 50 % uniform [M]); adjacent-position overlap is limited (26.4 distinct at n = 4 vs 37.7 uniform) [M].
- Dense vs sparse at equal total or active parameters cannot be compared: no dense student exists, and training one at 65–117 B is infeasible here **[U]**. Not claimed.
- Under a fixed budget, adding experts takes capacity from the backbone — here experts are 51.8 % and the dense trunk 4.2 % of parameters [V], but the trunk carries 4.11 GB of the 5.10 GB read per token.
- Locality-aware router fine-tune (cache-conditional routing, Skliar et al. 2412.00099 [L]): router is 31.5 M params (0.03 % of the tier). Quality cost on this model **[U]**.

### 3.7 F. Other alternatives
- The teacher is already a hybrid: 36 Gated-DeltaNet (2412.06464 [L]) + 12 sparse-attention layers; KV 24 KiB/token fp16 [V]. No further state-space or efficient-attention change is evidenced to help: attention reads ≤ 2,048 selected tokens per step (≈ 50 MB), ≈ 0.7 ms at 70 GB/s [D].
- Semi-autoregressive / block diffusion (2503.09573 [L]) shares the wide-block cost problem of C.
- Distillation objective improvements (on-policy GKD 2306.13649, seqKD 1606.07947 [L]) apply only if some student is trained.

## 4. Ranking (for the architecture decision)

1. **A (systems) + B (speculation)** first: zero quality risk, directly attack measured costs, cheap to test. Proposed gate: ≥ 1.25× end-to-end over the *optimised* baseline (owner to approve).
2. **E (locality-aware router)** if A/B leave cold-expert reads dominant: needs a KL budget.
3. **A (re-quantisation)**: accept only if end-to-end tok/s improves, not merely size (§3.2).
4. **C, D**: no-go at this budget on current evidence; D fails its own stated test.
5. **F**: nothing to schedule.
