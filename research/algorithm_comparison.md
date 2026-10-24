# Algorithm comparison

Labels: **[M]** measured (source), **[V]** verified from code/headers, **[D]** derived arithmetic, **[L]** literature claim (cited in `baseline_report.md`), **[U]** not measured. Nothing here is a student result; no student exists.

**Note on the families.** The brief's own A–F list could not be recovered after a context compaction in this session, so the six families below are my enumeration of everything the brief asks to compare, grouped by what they change. If the owner's A–F differ, the evidence rows (§2) still apply; only the grouping changes.

| | Family | What changes | Output distribution vs teacher | Standalone? |
|---|---|---|---|---|
| A | Teacher as-is + systems work | placement, kernels, planner, KV, cache policy; **no weights change** | identical (bitwise up to kernel nondeterminism) | yes |
| B | Better exact speculation on the teacher | new/retrained draft head, tree verification, adaptive depth, lookup drafters; teacher verifies | identical (Leviathan/Chen acceptance) | needs the teacher in the loop |
| C | Weight-derived compression of the teacher | re-quantise, low-rank, expert prune/merge (see `weight_provenance.md`) | approximate; must be measured | yes |
| D | Locality-aware MoE (router fine-tune, cache-conditional routing) | router (31.5 M params) and maybe expert placement | approximate; KL measurable | yes |
| E | NCM-style multiscale discrete-code student (wavelet/filter-bank + RVQ + coarse-to-fine) | new architecture trained from scratch or distilled | approximate | yes |
| F | Smaller distilled standalone student | fewer params, trained from teacher labels | approximate | yes (separate Pareto point) |

## 1. Scoring against the measured bottleneck

The dominant cost of a decode pass is cold-expert bytes read from RAM (`performance_bottleneck_analysis.md` §2, 71.2 ms predicted vs 71.5 ms measured, 0.4 % off [D]). Each family is scored on the three levers that move that cost.

| Family | (a) accepted tokens / pass | (b) cold-set size / VRAM hit rate | (c) bytes per weight | Training needed | Risk to quality |
|---|---|---|---|---|---|
| A | unchanged (≈2.4–3.2 [M]) | planner KV fix may free VRAM for more cached experts **[U]** | unchanged | none | none |
| B | **raises** (α 0.77 code, 0.5–0.63 prose [M] today) | unchanged; wider verify raises distinct experts (ρ(4) = 2.64 [M]) | unchanged | small (head only, ~0.1–0.8 GB) | none (exact) |
| C | unchanged | smaller model → more fits in VRAM | **lowers** (hc 25 % of bytes/token [D]) | none to small | measurable, nonzero |
| D | unchanged | **raises hit rate** (top-64 profile covers 57.5 % vs 25 % uniform [M]) | unchanged | router only | measurable, nonzero |
| E | unmeasured; no trained generator exists | **works against it**: n parallel positions → ρ(n) of 1.77 (n = 2) to 3.92 (n = 8) [M] | n/a | full-scale (≥ 10^19 FLOP, ≥ 1 TB optimizer state at 116.5 B [D]) | unknown |
| F | unchanged | smaller model fits VRAM | lowers | large (10^19–10^21 FLOP [D]) | unknown, likely worse than teacher |

## 2. Evidence per family

### A. Teacher as-is + systems work
- Recorded baseline: 44.8 tok/s decode, 335 tok/s prefill on this PC (tuner "search-9", 256 decode tokens, 3,001-token prompt, 2026-10-18) [M, recorded, not re-run]. Pass time 54–71 ms vs an 11.4 ms all-resident floor on the 5060 Ti at the vendor-spec 448 GB/s [D].
- Fixable defect: `pkg/autotune/workload.go` counts 48 KV layers instead of 12 → 4× too high KV estimate (24 GiB vs 6 GiB at 262 K, fp16) [V]. Whether the engine's VRAM split uses it is unknown **[U]**; test in E1.
- Never profiled (`nsys`/`ncu` absent) **[U]** — kernel, launch and sync costs cannot be ranked yet.

### B. Better exact speculation
- Teacher already ships a 1-block MTP head (116 MB dense + 708 MB experts, 512-expert router, ~2.25 bpw experts) [V]. Measured 2.4–3.2 tokens/pass [M]; the model E[tok] = (1 − α^(γ+1))/(1 − α) at α = 0.77, γ = 3 gives 2.82 [D], inside that range.
- Break-even: speculation wins iff E[tok] > ρ(γ+1). With measured ρ(4) = 2.64 for a fully expert-bound pass, 2.82 / 2.64 ≈ 1.07× [D]; with the CPU-pool share 0.69 as the expert-bound fraction ≈ 1.3× [D]. **Raising α is worth more than raising γ**: each extra drafted position costs distinct experts (ρ rises ≈ 0.3–0.45 per added position: 1.77, 2.22, 2.64 for n = 2, 3, 4, then 3.33 and 3.92 for n = 6, 8 [M]) while E[tok] saturates at 1/(1 − α) (4.3 at α = 0.77).
- Exactness: with a draft q and teacher p, accept with min(1, p/q), resample from the residual — output distribution = p (Leviathan et al. 2211.17192; Chen et al. 2302.01318) [L]. Greedy deterministic draft is the special case q = δ.
- Candidate drafters with published methods: EAGLE (2401.15077), EAGLE-3 (2503.01840), Medusa (2401.10774), SpecInfer tree verification (2305.09781), Gloeckle MTP (2404.19737) [L]. None measured on this checkpoint **[U]**.

### C. Weight-derived compression
Detailed in `weight_provenance.md` §4. Summary of what was measured:
- Hyper-connection (hc) tensors: 0.641 B params, BF16 at source, **1.283 GB = 25.2 % of bytes read per token** [V/D].
- Weight-space error of re-quantising hc (12 of 48 layers sampled, 32-value blocks, fp16 scale, symmetric absmax; an approximation of Q8_0/Q4_0, not the engine's actual formats) [M, `hc_requant_probe.py`]: 8-bit 0.61–0.76 % relative Frobenius error, 6-bit-like 2.5–2.9 %, 4-bit 11.1–12.3 %. The BF16 rounding already present is ≈ 0.11 % rms [D]. **Function-space error is not measured.**
- Rank truncation of the hc bottleneck: 90 % of Σs_j² needs ~234 of 320 units, 99 % needs ~307 [M, weight-only proxy] — only ≈ 25 % reduction is plausible before the proxy gets worse; activation-based check required **[U]**.

### D. Locality-aware MoE
- Skew is real and transfers: a static top-N profile from 8 files covers 31.3 / 43.8 / 57.5 / 75.1 % of activations for top-16/32/64/128 on 4 held-out files vs 6.2 / 12.5 / 25 / 50 % uniform [M].
- Adjacent-position overlap exists but is limited (26.4 distinct at n = 4 vs 37.7 uniform) [M]. A router fine-tuned to raise overlap (cache-conditional routing, Skliar et al. 2412.00099 [L]) could shift both numbers; the cost is output drift, measurable as KL on a held-out set. Router = 31.5 M params (0.03 % of the tier) [V].
- Not measured: how much quality a locality penalty costs on this model **[U]**.

### E. NCM-style student
- NCM (Kumbong et al., 2610.14162; code HazyResearch/ncm, MIT) [L]: per-dataset tokenizers 36–144 M params, generators ~100 M (725 M on LM1B only), fixed block length L 48–1024, K 8–32 levels, codebooks 4,096–8,192, reconstruction accuracy 98.8–99.99 %, speedups 10–31× **versus diffusion language models** on an H100 at ~100 M scale; limitations (App. I): fixed K and ℓ_k, separately trained tokenizer, scaling untested.
- Gaps against this task: no evidence at LLM scale, with an LLM teacher, on code, or at long context; compared against diffusion baselines, not against autoregressive-with-cache or MTP speculation.
- Hidden-state compression of *this* teacher [M, earlier session, not re-run]: 32 bits/position RVQ leaves 77 % residual NMSE and the teacher LM head recovers the right token at 12 % (filter bank 11.9 %, plain RVQ 12.1 %; SE ≈ 0.36 pts) — the filter bank gave no token-level benefit over RVQ.
- Structural cost: an NCM pass over ℓ_k positions per level re-processes Σℓ_k = L(K + 1)/2 positions for a linear schedule (16.5·L at K = 32) vs L with a KV cache [D].
- Same-budget training is infeasible here: AdamW state 16 B/param = 1.86 TB (1.05 TB excluding the PLE table); 10^9 tokens at 6·N_act·D = 3.98 × 10^19 FLOP = 11.5 GPU-days at an assumed 40 TFLOP/s sustained [D]; and the GPU has a fault history (Xid 79) [memory].

### F. Smaller distilled standalone student
- Teacher-label cost at the measured prefill rate (335 tok/s): 10^7 tokens = 0.3 days, 10^8 = 3.5 days, 10^9 = 34.5 days [D].
- Labels come from the *pruned, ~1.9-bit-effective* derivative p_T′, not the upstream model [V]; anything distilled inherits that error.
- Allowed only as a separate Pareto point (brief); not compared at equal tier.

## 3. Ranking (for the architecture decision)

1. **A + B** first: zero quality risk, directly attack the measured costs, cheap to test. Gate: ≥ 1.25× end-to-end over the *optimised* baseline (proposal, owner to approve).
2. **D** next if A/B leave cold-expert reads dominant: needs a KL budget.
3. **C (hc requant, trunk re-quant)** as byte reductions; accept only if end-to-end tok/s improves, not merely size.
4. **E, F**: no-go at this budget on current evidence; revisit only with a funded training budget.
