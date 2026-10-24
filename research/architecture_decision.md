# Architecture decision (for owner approval — nothing implemented)

Labels: **[M]** measured, **[V]** verified from code/headers, **[D]** derived, **[L]** literature, **[U]** not measured. The twelve sections are the brief's required items in its order. Nothing here is a student result: no student has been built, trained or benchmarked, and no GPU was used in this session.

## 1. The exact teacher architecture and verified parameter count
`qwen3.8-flash-next-coder` is a Strata registry name (HANDOFF.md:145), not an Ollama tag (`ollama show` → not found). Files: `Strata-data/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-0000{1,2}-of-00002.gguf` (29.61 + 28.80 GB). SHA-256 recorded by an earlier session, not recomputed: shard 1 `e11083ba…7087fad`, shard 2 `316b46f3…e161e113`.
Arch `qwen4exp`: 48 layers = 12 × [3 Gated-DeltaNet + 1 sparse attention]; hidden 2,560; 24 Q / 2 KV heads, head dim 256; 256 routed experts (pruned from 512) top-10 + 1 shared, expert FFN 640; hyper-connections (4 streams, rank 320); PLE n-gram table 160 × 320,001,536; vocab 248,320; ctx 262,144; MTP head (separate file).
**116.514 B total** (band 114.2–118.8 B), 65.314 B excluding the PLE table, **6.640 B active/token**, bytes read per token 5.096 GB [V/D]. It is an expert-pruned, quantised derivative of `Qwen/Qwen3.8-Flash-Next` (HF cards, secondary); teacher logits are p_T′. Licence conflict (Coder card Apache-2.0 vs upstream `qwen-community-1.0`) unresolved.

## 2. The strongest baseline
The **unmodified checkpoint run by Strata's tuned path with its own MTP head**: 44.8 tok/s decode, 335 tok/s prefill on this PC (tuner "search-9", 256 decode tokens, 3,001-token prompt, 2026-10-18) — **recorded; decode re-measured 2026-10-25 at 43.9-44.4 tok/s with 4 workers and `-no-autotune`, but only 31.7 tok/s on the default `strata serve` path (autotune sets 1 worker)** [M, recorded]. The RTX 5070 benchmark PC figures (58.9 → 42.8 tok/s from 1 K to 262 K context) are another machine. A clean baseline (no control-vector/abliteration flags, ≥ 5 repeats) is experiment E0 and needs the owner's OK to use the GPU.

## 3. Why the current approach underperforms
- No trained NCM-style student exists, so there is no measured student performance to explain; what can be said is why the evidence does not support one.
- **The cost per pass is bytes, not rounds**: 69 % of one captured round was the CPU expert pool, reproduced within 0.4 % by cold-expert bytes ÷ RAM bandwidth [M/D]. Block methods that raise positions per pass raise distinct experts per pass (ρ(2..8) = 1.77–3.92 [M]).
- Hidden-state codes at 8–32 bits/position recover the teacher's token at ≤ 12 %; the filter bank is no better than plain RVQ (worse at 8 bits) [M, earlier session].
- NCM's published speedups are against diffusion LMs at ~100 M scale, ≤ 1,024 positions [L].
- Factorised blocks cannot be exact; exactness needs a verifier, i.e. the teacher (Prop. 3 in `mathematical_analysis.md`).
- Same-tier training state is ≥ 1.05–1.86 TB [D].
- Separately, fixable defects exist that no new architecture should compensate for: planner KV overcount (4×; affects plan verdicts only, E1b). Kernels were profiled (E1): about two thirds of decode GPU time is waiting on CPU/PCIe expert work.

## 4. Candidate algorithms investigated
Brief's families A–F (`algorithm_comparison.md`): A improved AR student / systems / quantisation; B MTP and speculative decoding; C NCM; D wavelet/filter bank; E sparse MoE (incl. locality-aware routing); F other. Plus the weight-derivation options of the third message: direct reuse + quantisation, SVD/low-rank, structured pruning, clustering, function-preserving transforms, sensitivity-driven selection, weight-derived architecture + selective fine-tune, hybrids (`weight_provenance.md` §4).

## 5. Mathematical derivations and assumptions
`mathematical_analysis.md`: parameter accounting (Prop. 1–2), filter-bank PR conditions, RVQ distortion bound (32 bits cannot describe a flat-spectrum 2,560-dim vector), factorised-block inexactness (Prop. 3), speculative decoding exactness and the break-even E[tok] > ρ(γ+1), accepted-prefix model, MoE traffic, perturbation bounds, distillation objective, speed limits. Assumptions that carry weight: iid acceptance (varying α only bounded by Jensen), routing statistics from 12 prompts, vendor-spec bandwidths, assumed 40 TFLOP/s.

## 6. Expected parameter and runtime costs
| Option | Params vs P_T | Bytes / pass | Expected pass-time change | Basis |
|---|---|---|---|---|
| A systems only | 100 % | unchanged | unknown until E1 | [U] |
| B better MTP head | 100 % (+ head) | unchanged | tokens/pass up; ≤ ~1.3× at the measured CPU-pool share | [D] |
| E locality router | 100 % | cold set smaller if overlap rises | ρ(4) 2.64 → target ≤ 2.1 | [U] |
| A hc re-quant 8-bit | 99.99 % by value | −0.60 GB (−11.8 %) | ≈ 3–4 % via VRAM freed; 1.3 ms direct | [D] |
| Expert pruning 256 → 192 | 87.0 % (out of tier) | −25 % expert bytes | not estimated | – |
| NCM-style student | 100 % (trained) | n/a | not estimable; ρ(n) works against it | – |

## 7. Experimental evidence
Measured in this session, all offline (no GPU): header-exact parameter and byte counts; routing overlap on 2,109 positions; static-profile coverage; hc rank and re-quantisation weight-space probes; model-vs-measurement cold-expert check (0.4 %). Recorded, not re-run: 44.8 tok/s / 335 tok/s; the earlier session's RVQ / filter-bank head-agreement runs. **No end-to-end measurement of any derived variant exists.**

## 8. Expected quality trade-offs
A, B: none by construction (exact verification, identical weights). E, hc/trunk re-quantisation, pruning: nonzero and unmeasured; proposed bounds KL ≤ 0.02 nats/token, pass@1 ≥ teacher − 2 pts, needle retrieval equal at 32 K/128 K, 100 % token identity on lossless paths (all owner-approval items). Weight-space error of 8-bit hc re-quantisation is 0.61–0.76 % relative Frobenius error; function-space error unknown.

## 9. The selected algorithm and rejected alternatives
**Selected (pending approval): the best-measured autoregressive path — the teacher itself — improved by systems work (A), exact speculation (B), and, if justified by E0/E1, a locality-aware router fine-tune (E).** The resulting "student" is the teacher with ≈ 0.55 % derived and ≈ 0.03 % learned weights (S1/S2, `weight_provenance.md` §3); this follows the brief's rule "if the best solution is an improved autoregressive student with speculative decoding, select that".
**Rejected:** NCM-style generator at this tier (infeasible training; no evidence; works against the bottleneck); filter bank as a deciding factor (fails the brief's own test: no benefit over RVQ); merging hc down·up by SVD (not function-preserving: SiLU); expert pruning to 192 as a same-tier option (87 % of P_T; separate Pareto point only); clustering/shared basis (no measured compute reduction); smaller distilled model (separate Pareto study, not started).
**Open owner decision:** whether the teacher-derived S1/S2 counts as the "student" the mission wants. If not, the evidence does not support any different standalone architecture.

## 10. Step-by-step implementation plan (after approval)
1. E0 clean baseline with Xid monitoring (owner OK for GPU).
2. E1 profile + planner-KV check; E2 fix on a branch if confirmed.
3. E1b offline routing analysis (accepted-only overlap, co-activation clusters) — no GPU.
4. E3 function-space sensitivity of hc/router/trunk re-quantisation on a calibration set.
5. E4 MTP-head fine-tune / tree verification with exact acceptance; E5 locality router only if steps 1–4 leave cold reads dominant.
6. E7 soak and regression; keep the original model path as the fallback and benchmark baseline.
Full detail, costs and stop criteria: `experiment_plan.md`.

## 11. Go / no-go criteria
| Item | Decision | Criterion |
|---|---|---|
| NCM / filter-bank / same-tier standalone student | **NO-GO** | infeasible training state; no benefit over RVQ; ρ(n) against it |
| Systems fixes (A) | **GO** after GPU OK | gate: ≥ 1.25× end-to-end over the optimised baseline, lower 95 % CI > 1.0×, ≥ 5 repeats |
| Exact speculation (B) | **GO** | α up ≥ 0.05, outputs identical, same gate |
| Locality router (E) | **CONDITIONAL** | ρ(4) down ≥ 20 % at KL ≤ 0.02 |
| hc / trunk re-quantisation | **CONDITIONAL**, flag-gated | KL within margin **and** end-to-end tok/s up; a smaller file alone is rejected |
| Smaller distilled student | not started | separate Pareto study |
The 1.25× gate and every margin are proposals pending approval.

## 12. Known uncertainties
1. Live baseline: re-run (E0, E1 sweep). The default `strata serve` path gives ~31.7 tok/s because the autotune layer sets 1 pool worker; with `-no-autotune` and 4 workers it gives ~44 tok/s, matching the recorded 44.8. Only ~913-token prompts at batch 1 were re-measured; cause of the planner's choice of 1 not investigated.
2. Routing statistics: 12 prompts, 2,109 positions; includes rejected drafts from a low-acceptance run; weights were zero in the dumps.
3. hc probes are weight-space only; the re-quantiser is a simple block quantiser, not Strata's kernels; no activation data.
4. Quantised trunk tensors were not dequantised; SVD/pruning sensitivity unmeasured.
5. CPU, RAM speed and PCIe link not collected; 448/272 GB/s are vendor specs; 40 TFLOP/s is an assumption.
6. Planner KV defect: checked (E1b), it does not change the engine's VRAM split, only plan verdicts at 262 K; resident experts per layer ≈ 92 (4,424 slots, E0).
7. SHA-256 not recomputed; upstream facts from a summarising fetcher; licence conflict.
8. Earlier-session measurements cited with provenance, not re-run.
9. GPU: the 5060 Ti has a fault history (Xid 79); every GPU stage needs monitoring, and none has been run.

## Approval needed (owner)
G1 tier convention (all tensors / ex-PLE / active); G2 exact vs greedy-equivalent output; G3 quality margins and the 1.25× gate; G4 GPU permission and compute budget; G5 licence resolution before redistributing derived weights; G6 whether S1/S2 counts as the "student". No production code, checkpoint or model file was modified.
