# Architecture decision (for owner approval — nothing implemented)

Labels as in the other documents. The brief's own 12-item list for this document could not be recovered after a context compaction; the 12 sections below cover go/no-go, uncertainties and everything the brief asked this decision to contain, as best I can reconstruct. If an item is missing, say which and I will add it.

## 1. Teacher (verified)
`qwen3.8-flash-next-coder` is a Strata registry name, not an Ollama tag. Files: `Strata-data/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-0000{1,2}-of-00002.gguf` (58.41 GB). Arch `qwen4exp`: 48 layers = 12 × [3 Gated-DeltaNet + 1 sparse attention], hidden 2,560, 256 routed experts top-10 + 1 shared, MTP head, PLE n-gram table, vocab 248,320, ctx 262,144. **116.514 B total, 6.64 B active.** It is an expert-pruned (512 → 256), quantised derivative of `Qwen/Qwen3.8-Flash-Next` (secondary source), so teacher logits are p_T′. Licence conflict (Coder card Apache-2.0 vs upstream `qwen-community-1.0`) is unresolved.

## 2. Same-budget standalone feasibility
**No-go for training**: ≥ 1.05–1.86 TB optimizer state and ≥ 4 × 10^19 FLOP per 10^9 tokens vs 24 GB VRAM [D]. **Structurally possible only by copying teacher weights**, which turns the question into weight derivation (§8).

## 3. NCM / wavelet / sparse-MoE approach: diagnosis
- No trained student exists; nothing is measured for it.
- NCM evidence is ~100 M-param, fixed ≤ 1,024-position blocks, against diffusion baselines; no LLM-teacher, code or long-context evidence [L].
- On this teacher, hidden-state codes at 8–32 bits/position recover the right token at ≤ 12 %; the filter bank adds nothing over plain RVQ [M, earlier session].
- Parallel blocks raise distinct experts per pass (ρ(4) = 2.64, ρ(8) = 3.92 [M]) — the dominant cost.
- Exactness requires a verifier; standalone student cannot preserve the distribution (Prop. 3, 4).
→ **Rejected as the primary route at this budget.**

## 4. Bottleneck
Cold-expert bytes read from RAM dominate (69 % of one captured round; model within 0.4 %). Measured passes are 4.7–6.2× above the all-resident floor [M/D]. Compute is not binding (FLOP time ≈ 3 % of bandwidth time [D, A]).

## 5. Recommended path (owner to approve)
1. **A. Systems**: clean baseline (E0), profile (E1), fix the planner KV defect (E2). Zero quality risk.
2. **B. Exact speculation**: improve acceptance of the MTP head (E4). Zero quality risk; teacher verifies.
3. **D. Locality-aware router** (E5) if A/B leave cold-expert reads dominant. Bounded quality risk (KL ≤ 0.02 proposal).
4. **C. hc/trunk re-quantisation** (E3) only where end-to-end tok/s improves; hc 8-bit would remove ≈ 0.60 GB (11.8 %) of the 5.10 GB per token [D].
The "student" this yields is the teacher with ≈ 0.55–0.6 % derived and ≈ 0.03 % learned weights (S1/S2, `weight_provenance.md` §3), not a new architecture. **Owner decision G6: does that count as the student the mission wants?** If not, the honest answer is that a different standalone architecture is not supported by the evidence.

## 6. Rejected / deferred
| Option | Reason |
|---|---|
| NCM-style generator at 116.5 B | Training infeasible; no evidence; works against the bottleneck |
| Filter bank vs RVQ as a decision factor | No measured token-level benefit; does not change bytes read |
| Merging hc down·up by SVD | Not function-preserving (SiLU) |
| Expert pruning 256 → 192 | Exits the ±2 % tier (101.4 B); second round on an already pruned model; separate Pareto point only |
| Clustering / shared basis | No measured compute reduction; reject unless measured |

## 7. Quality margins and gates
As in `experiment_plan.md` (proposal G3): ≥ 1.25× end-to-end, 100 % token identity for lossless paths, KL ≤ 0.02 nats/token and pass@1 ≥ −2 pts for modified weights, needle equal at 32 K/128 K, stability soak.

## 8. Weight provenance
S1: 99.45 % copied, 0.55 % derived; S2: 99.42 % / 0.55 % / 0.03 % learned. High reuse % is not proof of preserved knowledge; function-space error is unmeasured. See `weight_provenance.md`.

## 9. Go / no-go
| Item | Decision |
|---|---|
| Same-tier standalone NCM student | **NO-GO** |
| Systems fixes (A) | **GO** (E0–E2) after owner OK for GPU runs |
| Exact speculation improvements (B) | **GO** (E4) |
| Locality-aware router (D) | **CONDITIONAL GO** after E0/E1 |
| hc re-quantisation (C) | **CONDITIONAL**, flag-gated, after E3 |
| Smaller distilled model (F) | Separate Pareto study; not started |

## 10. Known uncertainties
1. Live baseline not re-run: 44.8 tok/s / 335 tok/s are recorded values (2026-10-18), prior variance "several percent".
2. Routing statistics: 12 prompts, 2,109 positions, includes rejected-draft positions from a low-acceptance run; weights were zero in dumps.
3. hc probes are weight-space only; no activations, so function error unknown. Re-quant probe uses a simple block quantiser, not the engine's kernels.
4. Trunk (quantised) tensors not dequantised; SVD/pruning sensitivity unmeasured.
5. Hardware: CPU, RAM speed, PCIe link not collected; 448/272 GB/s bandwidths are vendor specs.
6. Planner KV defect's effect on the engine split is unknown.
7. SHA-256 not recomputed; upstream facts from a summarising fetcher.
8. Earlier-session measurements (hidden-state RVQ, head agreement) are cited with provenance but not re-run.
9. GPU: 5060 Ti fault history (Xid 79); every GPU stage needs monitoring.

## 11. Owner decisions needed
G1 tier convention (all tensors / ex-PLE / active). G2 exact distribution vs greedy equivalence. G3 quality margins. G4 GPU permission, compute budget and monitoring. G5 licence resolution before redistributing derived weights. G6 whether S1/S2 counts as the "student".

## 12. Approval statement
No production code, checkpoint, or model file has been modified. No training or GPU work was run in this session. Implementation waits for approval of §5 and §11.
