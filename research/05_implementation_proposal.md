# 05 — Implementation proposal (a proposal only; nothing here is implemented)

Budget target: P_T = 116,514,464,640 ± 2 % (114.18–118.85 B) if lookup tables count (G1 in 04); otherwise 64.68 B ± 2 %.

## 5.1 Candidate A — same-budget dense multiscale student

- ~116 B dense parameters, NCM-style multiscale tokenizer + parallel coarse-to-fine generator.
- Per-token compute: 116.5 B / 6.64 B = **17.5× the teacher's active compute** (derived), and all weights are read every pass.
- Training: ≈ 1.6×10²⁴ FLOP at 20 tok/param (03 §3.4). Not feasible here.
- Parallel positions are cheap *relative to a dense AR model* (weights read once per level), so the NCM speed argument holds better for A than for a MoE — but only if 116 B of weights fit in fast memory. They do not (24 GB VRAM).
- Evidence: NCM paper, ≤ 725 M. Extrapolation of 160× in size. **Empirical hypothesis.**

## 5.2 Candidate B — same-budget sparse MoE multiscale student

- ~116 B total, ~6.6 B active (teacher-like routing: 256 experts, top-10), multiscale generator over the codes.
- Training FLOPs ~17× lower than A, optimizer state unchanged (~1.9 TB). Still not feasible here.
- Parallel levels touch ≥ 92 % of experts once B ≥ 64 positions (uniform-routing assumption, 03 §3.5): the sparsity benefit at inference vanishes exactly where NCM wants large parallel blocks.
- Routing during a coarse-to-fine pass is untested; load-balancing and expert specialisation over *codes* (not tokens) are unknown.
- Evidence: none for MoE + NCM. **Empirical hypothesis, weaker than A on the speed side, stronger on the training-compute side.**

## 5.3 Recommendation (by evidence, not by preference)

Neither candidate is justified for implementation. Ranked recommendation:

1. **Do not build a same-budget standalone student** (both A and B fail on training feasibility alone; this is arithmetic, not a judgement).
2. **Cheapest informative step:** E1 (real routing coverage) then E5 (sampling-exact verifier on a toy), both small, GPU-optional, no engine change needed for E5-as-standalone-test.
3. If the owner wants to test the NCM idea itself: E2/E3 at ~100 M on the 4060 as a falsification test of "NCM tokenizes code", before any distillation work.
4. Integration route if E5/E6 pass: small drafter + teacher kept as verifier inside Strata (preserves the teacher's sampling distribution exactly; speed-up empirical).

Between A and B, if forced: **B** for training/inference compute per useful parameter, **A** for the NCM parallel-block speed argument. The evidence cannot separate them; E1 and a toy MoE-over-codes experiment would.

## 5.4 What implementation would touch (if ever approved)

`gguf_reader.hpp` (architecture check), `qwen_hybrid.hpp` (constants), a new forward graph, `orchestrator.go` (block output), `registry.go` (fix wrong coder numbers first). No production file is changed by this proposal.

## 5.5 Risks

| Risk | Severity |
|---|---|
| Training compute is 3+ orders of magnitude beyond the hardware | Fatal for A/B as standalone |
| NCM never shown on code, large vocab, long context, or LLM teacher | High |
| Within-level independence cannot represent correlated tokens (02 §2.3) | High for code |
| MoE bandwidth advantage erased by parallel blocks | High for B |
| Hardware faults (Xid 79) | Medium; no CPU fallback allowed |
| Teacher is quantized and pruned; student learns its errors | Medium |
| License: base-model text not read | Unknown |
