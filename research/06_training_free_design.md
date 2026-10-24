# 06 — A faster model from teacher-derived weights only (no training)

Owner request (2026-10-24): "design faster and smarter ncm model that is better than this local qwen 3.8; use teacher-derived weights if possible so no training is needed."
This is a design; nothing below is implemented or measured unless it cites a measurement.

## 6.1 What is possible and what is not

| Goal | Training-free? | Status |
|---|---|---|
| **Faster, with identical output** to the teacher | Yes, by self-speculation: draft with cheap teacher-derived compute, check with the full teacher | Proven guarantee (greedy-exact today; sampling-exact needs a verifier, 02 §2.5) |
| **Smarter** (higher quality than the teacher) from the teacher's weights alone | **No.** Any model obtained by removing or reusing teacher weights computes a function the teacher already computes, or a degraded one (pruning, fewer experts). There is no training-free operation known to raise quality. | Reasoned, not a theorem; no counterexample known to me |
| Smarter *answers* using more test-time compute (run the code's tests, retry on failure, pick among samples by a checker) | Yes, a harness around the same model | Empirical; the model does not change |
| A true NCM model (multiscale code tokenizer + parallel generator) | **No.** Both NCM parts are learned (NCM Eq. 9, 10). Teacher weights could initialise them but cannot replace training. | Fact about the method |

So the honest version of the request is: **teacher-quality output, faster, no training**. That is self-speculative decoding. NCM's coarse-to-fine idea can only enter as the *shape of the draft*, not as a new model.

## 6.2 What Strata already has (measured in docs/DETAILS.md, not by me)

- The teacher's own **MTP draft layer** proposes up to 3 tokens; one 48-layer pass checks them: **2.4–3.2 tokens per pass** on average (`docs/DETAILS.md:1294`).
- **Prompt lookup** drafts up to 5 tokens copied from the context where measured acceptance pays: code edits 6–11 % faster (`docs/DETAILS.md:1295-1297`).
- Verification is greedy-exact (`core/verify.cpp`), so with `--greedy` the output equals the teacher's.

Any new design has to beat this baseline, not plain one-token decode.

## 6.3 Design: "Strata-Lite draft" — three training-free draft sources, one exact verifier

All three reuse teacher tensors in place: no new weights, no copies, no training.

**D1. Reduced-expert self-draft.** Run the full teacher graph for the draft token, but keep only the router's top-k′ of its own top-10 experts (k′ = 2–4), renormalising the router weights. The shared expert, attention, DeltaNet and head stay as they are.
- Cost: expert reads per draft token drop ~10/k′×. Experts are 51.8 % of stored weights and the part streamed over PCIe (03), so this attacks the actual bottleneck.
- Quality: unknown. Top-k′ truncation of a top-10 router is a lossy approximation; acceptance must be measured (E7).
- Engine change: a router top-k override for the draft pass only (the engine has no such flag today; checked against the flag list in `src/program/generate.cpp`).

**D2. Coarse-to-fine draft tree (the NCM idea, training-free).** NCM goes from coarse to fine over scales. The training-free counterpart: draft a short span cheaply (MTP or D1), then refine only the positions where the draft is unsure (low top-1 probability) with a second, more expensive draft (D1 with larger k′), then verify the whole span once with the full teacher. Coarse = cheap draft over the whole span; fine = more compute only where it is needed.
- Guarantee: the verifier decides, so output is unchanged whatever the drafts do.
- Risk: more draft passes cost time; it pays only if acceptance rises enough (E8).

**D3. Lookup drafts (existing).** Keep prompt lookup for repeated text; choose per step the draft source with the best measured acceptance-per-cost, as the engine already does for lookup.

**Verifier.** Keep the greedy-exact verifier for greedy use. For sampling (temperature 1.0 is the header default), add a rejection-sampling verifier (Leviathan et al.), which needs the draft's probabilities; D1 and MTP produce them as a side effect. Without it, sampled output is not guaranteed to match the teacher.

## 6.4 Why not a bigger parallel block (true NCM width)

Parallel positions touch more experts: ~38 of 256 per layer for 4 positions, ~185 for 32, ~236 for 64 (uniform-routing estimate, `research/scripts/moe_coverage.py`). The verify pass for a long block then streams most experts, and at 2.9 GB/s that costs seconds per pass (03 §3.5). Draft spans should stay short (3–8 tokens). That fits the existing verify window (up to 8 rows).

## 6.5 Expected gain: bounded, not claimed

If a verify pass costs C_v and a draft token costs C_d, and a pass accepts a tokens on average, throughput ≈ a / (C_v + n_draft·C_d). The MTP baseline already gives a = 2.4–3.2. D1 helps only if it gives higher acceptance than MTP at an acceptable C_d. No number for a or C_d exists yet for D1 or D2, so **no speed-up is claimed.**

## 6.6 Experiments (proposed; need approval because they change engine code)

| # | Measurement | Go if (proposed) |
|---|---|---|
| E7 | D1 acceptance vs MTP on 50 code prompts, k′ ∈ {1, 2, 4}, greedy; tokens per pass and wall-clock tok/s, 4060 | tokens/s ≥ 1.10× the MTP-only baseline on the same prompts |
| E8 | D2 (MTP coarse + D1 refine on low-confidence slots) vs best of MTP/D1 | ≥ 1.05× over the better single source |
| E9 | Sampling-exact verifier on a toy target: χ² equality test, 10⁶ draws | equality not rejected at α = 0.01 |
| E10 | Output identity check: greedy output with drafts == greedy output without, 50 prompts × 256 tokens | 100 % identical (any mismatch is a bug) |

Order: E10 harness first (cheap, protects correctness), then E7. If E7 fails, the design reduces to "keep MTP + lookup", which is what Strata ships today.

## 6.7 Bottom line

- A training-free model *better* than the teacher: not achievable by any method I know of; reported as such, not designed around.
- A training-free model *faster* than the teacher with identical output: achievable in principle by self-speculation. Strata already does a version of it (MTP + lookup). The new pieces proposed here (reduced-expert self-draft, coarse-to-fine draft refinement, sampling-exact verifier) are untested, and their gain over the existing baseline is unknown until E7–E10 run.
