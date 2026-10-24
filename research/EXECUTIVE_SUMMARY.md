# Executive summary — Strata-NCM research mission

**Verdict (proposed; owner must approve criteria): NO-GO for a same-budget standalone NCM student. REVISE toward a small exact-verification drafter and a ~100 M NCM falsification test.**

| Question | Answer | Status |
|---|---|---|
| What is the teacher? | A local GGUF, `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M` (arch `qwen4exp`): 116.5 B stored params, 48 layers (36 recurrent + 12 sparse-attention), 256 experts top-10, 6.64 B active/token (5.7 %). Not an Ollama tag, not 8 B. | Verified (00) |
| Is a same-budget (±2 %) student feasible? | Not on this hardware: ≥ 1.0 TB optimizer state and ~10²³–10²⁴ FLOP (hundreds to thousands of GPU-years). | Conditionally proven (assumptions in 03) |
| Established guarantees | Lifting is perfectly reconstructing; speculative rejection sampling preserves the teacher's distribution exactly. | Proven (02) |
| Empirical-only | Reconstruction of code at LLM vocab; quality of parallel coarse-to-fine generation at scale; acceptance rates; MoE routing over codes. | Hypothesis (02, 04) |
| NCM generation = teacher distribution? | No guarantee: within-level independence and argmax decode; the paper makes no such claim. | Proven counterexample (02) |
| Parameter / memory budget | 114.2–118.9 B band (or 63–66 B if lookup tables excluded — owner decision G1); KV 6.0 GiB fp16 at 262k ctx; recurrent state ≈ 108 MiB (assumed). | 03 |
| Most promising architecture | Neither A (dense) nor B (MoE) as standalone. Evidence favours keeping the teacher as verifier with a small drafter. B vs A undecidable on current evidence. | 05 |
| Largest risks | Training compute; NCM unproven on code/LLM-teacher/scale; parallel blocks erase MoE sparsity (≈92 % experts touched at 64 positions, uniform-routing assumption); Strata verifies greedy-exact only; GPU faults. | 03, 05 |
| Recommendation | Do not implement. Next: E1 (routing skew), E5 (sampling-exact verifier on a toy), then E2/E3 if the owner wants the NCM test. | 04 |

## Disclosure
Before this brief arrived, the branch already contained work beyond it: a small additive patch to `src/program/generate.cpp` (`--dump-final-r` on the `--spec` path, new `--head-eval`), a teacher capture (~41k positions, outside git), and training experiments on compressed final-layer states. Branch `worktree-strata-ncm` is **local only, not pushed; `main` untouched**. These can be reverted on request.

## What was *not* done
Not read: GSQ/RCO papers, base-model license text. Not run: any NCM reproduction, any sampling-exact verifier. Literature sampled via abstracts plus the NCM full text only. All speed numbers are analytic or from the earlier 3.2 tok/s capture; none are benchmarks of a student.
