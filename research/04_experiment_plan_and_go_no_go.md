# 04 — Experiment plan and go/no-go

**All thresholds below are PROPOSED. They need owner approval before they gate anything.** No experiment here has been run unless stated; nothing in this file is a result.

## 4.1 Decisions the owner must make first

| ID | Question | Why it matters |
|---|---|---|
| G1 | Does "same budget (±2 %)" count the 51.2 B PLE lookup table? | Target is 114–119 B vs 63–66 B (03 §3.1) |
| G2 | Is the goal exact *sampling* equivalence, greedy equivalence, or "good enough quality"? | Decides whether only the speculative route qualifies (02 §2.5) |
| G3 | May the earlier engine patch (`generate.cpp`, `--dump-final-r`, `--head-eval`) stay on the local branch, or be reverted? | It exceeds this brief's "no production changes" rule |
| G4 | Compute budget for any training (GPU-hours, which card) | 4060-only per the fault history; GPU only |
| G5 | Is distilling this checkpoint permitted? Card states Apache-2.0 inherited from the base; the base license text was not read | Legal gate |

## 4.2 Experiments (ordered, cheap first)

| # | Experiment | Hypothesis | Measurement | Cost |
|---|---|---|---|---|
| E0 | Re-run `budget_model.py`, `moe_coverage.py`; re-hash shards | Numbers in 00/03 reproduce | Identical output | Minutes, no GPU |
| E1 | Measure real MoE routing skew on the 41k captured positions (routing dump exists, ids only) and recompute expert coverage per block size | Real coverage at B=64 is below the uniform 92 % | Distinct experts/layer vs B on real routes | Minutes; data already captured |
| E2 | Small NCM reproduction on a public tiny dataset using the authors' code (MIT) at ~100 M | Their reported reconstruction and generation numbers reproduce | Recon accuracy, generation quality vs their table | Hours on the 4060; needs approval for download of code/data |
| E3 | NCM tokenizer on **code** tokens at the teacher vocabulary subset | Recon ≥ 98 % at L=256 | Exact-match reconstruction | Hours |
| E4 | Teacher-vs-student distribution gap of the E2/E3 generator, measured as KL on teacher-scored samples | A KL number exists and is small | KL per token on a held-out set | Needs teacher scoring (3.2 tok/s) |
| E5 | Sampling-exact speculative verifier prototype (CPU/GPU unit tests with a toy target) | Output distribution equals the target (χ² on a toy) | Statistical test on 10⁶ samples | Small code, needs approval to touch `core/verify.cpp`; can be built as a standalone test first |
| E6 | Block drafter acceptance rate against the real teacher with E5 verification | Acceptance ≥ proposed threshold | Mean accepted tokens per teacher pass | Depends on E2–E5 |

Existing partial evidence that bears on E4/E6 (already run, small): next-token agreement through the real LM head, 8,192 held-out positions — true stream mean 79.6 %, best quantized codec 12–14 %; head-aware training lifted an *unquantized* linear decoder 30 → 55 % (docs/ncm_results.md). These concern *compressing the teacher's final state*, a different object from NCM's text tokenizer, and do not transfer directly.

## 4.3 Proposed go / no-go criteria (owner approval required)

| Gate | GO if | NO-GO if |
|---|---|---|
| Training feasibility (same-budget student) | ≥ 10 % of required FLOPs available within 6 months (not true: 03 §3.4) | otherwise → **NO-GO now** |
| E2 reproduction | recon ≥ 98 % and generation within the paper's reported quality | cannot reproduce → stop NCM branch |
| E3 code tokenizer | recon ≥ 98 % exact-match on code at L=256 | < 90 % → NCM unsuitable for code |
| E5 verifier | χ² fails to reject equality at α=0.01 on 10⁶ draws | rejects → fix before anything else |
| E6 speedup | mean accepted tokens/teacher pass ≥ 2.0 and wall-clock gain ≥ 1.3× on the real engine | below → drafter not worth integration |

## 4.4 Stop conditions

Stop and report if: CUDA fails to initialise (no CPU fallback; the owner decides), the teacher hash changes, a result contradicts a claim in 02, or any experiment would exceed G4's budget.

## 4.5 Reproducibility

Every experiment logs: git commit, script, seed, shard hashes, GPU UUID, torch version. Results go to `bench/ncm/*.json` with the command line.
