# Experiment plan

Status: **nothing below has been run.** Stages are ordered by cost; each has a predeclared pass/fail criterion. GPU stages need owner approval and Xid monitoring (the 5060 Ti has fallen off the bus before; the 4060 has been stable). No stage trains on CPU. No stage modifies the original checkpoint or production Strata code; engine changes happen on a branch only after approval.

Margins below are **proposals for the owner to approve or change** (G3); none were given in the brief.

| Margin | Proposed value | Applies to |
|---|---|---|
| Throughput gate | ≥ 1.25× end-to-end decode tok/s over the *optimised* baseline (E0/E1), same prompts, ≥ 5 repeats, lower 95 % CI above 1.0× | any change |
| Lossless paths | 100 % token identity at temperature 0 on 200 prompts × 512 tokens (allowing documented float-order differences ≤ 0.1 %) | A, B |
| Modified weights | KL(p_T ‖ p_S) ≤ 0.02 nats/token on held-out text and code; pass@1 on a held-out code set ≥ teacher − 2 pts (95 % CI) | C, D |
| Long context | needle retrieval equal to teacher at 32 K and 128 K | all |
| Stability | no NaN/Inf, no Xid event, peak VRAM within 95 % of capacity over a 30-minute soak | all |

## Stages

| ID | Stage | Needs GPU | Cost | Question | Pass / stop criterion |
|---|---|---|---|---|---|
| E-1 | Offline analysis on existing dumps (done in part) | no | done | expert skew, overlap, bytes | – (see `performance_bottleneck_analysis.md`) |
| E-1b | Accepted-only expert overlap from routing dumps; expert co-activation clusters | no | hours | is overlap higher on accepted tokens? can experts be laid out contiguously? | report numbers; informs D |
| E0 | Clean live baseline: no control-vector/abliteration flags, ≥ 5 repeats, 1K/4K/32K/128K contexts, prefill + decode, batch 1 and 2, peak VRAM, KV, load time, logit dump for 20 prompts; capture CPU model, RAM size/speed, PCIe link width | yes (5060 Ti + 4060) | ~2 h | what is the real, current baseline? | CV < 5 %; otherwise fix measurement before any A/B |
| E1 | Profile + planner check: `nsys`/`ncu` (install if absent), per-kernel and per-sync breakdown; test whether the 48-layer KV estimate changes the engine's VRAM split (set `--kv-resident` by hand vs planner) | yes | ~4 h | where do the 54–71 ms go on the tuned config? | identifies ≥ 1 non-bandwidth cost > 5 % of pass, or confirms bandwidth dominates |
| E2 | Planner KV fix on a branch (family A) | no | small | **E1b showed the engine ignores the planner estimate, so no tok/s gain is expected;** fix only as a correctness bug, with a unit test using the real layer pattern | tests pass; no throughput claim |
| E3 | Activation-based sensitivity: perturb hc (8/6/4-bit, rank cut), router, trunk per layer; measure KL and routing agreement | yes | ~1 day | which derived transforms stay within the 0.02-nat margin? | a transform passes only if KL ≤ margin **and** end-to-end tok/s improves (not just size) |
| E4 | MTP-head fine-tune / alternatives (B): retrain or replace head on teacher-labelled data; acceptance α by task; tree vs chain verification | yes | days (labels: 10^7 tokens = 0.3 d at 335 tok/s) | does α rise from 0.77 / 0.5–0.63? | α up by ≥ 0.05 with identical outputs (exact verification) and passes the 1.25× gate |
| E5 | Locality-aware router (D): fine-tune router (31.5 M params) with an overlap/hit-rate penalty; measure D(n), hit rate, KL | yes | days | does overlap rise enough to pay for the KL cost? | ρ(4) down ≥ 20 % at KL ≤ 0.02 |
| E6 | Offline NCM-style probe (only if owner wants E revisited): train a small tokenizer + generator on a *tiny* tier (≤ 100 M params, ≤ 10^9 positions) with teacher text, compare to MTP-style drafter at equal params | yes | weeks | does the approach beat a drafter at small scale, on code? | otherwise stop; do not scale |
| E7 | Soak + regression on the chosen variant | yes | 1 day | stability, memory, long context | all margins above |

## Decision tree
- If E0/E1 show bandwidth dominates and no defect > 5 %: go B (E4) and D (E5) only.
- If E1 finds a launch/sync cost > 15 % of a pass: fix that first (A); it is the cheapest lever.
- If E3 shows hc 8-bit holds KL ≤ 0.005: ship as an option behind a flag only if E0-style repeats show the pass shortens (hc bytes = 11.8 % of traffic at 8-bit; expected ≤ 1.7 ms of 54–71 ms **[D]**).
- NCM-style generator (E6): not scheduled without owner approval and a funded training budget.

## Safety
- Run `nvidia-smi` and `dmesg | grep -i xid` before and after every GPU stage; abort on any Xid. Do not use the 5060 Ti for long unattended runs until the fault cause is understood (memory: GPU fault history).
- All derived artifacts go to a new directory with a provenance manifest (`weight_provenance.md` §5); originals read-only.
