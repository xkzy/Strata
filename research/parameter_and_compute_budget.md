# Parameter and compute budget

Labels: **[V]** verified from GGUF headers/code, **[M]** measured/recorded (source), **[D]** derived arithmetic, **[A]** assumption (not measured), **[U]** unknown. All counts reproducible with `scripts/budget_model.py`, `bytes_model.py`, `provenance.py` (outputs in `data/`).

## 1. Parameters

| Item | Value | Source |
|---|---|---|
| Total (all GGUF tensors) | 116.514 B | [V] |
| PLE n-gram table (160 × 320,001,536, IQ4_NL, gather) | 51.200 B (43.94 %) | [V] |
| Routed experts (256 × 48 layers) | 60.398 B (51.84 %) | [V] |
| Excluding PLE | 65.314 B | [D] |
| Active per token (top-10 + shared + dense) | 6.640 B | [D]; upstream card "~6 B" |
| ±2 % band | 114.18–118.84 B (all) · 64.01–66.62 B (ex-PLE) · 6.51–6.77 B (active) | [D] |
| Router | 0.031 B | [V] |
| Hyper-connection | 0.641 B | [V] |
| Upstream (pre-prune) | ≈ 125 B + 51 B n-gram + 4 B MTP | HF card, secondary |

Student budget split for a hypothetical NCM-style same-tier standalone (P_S = P_G + P_E + P_R + P_M): P_E ≈ 60 B, P_R ≈ 0.03 B, P_M ≈ 0.3–0.6 B **[A]**, P_G ≈ the rest. The transform/reconstruction share is < 1 % of the tier, i.e. it is not what decides feasibility.

## 2. Bytes

| Item | Value | Source |
|---|---|---|
| Stored model | 58.40 GB (29.61 + 28.80 GB shards) | [V] |
| Bytes read per decoded token (non-gather) | **5.096 GB** = dense/other 4.114 + routed 0.982 | [D] exact per-tensor types |
| Hyper-connection per token | 1.283 GB = 25.2 % of all bytes | [D] |
| Avg bytes / expert (gate+up+down) | 2.046 MB | [D] |
| All-resident floor per pass | 11.4 ms (5060 Ti at 448 GB/s, vendor spec **[A]**); 18.7 ms (4060 at 272 GB/s **[A]**) | [D] |
| Measured pass time | 54–71 ms (44.8 tok/s at 2.4–3.2 tok/pass) | [M] derived from recorded run |

Stored 58.4 GB vs 24 GB VRAM: at most ≈ 41 % of the file can be VRAM-resident, less after KV, activations and the PLE table placement **[D]**.

## 3. KV and state memory

| Context | KV, hybrid-correct (12 layers, fp16) | Planner estimate (48 layers) |
|---|---|---|
| 32 K | 0.75 GiB | 3.00 GiB |
| 262 K | 6.00 GiB | 24.00 GiB |

24,576 B/token fp16 (12 × 2 × 256 × 2 elements × 2 B); engine default 8-bit ≈ 13.7 KB/token [M, DETAILS.md:72]. DeltaNet state ≈ 108 MiB/sequence, fp32, independent of context [D, inferred shape]. The planner's 4× overestimate (`pkg/autotune/workload.go`) is a verified code defect; its effect on the engine split is **[U]**.

## 4. Compute

| Item | Value |
|---|---|
| FLOP per token (active) | ≈ 2 × 6.64 B = 13.3 GFLOP [D] |
| Roofline at batch 1 | bandwidth-bound: 5.096 GB / 448 GB/s = 11.4 ms vs 13.3 GFLOP at (say) 40 TFLOP/s = 0.33 ms [D, A] → **~30× more bandwidth-limited than FLOP-limited** |
| NCM-style linear schedule | re-processes Σℓ_k = L(K + 1)/2 positions per block (16.5·L at K = 32) vs L with a KV cache [D] |
| Expert coverage, uniform | n = 1: 3.9 %, 4: 14.7 %, 16: 47.1 %, 64: 92.2 %, 128: 99.4 % [D] |
| Expert distinct/layer, measured | n = 1: 10.0, 2: 17.68, 3: 22.21, 4: 26.42, 6: 33.32, 8: 39.18 [M, 12 prompts] |

## 5. Training and labelling cost (all [D] unless stated)

| Item | Value |
|---|---|
| FLOP, 6·N_act·D | 10^9 tokens: 3.98 × 10^19 · 10^10: 3.98 × 10^20 · 10^11: 3.98 × 10^21 |
| GPU-days at assumed 40 TFLOP/s sustained **[A]** | 11.5 · 115 · 1,153 |
| AdamW full state (16 B/param) | 1.86 TB (1.05 TB ex-PLE) vs 24 GB VRAM |
| Teacher scoring at 335 tok/s prefill [M] | 10^7 tokens 0.3 d · 10^8: 3.5 d · 10^9: 34.5 d |
| NCM reference budgets | 13.4 B – 127 B positions for ~100 M params [paper] |
| Student-parameter-efficient (router + MTP head) | trainable ≈ 10^8 → state ≈ 1.6 GB; feasible on one GPU |

## 6. Budget-driven conclusions
1. A same-tier (116.5 B or 65.3 B ex-PLE) student cannot be trained on this PC: ≥ 1 TB optimizer state and ≥ 10^19 FLOP.
2. Only teacher-initialised variants with ≤ ~10^9 trainable parameters and cached teacher logits are feasible, and label generation (0.3–34 days of GPU time for 10^7–10^9 tokens) is the long pole, on a GPU that has a fault history.
3. Per-pass cost is set by bytes, not FLOPs; any budget table that lists only parameters hides the lever.
