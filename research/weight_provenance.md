# Weight provenance and weight-derived transformations

Labels: **[M]** measured, **[V]** verified from GGUF headers/code, **[D]** derived arithmetic, **[U]** not measured. Parameter counts are exact from the GGUF headers (`scripts/provenance.py` → `data/provenance_output.txt`). Provenance categories for a candidate are **design claims about what it would do**; no candidate has been built. **High reuse % is not evidence of preserved knowledge** — it only says how many weights are bitwise or rule-derived from the teacher.

Categories: **C** copied unchanged · **D** derived deterministically from teacher weights (no training) · **M** mapped into a changed architecture · **L** learned (trained or fine-tuned).

The original checkpoint (`Strata-data/models/coder-IQ1_M/*.gguf`, 2 shards, 58.41 GB) is never modified; every derived artifact would be a new file with its own manifest (§5).

## 1. Parameters by role, whole teacher [V]

| Group | Params (B) | % of 116.514 B | Bytes read per decoded token |
|---|---|---|---|
| Routed experts (256 / layer, top-10) | 60.398 | 51.84 | 0.982 GB (10 of 256) |
| PLE n-gram table (gather, IQ4_NL) | 51.200 | 43.94 | ≈ 0 (gather) |
| DeltaNet mixer (36 layers) | 2.087 | 1.79 | `attn_qkv` 711 + `ssm_out` 444 + `attn_gate` 379 MB |
| Embedding + LM head | 1.271 | 1.09 | `output.weight` 522 MB |
| Hyper-connection (BF16) | 0.641 | 0.55 | **1,283 MB (25.2 % of all bytes)** |
| Sparse attention + indexer (12 layers) | 0.617 | 0.53 | `attn_q` 262 + `attn_output` 151 MB + small |
| Shared expert (+gate) | 0.236 | 0.20 | in dense total |
| PLE q/k/v projections | 0.033 | 0.03 | `ple_key` 52 MB |
| MoE router | 0.031 | 0.03 | 63 MB (BF16) |
| **Total** | **116.514** | 100 | **5.096 GB** (dense/other 4.114 + experts 0.982) |

Reuse by params says almost nothing here: 95.8 % of the parameters are experts and the PLE table, which carry only 0.98 GB of the 5.10 GB read per token. The 4.11 GB of per-token dense traffic sits in the 4.2 % of parameters that are *not* experts or PLE.

## 2. Layer-by-layer map [V]

48 layers = 12 × [3 Gated-DeltaNet + 1 sparse attention]; attention layers are idx 3, 7, …, 47.

| Layer kind | Layers | Tensors (per layer) | Present in |
|---|---|---|---|
| DeltaNet block | 36 | `attn_qkv` [2560×10240], `attn_gate` [2560×6144], `ssm_out` [6144×2560], `ssm_alpha/beta` [2560×48], `ssm_conv1d` [4×10240], `ssm_a/dt/norm` | 0–2, 4–6, … |
| Sparse-attention block | 12 | `attn_q` [2560×12288], `attn_k/v` [2560×512], `attn_output` [6144×2560], q/k norms, `indexer.{q_proj,k_proj,norms}` | 3, 7, …, 47 |
| MoE FFN (all 48) | 48 | `ffn_{gate,up,down}_exps` [256 experts], `ffn_gate_inp` router [2560×256], shared expert (gate/up/down + `gate_inp_shexp`) | every layer |
| Hyper-connection (all 48) | 48 | `hc_{attn,ffn}_{down [10240×320], up [320×10240], inject [10240×4], norm}` | every layer |
| Once per model | 1 | `token_embd` (IQ4_XS), `output` (Q6_K), `output_hc_*`, `ple_{key,value,conv1d,norms}`, `per_layer_token_embd` [160 × 320,001,536] | – |
| MTP head (separate artifact, `Strata-data/mtp/rt/`) | 1 block | dense 116 MB + experts 708 MB, 512-expert router (not pruned) | outside the 116.5 B |

Quantisation per tensor [V]: trunk tensors are mixed Q4_K / Q5_K / Q6_K / IQ4_XS (e.g. `attn_qkv` 6.02 bits/weight average over 36 layers, `output` 6.56), experts IQ3_XXS / IQ4_NL / IQ2_S, hc and routers BF16, norms F32, plus private type 42 (Strata Q2_0). Expert FFN 640 wide × 256 experts: 2.046 MB average per expert (gate+up+down).

## 3. Provenance of candidate students [D, design claims]

From `scripts/provenance.py`:

| Candidate | C | D | L | Removed | Notes |
|---|---|---|---|---|---|
| **S0** teacher + placement / kernel / planner work | 116.514 B (100 %) | – | – | – | Family A. Distribution identical. |
| **S1** S0 + hc re-quantised | 115.874 B (99.45 %) | 0.641 B (0.55 %) | – | – | Derived from BF16 by a fixed rule. Function error unmeasured. |
| **S2** S1 + router fine-tune | 115.842 B (99.42 %) | 0.641 B | 0.031 B (0.03 %) | – | Family D. Needs a KL budget. |
| **S3** S2 + experts 256 → 192 per layer | 100.743 B (86.46 %) | 0.641 B | 0.031 B | 15.099 B (12.96 %) | Total 101.4 B = 87 % of P_T → **outside the ±2 % tier**; separate Pareto point. Router rows for dropped experts removed (mapped), surviving experts copied. |
| MTP head retrain (any S) | – | – | 0.82 GB on disk (116 MB dense + 708 MB experts); param count **[U]** | – | Separate artifact; learned from teacher outputs. |

The earlier ISTA-DASLab pruning (512 → 256 experts, RCO minimising KL on calibration data) is already baked into the teacher [L, HF card via a summarising fetcher; secondary source]: the teacher is itself a derived model, so "teacher logits" are p_T′, not the upstream model's.

## 4. Transformation-by-transformation analysis

For each: params, ops, memory traffic, error, equivalence conditions, decode/KV impact, hardware conditions. "Equivalent" means function-preserving by algebra; **similarity of weights does not establish it.**

### 4.1 Direct reuse + better kernels / placement / quantisation of already-quantised weights
- Params unchanged; bytes unchanged; error 0 (if kernels are exact up to float reordering).
- Equivalence: exact. Impact: only through the measured bottlenecks (cold-expert reads 69 % of a captured round, planner KV defect, unprofiled kernels).
- Condition: wins only if profile-guided; measure with ≥ 5 repeats (run-to-run variance "several percent").

### 4.2 Re-quantising hc tensors (0.641 B params, 1.283 GB/token)
- Bytes: 1,283 → 682 MB (Q8-like, 53 %), 526 MB (6-bit-like, 41 %), 361 MB (4-bit, 28 %) of BF16 [D]. Saves 0.60 / 0.76 / 0.92 GB of 5.10 GB per token = 11.8 / 14.9 / 18.1 % of bytes [D]. At the dense-resident 448 GB/s roofline that is 1.3 / 1.7 / 2.1 ms per pass of 54–71 ms measured [D] — **a real byte reduction, but only worth it if the bytes are on the critical path** (they are VRAM-resident or cold? **[U]**).
- Error, weight space only [M, `hc_requant_probe.py`, 12 of 48 layers, symmetric absmax per 32 values with fp16 scale; not the engine's own Q8_0/Q6_K code]: relative Frobenius error 0.61–0.76 % (8-bit), 2.5–2.9 % (6-bit-like), 11.1–12.3 % (4-bit); router `ffn_gate_inp` 0.76 / 2.9 / 11.4 %. The BF16 rounding already in the weights is ≈ 0.11 % rms [D].
- Equivalence: none; it is an approximation. 8-bit is within ~6× the existing BF16 rounding in weight space; function-space effect needs activation data **[U]**. The router must not be coarsely quantised without a routing-agreement test (a flipped top-10 changes the executed experts).
- Hardware: needs dequant kernels for hc paths in the engine (the engine already has Q8_0 / Q4 / Q6_K dequant for the trunk).

### 4.3 Low-rank (SVD) of dense tensors
- Params for W ∈ R^{m×n} at rank r: r(m + n) vs mn; ops likewise; break-even r < mn/(m + n) (e.g. 2560×10240: r < 2,048).
- hc bottleneck is already a rank-320 factorisation (down [10240→320], SiLU, up [320→10240]). Weight-only probe [M]: 90 % of Σ‖D_j‖²‖U_:j‖² needs 234 (attn) / 235 (ffn) of 320 units, 99 % needs 307 / 306; SVD(D) rank 229 / 212 for 90 % energy, 306 / 302 for 99 %; stable rank 165.5 / 134.5. So ≈ 25–30 % rank cut at 90 % energy is the most this tensor could give, and 90 % energy ≠ 90 % function.
- **Merging down·up is not function-preserving** because of the SiLU between them (`gr.cu`: norm → down → SiLU → gate/up) [V].
- Quantised trunk tensors: SVD needs a dequantised copy; not done (Python `gguf` package absent, dequant for Strata types must come from engine code) **[U]**. Low-rank of already ~6-bit trunk adds approximation on top of quantisation error; typical result is that the rank needed for tolerable error is close to full for attention/QKV-type matrices, but this is not measured here.
- Decode impact: fewer bytes only if the factors are stored low-bit; two small GEMVs instead of one adds a launch and an intermediate (latency-bound at batch 1).

### 4.4 Structured pruning (experts, heads, DeltaNet groups)
- Experts: 256 → 192 per layer removes 15.1 B params (12.96 %) and 25 % of cold-set bytes, but the model was already pruned 512 → 256 using a KL-on-calibration criterion; a second round compounds error and calibration-set dependence [L]. Router must drop columns; top-10 stays 10.
- Usage skew (top-64 covers 57.5 % of activations [M]) says some experts are rare, but "rare" ≠ "unimportant": rarely used experts can carry specialised capability (code languages). Needs importance by output-change (KL) not frequency.
- Equivalence: none. Quality: KL and task regression to be measured **[U]**.
- Impact: shrinks the 58.4 GB file → more fits in VRAM → fewer cold reads; exits the ±2 % tier (separate Pareto point).

### 4.5 Clustering / shared basis
- Merge expert groups into a shared basis + small deltas. Reduces storage only if the deltas are cheaper than the originals; **reduces compute only if the active path gets cheaper** — a shared basis still needs one read of the basis plus the per-expert delta, so memory traffic per active expert does not fall unless the delta is much smaller. Reject unless measured end-to-end gain (brief's rule).
- No measurement of cross-expert similarity has been made **[U]**.

### 4.6 Function-preserving transforms (exact algebra)
- Candidates: permutation of experts / hidden units (changes nothing but layout; useful for cache locality of expert blocks), folding norm scales into following projections, merging BF16 `inject` vectors where algebraically linear. These keep the function exactly and change only layout and bytes alignment.
- Not applicable to hc down·up merge (SiLU), nor to quantised blocks without re-quantisation (re-quantising after folding changes the rounding).
- Value: low expected speedup; main use is expert reordering so that co-activated experts are contiguous (measurable from routing dumps) **[U]**.

### 4.7 Layer-wise sensitivity-driven selection
- Choose bits / rank / pruning per layer from measured sensitivity (output KL when that layer alone is perturbed). Requires activations: a teacher forward run over a calibration set (335 tok/s prefill → 10^6 tokens ≈ 50 min per full pass; per-layer sweeps multiply that) [D]. Needs GPU; not run (GPU fault history).
- Not measured **[U]**; this is the main gating experiment (E3).

### 4.8 Weight-derived architecture + selective fine-tune / distillation
- Initialise a student from teacher weights (copy, derived or mapped) and train only the changed parts (S2: router + MTP head ≈ 31.5 M + head params). Training cost is then 6·N_trainable·D FLOP, small for N_trainable ≈ 10^8, but each forward/backward still needs the full teacher or student forward for activations → bounded by the 335 tok/s teacher-scoring rate unless logits are cached once (10^8 tokens = 3.5 days) [D].
- Reuse % is high by construction and says nothing about quality; measure KL and task scores.

### 4.9 Hybrids
Only combine transforms after each has a standalone ablation and an additivity test (error of A+B vs sum of errors). No ablation exists; no hybrid is proposed beyond S1–S3 above.

## 5. Provenance manifest for any derived artifact (required before use)
Each derived file gets `manifest.json`: source shard SHA-256 (recorded by an earlier session, **not recomputed**: shard 1 `e11083ba…7087fad`, shard 2 `316b46f3…e161e113`), per-tensor category (C/D/M/L), transform + parameters, git commit of the script, seed, calibration set id, measured error (weight-space and function-space), date. The original shards stay read-only.

## 6. What this section does not establish
- No function-space error for any transformation (needs activations/GPU; held back by the GPU fault history).
- hc re-quant probe uses a simple symmetric block quantiser, not Strata's kernels.
- The trunk (quantised) was not dequantised; SVD/pruning sensitivity of trunk tensors is unmeasured.
- Licence conflict unresolved (Coder card Apache-2.0 vs upstream `qwen-community-1.0`); derived weights may not be redistributable until resolved.
