# Baseline report

Status labels: **[M]** measured (source cited), **[V]** verified by reading a file/header in this session, **[I]** inferred, **[U]** unknown.
Access date for all web sources: 2026-10-24.

## 0. What was and was not run

- **Not re-run in this session: a live teacher generation benchmark.** The 5060 Ti has a recorded history of falling off the bus (Xid 79) and taking CUDA down (project memory `ncm_gpu_fault_history`), and a 58 GB checkpoint with RAM-resident experts is a long, heavy run. A live baseline is experiment **E0** in `experiment_plan.md` and needs the owner's go-ahead for GPU time. Until then every speed number below is a **recorded measurement from repository files**, with its conditions.
- **No student model exists, so there is no student throughput to benchmark.** The "current NCM-inspired algorithm" in code (branch `worktree-strata-ncm`, local only, `ncm/*.py`, plus a small engine patch `--dump-final-r` / `--head-eval`) consists of *compression experiments on the teacher's final-layer state*, not a trained generator. The "has not performed well" evidence is therefore the experiment results in §5, not a tokens/s gap.
- Verified in this session (headers read, no weights loaded): `research/scripts/gguf_dump.py` → `research/data/teacher_gguf_headers.json`; `param_count.py`, `budget_model.py`, `latency_model.py` → outputs in `research/data/`.

## 1. Teacher identity [V]

| Field | Value | Source |
|---|---|---|
| Requested name | `qwen3.8-flash-next-coder` | task |
| What it is | A Strata built-in **registry name** (`HANDOFF.md:145`), not an Ollama tag (`ollama show` → "not found"; Ollama 0.40.6 lists no Qwen3.8-Flash-Next) | shell |
| Files | `Strata-data/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-0000{1,2}-of-00002.gguf` (29,608,446,496 + 28,800,138,432 bytes) | `ls` |
| SHA-256 shard 1 | `e11083ba855e7666b48ea3f2db6a9c3a20c18751a012cc24f948de91b7087fad` | recorded by an earlier session (`/home/khing/.claude/jobs/a719c9cc/tmp/teacher.sha256`); **not recomputed here** |
| SHA-256 shard 2 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` | same |
| GGUF `general.name` | "Qwen3.8-Flash-Next GSQ-RCO Coder", arch `qwen4exp`, description "A Preview of the Qwen4 Architecture", `size_label` `512x56B` | header |
| Lineage | Expert-pruned (512→256 experts/layer, chosen by RCO to minimise KL on calibration data) and quantised (GSQ) derivative of `Qwen/Qwen3.8-Flash-Next` by ISTA-DASLab. The "512x56B" label is the *pre-pruning* expert count. | HF model cards (see §6; read through a summarising fetcher, so treat details as secondary) |
| Card-reported quality | SWE-bench Verified 79.20 vs 82.80 BF16 base (95.7 % retained); LiveCodeBench v6 87.43 both | Coder model card, **vendor-reported, not reproduced** |
| Licence | Coder card: Apache-2.0 "inherited from base". Upstream base card: `qwen-community-1.0`. **These disagree; unresolved.** Distilling and redistributing needs the actual licence texts. | HF cards |

**Consequence for everything below:** the teacher whose logits are available locally is a *pruned, ~1.9-bit-effective* derivative, `p_T'`, not the BF16 base. A student distilled from it can at best match `p_T'`.

## 2. Architecture [V unless marked]

From GGUF metadata (`qwen4exp.*`) and the 1,224 tensor shapes:

- 48 layers = 12 repeats of (3 × Gated-DeltaNet mixer + 1 sparse-attention layer); attention layers are indices 3, 7, …, 47. Hidden 2560.
- Attention: 24 Q heads, 2 KV heads, head dim 256, RoPE dim 64, base 1e7, indexer (4 heads × 128) selecting top-2048 tokens; `compress_ratios` = 4 on attention layers. The *meaning* of `compress_ratio` is **[I]** (block size 4 for selection, consistent with the card's "512 blocks = 2048 tokens"); not read from loader code.
- DeltaNet: conv kernel 4, state 128, 16 QK groups, 48 V heads, inner 6144.
- MoE: 256 experts/layer (pruned from 512), top-10 + 1 shared, expert FFN width 640, router `ffn_gate_inp` 2560×256.
- Hyper-connections: 4 streams, rank 320 (4 × 10240×320 matrices per layer).
- Per-layer-embedding (PLE) n-gram table: `per_layer_token_embd.weight` shape 160 × 320,001,536 (IQ4_NL), 16 hash heads, bigram/trigram; read as a row gather.
- Vocabulary 248,320 (GPT-2-style BPE, `pre=qwen35`), context 262,144 native; rope scaling beyond that is experimental in Strata.
- Multi-token prediction: **present** — one MTP layer shipped separately in `Strata-data/mtp/rt/` (`dense.bin` 116 MB, `experts.bin` 708 MB, `draft_vocab.bin`). `dense.txt` lists a full block: Q/K/V/O projections, indexer, a **512-expert** router, shared expert, three hyper-connection mixers, `fc_embedding`/`fc_hidden`. Experts are ≈ 2.25 bits/weight (708 MB / 512 / 4.915 M params). Strata drafts up to 3–4 tokens with it.
- Quantisation mix in shard 1 (ggml type ids): Q4_K/Q5_K/Q6_K, IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS, IQ4_NL (table), BF16 for routers/hyper-connections, and **private type 42** (Strata's 64-value/18-byte Q2_0) for some expert down matrices — Ollama cannot load this file (`import-ollama-coder.sh`).

### Parameter counts (from tensor shapes, `budget_model.py`)

| Quantity | Value |
|---|---|
| Total unique parameters stored `P_T` | **116.514 B** (ε = 2 % band 114.2–118.8 B) |
| of which PLE lookup table | 51.200 B (43.9 %) — a gather, ~0 FLOPs; 28.80 GB at IQ4_NL (matches shard 2) |
| of which routed experts | 60.398 B (51.8 %) |
| Total excluding PLE table | 65.314 B |
| Active per decoded token `P_A` | **6.640 B** (non-expert 4.28 B + 10/256 of routed 2.36 B); upstream card says "~6 B" |
| MTP layer (not in `P_T`) | ≈ 2.5 B routed + dense; card says 4 B at BF16 **[U exact]** |

Cross-check: unpruned base ≈ 120.8 B routed + 4.9 B trunk ≈ 125.7 B matches the upstream card's "~125 B" (excluding 51 B n-gram, 4 B MTP). Independent of the earlier session's identical 116.5 B / 6.64 B result.

## 3. Hardware [V/M]

- GPU 0: RTX 5060 Ti 16 GB; GPU 1: RTX 4060 8 GB (`nvidia-smi`, both idle, P8). `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Memory bandwidth: vendor specs (448 / 272 GB/s) are **assumed**, not measured here. CPU, RAM size and PCIe link widths: **[U] not collected** (to do in E0).
- Run configuration in use (`strata-coder-iq1_m.json.bak4`): `--native` shard 1, `--ple-gguf` shard 2, `--expert-profile`, `--expert-cache auto`, `--spec 4 --spec-min-p 0.5`, `--mtp`, `--max-context 262144`, `--kv-resident 20480`, `--mmap-experts`, `--peer-device 1`, control-vector/abliteration flags (these modify outputs; **a clean baseline must run without them**, see E0).
- Build: `engine/BUILD.json` exists (flags not yet read) **[U]**.

## 4. Recorded throughput

| Number | Value | Conditions | Source |
|---|---|---|---|
| Decode, this PC | **44.8 tok/s** (config "search-9"; 37.35 for "peer") | 256 decode tokens after a 3,001-token prompt, MTP spec on, both GPUs, tuned 2026-10-18 | `strata-coder-iq1_m.json.bak4` `tuned.results` **[M]** |
| Prefill, this PC | **335 tok/s** (291.85 "peer") | same run | same |
| Decode, benchmark PC (RTX 5070 12 GB, Ryzen 5 7600, 64 GB, engine 0.1.26) | Coder 58.9 / 55.1 / 54.9 / 53.2 / 43.0 / 42.8 tok/s at 1K/4K/32K/64K/128K/262K | one code-agent prompt per length, 256 tokens, MTP on | `docs/DETAILS.md:45-55` **[M]** |
| Prefill, benchmark PC | Coder 656 / 1,583 / 2,177 / 2,236 / 2,208 / 1,034 tok/s (262K row from engine 0.1.14) | same | `docs/DETAILS.md:31-43` **[M]** |
| Speculation | 2.4–3.2 tokens per verify pass (MTP, up to 3 drafts); draft acceptance 0.77–0.78 on code (0.51–0.63 French/English text) | | `docs/DETAILS.md:1294`, `:114-116` **[M]** |
| Same-rig variance | "a different answer to the same prompt moves it by several percent"; 88.5 vs 85.7 tok/s back-to-back | | `docs/DETAILS.md:57-59` **[M]** |

**Strongest baseline** = this checkpoint on the unmodified engine with its own MTP head and prompt lookup, i.e. **44.8 tok/s decode / 335 tok/s prefill on this PC, already ≈ 2.4–3.2 tokens per full pass.** Any student must beat that, not plain one-token decoding.

## 5. Evidence on the current (NCM-style) approach

All from files left by the earlier session (job `a719c9cc`), raw JSON in `Strata-data/ncm/`. Not re-run here. Teacher capture: final-layer residual stream of the teacher on ~30–40 prompts, windows T = 128, PCA to R = 128 dims, 256-code codebooks, 600 steps, seeds 0–2, 256 train / 64 test windows (`stage2_cap2.json`).

**Reconstruction error (test NMSE = ‖ẑ−z‖²/‖z−mean‖², mean of 3 seeds; 1.0 = predicting the mean):**

| bits/pos | A: plain RVQ | B: filter bank + RVQ | Difference |
|---|---|---|---|
| 8 | 0.853 | 0.838 | B 0.015 better |
| 16 | 0.809 | 0.791 (coarse-heavy 0.789) | B 0.018 better |
| 24 | 0.793 | 0.769 (coarse-heavy 0.768) | B 0.024 better |
| 32 | 0.774 | 0.753 (coarse-heavy 0.752) | B 0.021 better |
| PCA R=128, unquantised | 0.557 | | |

Seed std ≈ 0.001–0.016, so B's NMSE edge (~0.02) is real but small; **a 77 % residual error at 32 bits/position is nowhere near reconstructing a state**. With only 9 train sequences (`ms_result.json`) test NMSE ≥ 1.0 (worse than the mean): overfitting.

**What matters, token agreement** (teacher's own LM head argmax on the reconstructed state vs the teacher's recorded next token; 8,192 held-out positions; seed 0 only; `head_agreement_cap2.json`, definitions from `ncm/head_agreement.py`):

| Variant | Top-1 agreement |
|---|---|
| true state (sanity) | 1.000 |
| true 4-stream *mean* broadcast (hyper-connection streams collapsed) | 0.796 |
| train-mean state | 0.022 |
| PCA-128 only (no quantiser) | 0.297 |
| A, 8 bits/pos | 0.078 |
| B, 8 bits/pos | 0.061 |
| A, 32 bits/pos | 0.121 |
| B, 32 bits/pos | 0.119 |

Binomial s.e. at p≈0.12, n=8192 is ≈ 0.0036. So **B is not better than A on the metric that matters** (worse at 8 bits; tied at 32). Caveat: this tests quantising a *hidden state*, which is a different object from NCM's text tokenizer (which reconstructs *tokens* with a trained decoder; NCM reports 98.8–99.99 % exact token reconstruction at ≈100 M scale). It shows hidden-state codes are the wrong target, not that NCM text codes fail.

**Performance gap / quality gap:** there is no student, so the throughput gap is undefined; the representation experiments above show a ≥ 80-point token-agreement gap for compressed-state codes.

## 6. Sources

- NCM: Kumbong et al., *Next-Concept Modeling*, arXiv:2610.14162; code github.com/HazyResearch/ncm (MIT).
- Coder card: huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF; base card: huggingface.co/Qwen/Qwen3.8-Flash-Next.
- Repo: `docs/DETAILS.md`, `HANDOFF.md`, `pkg/autotune/workload.go`, `strata-coder-iq1_m.json.bak4`.

## 7. Reproducibility status

| Item | Status |
|---|---|
| Header dump, parameter counts, latency model | reproducible: `python3 -I research/scripts/{gguf_dump,param_count,budget_model,latency_model}.py` |
| Checkpoint hashes | recorded by earlier session, **not re-verified** |
| 44.8 / 335 tok/s | single tuner run (n=1, 256 tokens); variance unknown; **E0 must repeat ≥ 5×** |
| Compression experiments | raw JSON present; producing scripts in `worktree-strata-ncm/ncm/`; not re-run |


## 9. E0 clean live baseline (measured 2026-10-25, this PC)

Source: server log of a live run, parsed by `scripts/e0_parse.py` into `data/e0_results.json`; client script `scripts/e0_bench.py`. Owner approved GPU use for this run.

**Setup [M].** AMD Ryzen 9 9950X3D (16 cores / 32 threads), 60 GB RAM (RAM speed not collected: needs root), RTX 5060 Ti 16 GB (CUDA device 0 only), RTX 4060 not used. Engine `engine/strata` 0.1.40 (sm_89 + sm_120, CUDA 13.2), driver 595.112.07. Config = the tuned "search-9" config of 2026-10-18 **minus** the control-vector, abliteration and `STRATA_CVEC_FUSE` settings (`--spec 4 --spec-min-p 0.5 --mtp --expert-cache auto --expert-profile --kv-resident 20480 --mmap-experts --pcie-frac 0.25 --pool-workers 4`). Temperature 0, 256 generated tokens, unique prefix per request to defeat the prompt cache, one warm-up request per size (excluded), 5 measured repeats per size, one request at a time. Prompt text is repetitive filler plus a code request; it may flatter draft acceptance.

| Prompt | n | Prefill tok/s | Decode tok/s (sd; min–max; CV) | Draft acceptance | Expert-cache hit rate |
|---|---|---|---|---|---|
| ~1.0K tokens | 5 | 476 | **31.7** (2.5; 28.6–35.6; 8 %) | 0.76 | 56 % |
| ~3.8K tokens | 5 | 1,196 | **32.0** (1.5; 30.6–34.5; 5 %) | 0.75 | 57 % |
| ~16.2K tokens | 5 | 1,482 | **31.1** (0.8; 30.4–32.0; 3 %) | 0.73 | 57 % |

**Engine facts from the log [M].** 4,424 expert slots resident in 8.42 GiB of VRAM (about 92 per layer if spread evenly), filled from the profile in 2.8 s; 335 MiB VRAM free after load; 15.6 GB of the 16 GB card in use with the model loaded (steady state, not a peak reading); the prompt path borrows 2,066 cache slots (3.90 GiB); first request after start took 4.3 s for 84 tokens (graph capture and warm-up).

**Against the recorded baseline [D].** Decode 31–32 tok/s vs recorded 44.8 tok/s: **about 29 % lower** on this clean config. Cause not established. Candidates: the removed speed-projection control vector (changes the output distribution and may change acceptance), the prompt text, a different thermal or power state, and the recorded run being a different engine build; none has been isolated. Prefill is not comparable (recorded 335 tok/s on a 3,001-token prompt; here 1,196 tok/s at 3.8K on repetitive text).
**Consistency with the analysis [D].** The measured decode hit rate (56–57 %) matches the 57.5 % that the static top-64 profile coverage predicted (`performance_bottleneck_analysis.md` §3 item 5), now with about 92 resident per layer, so the profile coverage curve is a reasonable predictor of the hit rate.

**Not collected.** Context sizes of 32K, 128K and 262K; batch/concurrency; peak VRAM during a request; load time; power; RAM speed; logit dump and quality; a clean-config run with the control vector restored to isolate the 29 % gap. Xid: a kernel-journal watcher ran during the benchmark and raised no event; I could not re-query the journal after the run because the shell wrapper refused `journalctl`, so the end-of-run check is by that watcher only. Server stopped after the run.
