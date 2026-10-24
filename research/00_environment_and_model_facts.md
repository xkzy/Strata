# 00 — Environment and model facts

Status key: **[V]** verified by reading a primary artifact this session; **[I]** inference (reasoned from [V], not directly read); **[U]** unknown.
Access date for all web sources: 2026-10-24. Local commands were run in the worktree `strata-ncm` (branch `worktree-strata-ncm`, based on `main`).

## 0.1 What `qwen3.8-flash-next-coder` is

**It is not an Ollama tag and not an 8B model.** `ollama show qwen3.8-flash-next-coder` returns `model ... not found`; `ollama list` has no
`flash-next` entry (installed: `qwen3.8-27.3B-Q4_K_M`, `qwen2.5-coder:7b-instruct`, `glm-4.7-flash`, others). [V]

The identifier resolves to a GGUF held by Strata's data directory, in two shards (58.4 GB total), loaded through Strata's own pack format. [V]
Its header (`general.*`, read with `ncm/param_count.py`, `research/scripts/budget_model.py`):

| Field | Value | Source |
|---|---|---|
| `general.name` | `Qwen3.8-Flash-Next GSQ-RCO Coder` | GGUF header [V] |
| `general.architecture` | `qwen4exp`; `general.description` = "A Preview of the Qwen4 Architecture" | GGUF header [V] |
| `general.quantized_by` / `repo_url` | ISTA DASLab / `huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` | GGUF header [V] |
| `general.size_label` | `512x56B` (the original expert count and per-expert-group size label) | GGUF header [V] |
| Base model | `Qwen/Qwen3.8-Flash-Next`, 176.9 B parameters, 354 GB at BF16 | model card, fetched [V] |
| What was done to it | experts pruned 512 → 256 per layer (RCO, arXiv 2605.00649 per the card); weights quantized (GSQ, arXiv 2604.18556 per the card); n-gram table stored IQ4_NL, neither pruned nor searched | model card [V] |
| Quantization | file mix of IQ1_M / IQ2_S / IQ3_S / Q2_0 / Q6_K / BF16 / F32 and a custom type id 42 (tensor types counted in `docs/parameter_budget.md`); card says 3.5 bpw for retained transformer weights and calls the repo tag IQ1_M without reconciling them | header + card [V]; the discrepancy is unresolved [U] |
| License | `apache-2.0`, "inherit the license of the base model"; no stated condition on fine-tuning, distillation or redistribution | model card [V]; base-model license text itself not fetched [U] |
| Reported quality vs unpruned base (card) | SWE-bench Verified 79.20 vs 82.80; LiveCodeBench v6 87.43 vs 87.43 | model card [V] (the card's numbers, not re-measured here) |

The card also states the model is an "experimental release" tuned for coding, agentic tool use, vision and spatial reasoning, with degradation
expected elsewhere. [V] Strata is named on the card as a supported third-party inference engine. [V]

## 0.1b Architecture (all from the GGUF header unless noted)

| Item | Value |
|---|---|
| Layers (`block_count`) | 48 = **36 gated-linear-recurrent ("GatedDeltaNet"-type, `ssm_*` tensors) + 12 sparse-indexer attention layers** (`full_attention_interval` = 4; `attention.compress_ratios` is 0 on 36 layers and 4 on 12). The mapping "ssm_* = Gated DeltaNet" is an inference from tensor names and the Qwen3-Next lineage [I]; the update rule itself was not read from engine source here. |
| Hidden size | 2560; **hyper-connections, 4 residual streams** (`hyper_connection.count` = 4, low-rank 320; the residual is a `4 × 2560` stack per token) |
| Attention (12 layers) | 24 query heads, **2 KV heads**, key/value length 256; sparse **indexer** (4 heads, key length 128, **top-k 2048**); RoPE dim 64 (sections 11/11/10/0), base 1e7 |
| Linear-recurrent (36 layers) | `ssm.inner_size` 6144, `state_size` 128, `time_step_rank` 48, `group_count` 16, conv kernel 4 |
| Context | `context_length` = 262,144 |
| MoE | 256 routed experts per layer (pruned from 512), **top-10**, expert FFN width 640, plus a shared expert (width 640) with its own gate |
| Vocabulary | 248,320 rows in `token_embd` and `output.weight` (separate tensors, **not tied**); `tokenizer.ggml.pre` = `qwen35`, byte-level BPE (gpt2 model), BOS = pad = 248044, EOS = 248046 |
| Per-layer embedding (PLE) | n-gram hash table `per_layer_token_embd.weight` (160 × 320,001,536; 8 heads per n-gram, n-gram size 3), plus small projection tensors. A lookup, not a matmul. |
| Multi-token prediction | An MTP layer exists in the engine (`src/core/mtp.cpp`) and is used as a draft source; whether the GGUF stores trained MTP head tensors was not separately enumerated here [U]. |
| Sampling defaults in header | temperature 1.0, top-k 20, top-p 0.95 |

Parameter counts (`research/scripts/budget_model.py`, header only; tensor shape products, each tensor once): [V]

| Quantity | Value |
|---|---|
| Unique stored elements, all 1,224 tensors | **116,514,464,640** (card sidebar: "117 B") |
| of which PLE n-gram table | 51,200,245,760 (43.9%) |
| of which routed experts | 60,397,977,600 (51.8%) |
| Matmul'd weights (everything except the PLE table and the token-embedding row lookup) | 64,678,519,680 |
| **Active per token** (top-10/256 of routed + all non-expert matmul weights + LM head; embedding and PLE rows excluded) | **6,639,838,080 (5.70% of stored)** |

Unknown or unresolved here: the card's 3.5 bpw vs IQ1_M label; whether MTP head weights are in the file; exact Gated-DeltaNet state shape (the memory
model in 03 states its assumption); the base model's license text.

## 0.1c Checkpoint integrity

Both shards were opened read-only; no model file was modified. SHA-256 of the two shards is computed once and stored in
`research/teacher.sha256`, computed 2026-10-24 with `sha256sum` [V]:

```
e11083ba855e7666b48ea3f2db6a9c3a20c18751a012cc24f948de91b7087fad  ...-00001-of-00002.gguf
316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113  ...-00002-of-00002.gguf
```

Any later experiment should re-hash before and after to show the teacher was untouched.

## 0.2 Strata audit (`xkzy/Strata` fork, this worktree)

Detailed findings with file and line references are in [`docs/architecture_audit.md`](../docs/architecture_audit.md) (audited at commit `58c5fb26`; static
read, tests not run). Summary [V unless marked]:

- **Languages / build:** Go module `strata` (server, installer, tokenizer, web app) + C++20/CUDA (HIP, SYCL experimental) engine. The Go server spawns the
  C++ engine and speaks a line protocol (`pkg/engineipc/orchestrator.go`): **one token id per line**.
- **Model support is single-architecture:** `check_architecture` (`include/strata/artifact/gguf_reader.hpp:637`) accepts only `qwen4exp`; layer and expert counts
  are `constexpr` (`include/strata/models/qwen_hybrid.hpp:22-32`). `ModelAdapterRegistry` is planning metadata only. A different student graph needs a new forward graph,
  not a configuration change.
- **No training support:** no backward pass, optimizer or loss anywhere in `src/`, `include/`, `pkg/`, `cmd/`. Training needs a separate framework (PyTorch here) and an export step.
- **Multi-token paths that exist:** speculative verify window (`src/core/verify.cpp`, up to 8 rows, **greedy-exact only**), MTP draft (`src/core/mtp.cpp`),
  suffix/lookup drafts (`src/spec/`), batched prefill (`src/prefill/`). There is no sampling-exact (rejection-sampling) verifier. [V by reading; not run]
- **KV / context:** per-attention-layer KV with streaming residency, parked-conversation snapshots, a message-boundary cache. `src/context` and
  `pkg/virtualcontext` are policy layers; whether they drive the real GPU KV was not checked. [U]
- **Tokenizer:** Go byte-level BPE (`pkg/tokenizer`); the C++ engine receives ids only.
- **Registry error:** `pkg/modelschema/registry.go:77-100` describes the coder model as 40 layers / hidden 3584 / 64 experts; the GGUF says 48 / 2560 / 256. Not corrected (out of scope).

**Dependency and compatibility map for a student** (what must change if a new architecture is ever served): `gguf_reader.hpp` (architecture check),
`qwen_hybrid.hpp` (constants), a new forward graph beside `verify.cpp`/`generate.cpp`, `orchestrator.go` (block emission), `registry.go` (manifest). None of
this has been done.

## 0.3 Hardware and environment

RTX 5060 Ti 16 GB (PCI 01:00.0, sm_120) + RTX 4060 8 GB (sm_89), 60 GB RAM, torch 2.10 (cu128). CUDA 13.2 at `/usr/local/cuda-13.2`. The 5060 Ti has faulted twice with
Xid 79 under sustained load (2026-10-23, 2026-10-24 19:26); after a fault CUDA is unavailable to all processes until the driver is reloaded or the machine rebooted
([`docs/distillation_plan.md`](../docs/distillation_plan.md)). Standing owner rule: training runs on the GPU only. [V]

## Work done before this research brief (disclosure)

The earlier mission authorized implementation, so before this brief arrived the branch gained: a read-only parameter counter; PyTorch prototypes (`ncm/`); a teacher
capture of 160 prompts × 256 new tokens (~41k decode positions of final-layer residuals; data outside git); and **two small engine patches in
`src/program/generate.cpp`** (`--dump-final-r` on the `--spec` verify path, and `--head-eval`). This brief says not to modify production code before approval. The patches
are additive (new options, default behaviour unchanged; an unpatched and a patched binary produced identical tokens on a 2-token check), are committed on a local branch,
have **not** been pushed, and can be dropped by resetting that branch. Their results are used below only as labelled empirical evidence.
