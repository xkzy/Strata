# Strata-NCM: architecture audit

Audited at commit `58c5fb26` (static read of source; tests were not run). Items marked *not checked* were not verified.

## Headline findings

1. **The teacher is not an 8B model.** `qwen3.8-flash-next-coder` is the GGUF `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M`
   (two shards, 58.4 GB, arch `qwen4exp`, 48 layers, 256 of 512 experts kept, top-10, hidden 2560). See
   [parameter_budget.md](parameter_budget.md) for exact counts. The "Qwen3 8B-class" assumption in the mission is false for this checkpoint.
2. **The checkpoint is a heavily quantized artifact** (IQ1_M/IQ3_S/IQ2_S/Q2_0 mixes plus a custom type id 42). It is usable as an
   inference teacher through the Strata engine, but it is not a bf16 source model; weight-level initialisation of a student from it
   (Configuration B) would start from lossy weights.
3. **Strata cannot train.** No backward pass, optimizer or loss exists in `src/`, `include/`, `pkg/`, `cmd/`. Distillation needs a
   separate pipeline (here: `ncm/`, PyTorch). Strata is only the teacher-inference and serving side.
4. **The engine is single-architecture.** `check_architecture` (`include/strata/artifact/gguf_reader.hpp:637`) refuses anything but
   `qwen4exp`; layer/expert constants are `constexpr` in `include/strata/models/qwen_hybrid.hpp:22-32`. A new student architecture needs
   a new forward graph, not a config change. `ModelAdapterRegistry` (`src/models/model_adapter.cpp`) is planning metadata only.
5. **`pkg/modelschema/registry.go:77-100` is wrong for the coder model** (says 40 layers, hidden 3584, 64 experts; GGUF says 48 / 2560 / 256).
6. No `teacher`/`ncm` references exist anywhere in the repo before this work.

## Facts verified in source

| Area | Finding |
|---|---|
| Build | Go module `strata` (go 1.26.8); CMake C++20 + CUDA (default arch 120) / HIP / SYCL(experimental). Go server spawns the C++ engine and talks a line protocol (`pkg/engineipc/orchestrator.go`). |
| Format | GGUF only (+ split shards). Quant types incl. Q2_0 (id 42, 64 el / 18 B). Python `gguf` 0.18 cannot read id 42; `ncm/param_count.py` reads headers without the enum. |
| Architecture | Hybrid: 36 GatedDeltaNet (linear-attention) + 12 sparse-indexer attention layers; hyper-connections (4 streams); MTP layer; PLE n-gram hash table. |
| Tokenizer | Go byte-level BPE (`pkg/tokenizer`); C++ receives token ids only. Vocab 248,320 rows in `token_embd`. |
| Autoregressive assumptions | Per-token CUDA graph (`core/session.hpp`), one `T <id>` line per token over the IPC pipe, one request at a time unless `--batch`. |
| Existing multi-token paths | Speculative verify window (`core/verify.cpp`, up to 8 rows, **greedy-exact only**), MTP draft (`core/mtp.cpp`), suffix/lookup drafts (`src/spec/`), batched prefill (`src/prefill/`). |
| KV / memory | Per-QSA-layer KV with streaming residency (`--kv-resident`), elastic VMM growth, parked-conversation LRU with on-disk snapshots, message-boundary cache. `src/context` + `pkg/virtualcontext` are policy layers; *not checked* whether they drive real GPU KV. |
| GPU safety | Bounded waits (`core/gpu_wait.hpp`), engine restart with backoff, stage-named GPU error reports. |
| Autotune | `pkg/autotune` constraint solver + benchmark + roofline cost; run config records a `tuned` block. |
| Tests | ~155 CMake tests, 87 Go test files. *Not checked:* whether they currently pass. |

## Files that would need modification to serve a new student

Only after a trained student exists, and only if it fits the `qwen4exp` graph (it does not, for multiscale generation):

- `include/strata/artifact/gguf_reader.hpp` (architecture check), `include/strata/models/qwen_hybrid.hpp` (constants)
- a new forward graph beside `src/core/verify.cpp` / `src/program/generate.cpp` (multi-token-per-round decode)
- `pkg/engineipc/orchestrator.go` (protocol emits one token per line; needs block emission)
- `pkg/modelschema/registry.go` (new manifest; also fix the coder entry)

Nothing in those files has been changed. The original Qwen path is untouched.

## Hardware available for this work

RTX 4060 8 GB (the run config also lists an RTX 5060 Ti 16 GB), 60 GB RAM, torch 2.10 CPU/GPU. No `nvcc` on PATH.
Running the 58 GB teacher requires Strata's expert tiering; it is not loadable in PyTorch here.
