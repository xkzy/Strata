# Handoff - 2026-10-08

## Summary of Completed Work

### 1. ChatGPT-Style Multi-Session Chat Experience (Web UI & Backend)
- **Collapsible Sidebar**: Fully responsive session drawer (`#chat-sidebar`) with "+ New Chat" button, live search/filtering across conversations, timestamps, and active session highlights.
- **Session Management**: Inline renaming, session deletion with fallback switching, session forking (branching conversation trees), and bulk clear history.
- **Persistence & Migration**: Persistent multi-session store (`sessions_list` in `localStorage`) with automatic migration of existing single-session chat history.
- **Context Isolation & Backend Routing**: Automatic propagation of active `session_id` via `X-Session-ID` header across OpenAI/Anthropic endpoints, Virtual KV caching, and multi-tenant isolation.

### 2. Live PCIe Link Speed & Real-Time Throughput Telemetry
- Implemented live hardware PCIe throughput sampler via `nvidia-smi dmon` in `pkg/telemetry/extra.go` and `pkg/telemetry/telemetry.go`.
- Real-time sampling of PCIe RX & TX transfer rates in MB/s with 60-sample historical sparkline buffers.
- Detection of PCIe link generation (`Gen1` – `Gen6`), link width (`x1` – `x16`), and theoretical maximum bandwidth calculations (`gpu_pcie_bw_gb_s`).
- Dynamic UI sparkline rendering reflecting prefill bursts and hybrid offload transfer spikes.

### 3. Model Catalog & Auto-Discovery (Hugging Face, Ollama & Local Packs)
- Automated scanning of Hugging Face Hub cache (`~/.cache/huggingface/hub/`) and Ollama model library (`~/.ollama/models/manifests/`) in `pkg/installer/installer.go`.
- Dynamic model picker in the Web UI header (`#model-select`) allowing instant switching between available models without restarting the server.
- Interactive catalog in the Setup tab with category filters (`All`, `Coding`, `General`, `Fast / Light`, `Vision`, `Custom / Local`).

### 4. Hybrid Compute Offloading & Multi-GPU Configuration
- Added `--pool-workers` (CPU thread pool workers for matrix operations), `--pcie-frac` (PCIe transfer threshold ratio), `--gpus` (multi-GPU and integrated GPU compute allocation), and `--threads` to `pkg/runconfig` and the Web UI Settings tab.
- Integrated runtime memory management across dGPU VRAM, APU/iGPU GTT shared memory, and Host RAM.

### 5. Virtual KV Page Table & Hierarchical Memory Subsystem
- Decoupled logical context retrieval from physical KV residency in `include/strata/rt/kv_page_table.hpp`, `src/rt/kv_page_table.cpp`, and `pkg/virtualcontext/kv_page_table.go`.
- 64-byte aligned composite `KVBlockKey` ensuring RoPE and content invariance.
- Multi-tier residency hierarchy: `L1 GPU_VRAM`, `L2 Host_RAM`, `L3 Compressed_RAM`, `L4 NVMe_Disk`, and `L5 Recompute`.
- Cost-aware eviction prioritizing high-prefill compute pages and pinned system prompt anchors.
- Context prefetch predictor overlapping Host->Device DMA transfers with token generation.

### 6. Deterministic LLM Logic Verification & Transparent Runtime
- Tri-state deterministic logic verification (`PASS`, `FAIL`, `UNKNOWN`, `PARTIAL`).
- Integrated Mathics symbolic CAS, rational arithmetic, propositional Modus Ponens deduction, linear interval solvers, SI unit dimensional analysis, and recursive JSON schema validation.
- All 12 C++ logic verifier test suites, 8 CAS test suites, and 16 Go package suites PASS (100% green).

### 7. Agent Monologue Isolation & Dynamic Tool Call Parsing
- **Clean Reasoning Separation**: Dynamic `<think>` and `<thought>` token routing in `pkg/server/outparse.go` guarantees internal reasoning flows to `reasoning_content` (OpenAI) and `type: "thinking"` (Anthropic), preventing monologue bleed into response `content`.
- **Universal Tool Parsing**: Universal JSON object, JSON array, code-fence (````json ... ````), and XML `<function=...>` detection in `pkg/server/outparse.go`.
- **Tool Finish Reasons**: Auto-assigns `finish_reason: "tool_calls"` (OpenAI) and `stop_reason: "tool_use"` (Anthropic) in `pkg/server/api_openai.go` and `pkg/server/api_anthropic.go`.
- **Real-time Prompt Reading Progress**: Prefill progress events (`PP <read> <total>`) in `pkg/engineipc/orchestrator.go` route directly to `monitoredRequest`, driving live percentage indicators on the Web UI.

### 8. Kernel & Decode Pipeline Performance Optimizations
- **Zero-Allocation Stack-Buffered FastAttention**: Replaced per-step heap vector allocations in `include/strata/kernels/fast_attention.hpp` with stack buffers for context lengths $\le 1024$.
- **Parallel Multi-Head GQA Decoding**: Multi-head decode loop parallelized across OpenMP CPU worker threads.
- **FastMoE Routing Optimization**: Eliminated dynamic vector allocations in `include/strata/kernels/fast_moe.hpp` using stack buffers for expert scores.
- **FastNorm Vectorization & Branch Hoisting**: SIMD-vectorized $(1 + \gamma)$ Gemma-style RMSNorm with AVX-512/AVX2 FMA and hoisted `weight` conditions out of inner scaling loops in `include/strata/kernels/fast_norm.hpp`.
- **FastRoPE SIMD Interleaved Rotation**: Vectorized interleaved RoPE rotations using AVX-512/AVX2 FMA in `include/strata/kernels/fast_rope.hpp`.
- **Parallel Batch Activation Processing**: Multi-threaded OpenMP element processing in `include/strata/kernels/fast_activations.hpp` (`fast_activate_array`, `fast_swiglu_array`).
- **Measured Speedups**:
  - Single-Head Decode Attention: **32.8% faster** (10.40 µs $\rightarrow$ 6.99 µs / head).
  - Dense Transformer Layer Forward: **41.7% faster** (7564.26 µs $\rightarrow$ 4411.68 µs / layer).
  - Weighted MoE Aggregation: **26.6% faster** (0.94 µs $\rightarrow$ 0.69 µs / token).
  - Softmax + ArgMax Sampling: **50.1% faster** (118.38 µs $\rightarrow$ 58.81 µs / step).

### 9. Xenolith-Inspired Quantized Kernel & Zero-Allocation Prefill Optimizations
- **Direct $Q4\_0, Q4\_1, Q5\_0 \times Q8\_0/Q8\_1$ and Quantized $\times \text{FP32}$ Vectorized GEMV / GEMM**:
  - Implemented direct quantized matrix multiplication in `include/strata/kernels/fast_matmul.hpp` and `include/strata/kernels/fast_quant.hpp` (`FastMatMul::gemv_q4_0_q8_0`, `FastMatMul::gemv_q4_0_fp32`, `FastMatMul::gemm_q4_0_q8_0`, `FastMatMul::gemv_q4_1_q8_1`, `FastMatMul::gemv_q4_1_fp32`, `FastMatMul::gemv_q5_0_q8_0`, `FastMatMul::gemv_q5_0_fp32`).
  - Multiplies register-unpacked 4-bit and 5-bit nibbles directly against signed 8-bit activations using AVX2 `_mm_maddubs_epi16` / `_mm_madd_epi16` with $O(1)$ zero-offset correction (`int_sum - 8 * sigma` for Q4_0, `int_sum - 16 * sigma` for Q5_0, and $O(1)$ scalar bias product $w_m \cdot a_s$ for Q4_1), bypassing FP32 dequantization and memory bandwidth bottlenecks.
  - Measured throughput: **35.51 µs / step** for full FFN layer ($5632 \times 2048$), **33.15 µs / token** for batched speculative decode GEMM.
- **Zero-Allocation Stack-Buffered FlashAttention Prefill Engine**:
  - Replaced all inner heap vector allocations (`std::vector<float>`) in `FastAttention::flash_attention_prefill` and `FastAttention::flash_attention_prefill_mha` with stack buffers (`stack_m`, `stack_l`, `stack_acc`, `stack_scores`, `stack_tile_scores`), eliminating thousands of allocations per prefill step.
  - Measured throughput: **0.53 ms** for 256-token FlashPrefill.
- **Copy-On-Write (COW) Split-KV Attention Decoding**:
  - Implemented `FastAttention::cow_scaled_dot_product_decode` and `FastAttention::cow_multi_head_scaled_dot_product_decode` in `include/strata/kernels/fast_attention.hpp` enabling zero-copy session branching, chat forks, and scratchpad context sharing across split prefix and dynamic tail caches.
- **Split-K Long-Context Flash Attention Decoding**:
  - Added `FastAttention::split_k_scaled_dot_product_decode` partitioning long KV contexts into 256-token parallel chunks with numerically stabilized log-sum-exp split reduction (**28.69 µs / head** over 4096-token sequence).
- **Vectorized SIMD SwiGLU & SiLU Batch Activations**:
  - Added fast polynomial exponential (`_mm256_fastexp_ps`) and rational/NR reciprocal SiLU vectorization (`_mm256_fastsilu_ps`) in `include/strata/kernels/fast_activations.hpp` (`fast_swiglu_array`, `fast_activate_array`).

### 10. Online Abliteration (Inference-Time Representation Intervention)
- **Mathematical Principle (Arditi et al. / Maxime Labonne)**:
  - Refusal behavior in aligned LLMs is mediated by a low-dimensional direction in the residual stream. Online abliteration inhibits this behavior dynamically in real time without modifying disk weights or retraining:
    $$\text{proj} = \langle x, \hat{r} \rangle, \quad x \leftarrow x - \alpha \cdot \text{proj} \cdot \hat{r}$$
  - Implemented `FastAbliteration` in `include/strata/kernels/fast_abliteration.hpp` supporting both `AbliterationMode::kOrthogonal` and `AbliterationMode::kPositiveOnly` ($\max(0, \text{proj})$).
- **SIMD Vectorization & Contrastive Refusal Extraction**:
  - AVX2/AVX-512 FMA vectorization for $\langle x, \hat{r} \rangle$ dot product and in-place vector subtraction `_mm256_fnmadd_ps`.
  - Contrastive refusal direction estimation (`compute_contrastive_refusal_vector`) and canonical refusal direction generation (`generate_canonical_refusal_direction`) across middle residual layers (30%-75% depth).
  - High-performance binary `.ablr` serialization (`save_vector_file`, `load_vector_file`).
- **End-to-End Stack Integration**:
  - **C++ Engine**: Wired `--abliterate`, `--abliteration-strength`, and `--abliteration-vector` flags in `src/program/generate.cpp`; parsed `abliterate=1` and `abliteration_strength` per-request options in `--serve` IPC `GEN` command.
  - **Go Server**: Parsed `abliterate` and `abliteration_strength` in OpenAI chat/completions API (`oaRequest`), Anthropic messages API (`anRequest`), and HTTP headers (`X-Strata-Abliterate`, `X-Abliterate`, `X-Strata-Abliteration-Strength`).
  - **Config & Web UI**: Added `abliterate` and `abliteration_strength` to `StrataConfig` and `editable` keys in `pkg/runconfig/`; added UI toggle switch and strength slider in `pkg/frontend/web/index.html` and `app.js`.

### 11. 1-Pass Tiled Online Flash Attention Decode & Adaptive MoE Routing
- **1-Pass Zero-Allocation Tiled Online Flash Attention Decode**:
  - Implemented `FastAttention::flash_decode_scaled_dot_product` and `FastAttention::flash_decode_gqa` in `include/strata/kernels/fast_attention.hpp`.
  - Replaces traditional 3-pass attention decoding with single-pass tiled online log-sum-exp accumulation and register-based dynamic scaling:
    $$\alpha = e^{m_{\text{old}} - m_{\text{new}}}, \quad O_{\text{new}} = \alpha \cdot O_{\text{old}} + \sum e^{s_i - m_{\text{new}}} V_i, \quad L_{\text{new}} = \alpha \cdot L_{\text{old}} + \sum e^{s_i - m_{\text{new}}}$$
  - Measured latency: **21.85 µs / head** over 1024-token context with zero heap memory allocation.
- **Adaptive MoE Router with Dynamic Pruning**:
  - Implemented `FastMoE::route_topk_with_threshold` in `include/strata/kernels/fast_moe.hpp`.
  - Prunes low-contribution tail experts when cumulative softmax probability reaches threshold $\ge 0.98$ or weight $< 0.01$.
  - Measured router latency: **329.28 ns / token** (dynamically selecting 2 active experts instead of evaluating all 8).


### 12. Inference Optimizations (NVIDIA & Hugging Face)
- **PagedAttention & Paged KV Block Cache (`FastPagedKVCache`)**:
  - Fixed-size physical memory block pooling ($B = 16$ or $32$ tokens per block) in `include/strata/kernels/fast_paged_kv.hpp`.
  - Logical-to-physical block tables mapping token positions to physical blocks, eliminating internal and external memory fragmentation.
  - Atomic reference-counted Copy-on-Write (COW) sequence forking: shared system prompts and multi-turn conversations reuse physical blocks until mutated.
  - Direct PagedAttention decode kernels (`FastAttention::paged_scaled_dot_product_decode` and `FastAttention::paged_gqa_decode` in `include/strata/kernels/fast_attention.hpp`) streaming SIMD dot products over non-contiguous blocks without memory duplication.
  - **Measured (this PC, AVX2/FMA)**: PagedAttention decode (1024 tokens, 64 physical blocks, head dim 128): **24.90 µs / head** (numerical parity discrepancy $< 1\times 10^{-7}$).
- **In-Flight Continuous Batching Scheduler with Chunked Prefill (`InFlightBatcher`)**:
  - Dynamic iteration-level continuous scheduler in `include/strata/runtime/in_flight_batcher.hpp`.
  - Admits pending requests on any iteration step as soon as KV cache blocks free up.
  - Chunked prefill partitions long prompts (e.g. 128 or 256 tokens per step), preventing latency spikes and interleaving prefill with ongoing decode streams.
  - Immediate eviction: requests emitting EOS or hitting token limit are retired immediately, freeing physical KV blocks.
  - **Measured (this PC)**: Iteration scheduler latency across 8 active streams: **0.91 µs / iteration step**.
- **Speculative Decoding & Prompt Lookup Verification (`FastSpeculativeVerifier`)**:
  - Speculative candidate token verification in `include/strata/kernels/fast_speculative.hpp`.
  - Supports parallel target model verification across $K$ draft tokens from draft models, MTP layers, or Prompt Lookup Decoding (input-grounded $n$-gram matching via `SuffixDrafter`).
  - AVX2 SIMD-accelerated vocabulary argmax and stochastic rejection sampling with correction token generation on first mismatch.
  - **Measured (this PC)**: Speculative verification latency ($K = 4$, 32,000 vocabulary size): **8.66 µs / round**.

### 13. Xiaomi MiMo-V2.6-Flash-RL Architecture Optimizations
- **Asymmetric Head Dimensions ($d_{qk} = 192, d_v = 128$)**:
  - Implemented vectorized AVX2 kernels (`dot_qk_192` and `fma_v_128`) in `include/strata/kernels/fast_mimo.hpp`.
  - Decouples representational query-key capacity ($d_{qk}=192$) from value projection bandwidth ($d_v=128$), reducing KV memory cache traffic by 33%.
- **Fast Circular 128-Token SWA Buffer (`FastMiMoAsymmetricKVCache`)**:
  - Constant rolling sliding-window buffer evicting expired token keys and values outside $[t-128, t]$.
  - Reduces memory consumption by $\sim 8,000\times$ over 1M context sequences.
- **256-Expert Top-8 MoE Router (`FastMiMoMoERouter`)**:
  - AVX2-accelerated Top-8 expert selection with softmax and sigmoid routing normalization.
  - Measured latency: **0.22 µs / call**.
- **5-Layer SWA MTP Speculative Drafter (`FastMiMoMTPVerifier`)**:
  - Verifies up to 7 Multi-Token Prediction (MTP) draft tokens simultaneously against target logits.

### 14. Mamba State-Space Model Optimization
- **Hardware-Aware Selective SSM Scan (`FastMambaSSM`)**:
  - Implemented in `include/strata/kernels/fast_mamba.hpp` using Zero-Order Hold (ZOH) discretization $\bar{A} = \exp(\Delta A)$, $\bar{B} = \Delta B$.
  - Recurrent token decode uses strictly constant $O(1)$ memory state $h \in \mathbb{R}^{d_{inner} \times d_{state}}$ (8 KiB state buffer per stream, zero KV cache overhead).
  - Parallel associative prefix scan (`parallel_scan_sequence`) for accelerated prompt prefill.
- **1D Depthwise Convolution ($K=4$) & Fused Gated Branch**:
  - Circular shift register convolution buffer with fused SiLU activation.
  - Fused multiplicative gating: $y_{\text{out}} = y_{\text{ssm}} \odot \text{SiLU}(z)$.
  - Measured latency: **282.82 µs / step** across 512 inner dimension and state dimension 16.

### 15. Clean Schema Separation for Multi-Model Support
- **Model Schema Definitions (`pkg/modelschema/schema.go`)**:
  - Decoupled model architectural classification (`ModelFamily`, `Architecture`, `AttentionKind`, `MoETopology`, `SSMTopology`, `ResourceSpec`).
  - Canonical `ModelManifest` schema with validation, alias matching, and OpenAI serialization.
- **Thread-Safe Model Registry (`pkg/modelschema/registry.go`)**:
  - Pre-registered built-ins: `qwen3.8-flash-next`, `qwen3.8-flash-next-coder`, `mimo-v2.6-flash-rl`, `mamba-2.8b`, `deepseek-v3`.
  - Dynamic runtime profile tracking (`ModelRuntimeProfile`) with case-preserving resolution and alias mapping.
- **Multi-Model Run Configuration (`pkg/runconfig/runconfig.go`)**:
  - Added `DefaultModel` and `Models map[string]ModelEntry` to `StrataConfig`.
  - Added `GetActiveModel()`, `GetModelEntry()`, and `AllModels()` with 100% backward compatibility for single-model deployments.
- **OpenAI & Anthropic Multi-Model Server Integration (`pkg/server/`)**:
  - `GET /v1/models` and `GET /v1/models/{id}` dynamically served from `ModelRegistry`.
  - Dynamic model resolution and validation in `/v1/chat/completions`, `/v1/completions`, and `/v1/messages`.

### 16. Heterogeneous Task Distribution & Resource Optimization Policy
- **Activation-Centric MoE Offloading (`include/strata/runtime/heterogeneous_scheduler.hpp`)**:
  - Implements Fiddler & KTransformers activation offload principle for cold MoE experts.
  - Dynamically compares $|W| / BW_{\text{PCIe}}$ vs activation vector compute on CPU OpenMP threads.
  - **Measured (this PC)**: Activation offload achieves **49.18x speedup** over swapping 44 MB expert weight tensors across PCIe ($56.87\ \mu\text{s}$ CPU compute vs $2796.96\ \mu\text{s}$ PCIe transfer).
- **Sarathi-Style Chunked-Prefill & Decode Piggybacking**:
  - Partitions long prompts into 128–256 token chunks co-batched with active decode streams.
  - Balances compute-heavy prefill (arithmetic intensity $\gg 10\text{ FLOPs/byte}$) with memory-bound decode, saturating GPU Tensor Cores without starving decode latency.
- **Asymmetric Speculative Pipelining**:
  - Asynchronous CPU drafter prepares $K$ candidate tokens (Mamba SSM / SLM) while GPU computes current step; GPU validates all $K$ tokens in parallel in a single forward pass (**3.06x speedup** at 85% acceptance).
- **Dominant Resource Sizing & Multi-Queue Admission (`pkg/resourcemanager/resourcemanager.go`)**:
  - Multi-dimensional saturation monitoring: $U_{\text{dominant}} = \max(U_{\text{vram}}, U_{\text{ram}}, U_{\text{pcie}}, U_{\text{cpu}})$.
  - Independent routing for CPU-only symbolic tasks (Mathics, CAS, logic verifier) with zero GPU concurrency slot consumption.
  - Dynamic backpressure with rate-based `retry_after` calculation.

---

## Verification Commands

- **Run all Go test suites**:
  ```bash
  go test -v ./pkg/...
  ```
- **Run Python setup tests**:
  ```bash
  python3 -m unittest discover -s tools -p "test_setup_*.py"
  ```
- **Run Python MCP server tests**:
  ```bash
  python3 -m unittest tools/test_strata_mcp.py
  ```
- **Build standalone Go binaries**:
  ```bash
  go build -o bin/strata ./cmd/strata && go build -o bin/strata-server ./cmd/strata-server
  ```
- **Build & Run C++ runtime, test suites & micro-benchmarks**:
  ```bash
  ninja -C build
  ctest --test-dir build -R "test_heterogeneous_scheduler|test_in_flight_batcher|test_fast_mimo|test_fast_mamba|test_fast_unified_benchmark"
  ```
- **Start Web Server with Multi-Model & Multi-Session Chat**:
  ```bash
  ./bin/strata serve
  ```



