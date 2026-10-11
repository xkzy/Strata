# Design: End-to-End Python to Go Migration for Strata

**Date:** 2026-10-12  
**Target:** Replace all Python scripts, installers, model packagers, and MCP tooling with native, zero-dependency Go packages unified into `bin/strata`.

---

## 1. Objectives & Architectural Overview

Strata currently uses Go for its inference server (`pkg/server`, `cmd/strata`), web interface (`pkg/frontend`), mathematical runtime (`pkg/mathruntime`), and logical verification engines. However, model installation (`setup.py`), packing tools (`tools/iq_pack.py`, `tools/mtp_pack.py`), GGUF reading (`tools/gguf_reader.py`), and the Model Context Protocol server (`tools/strata_mcp.py`) still require Python 3 and multiple pip dependencies (`numpy`, `psutil`, `jinja2`, `requests`).

This design migrates all remaining Python functionality into native Go:
- **Zero Python Runtime Requirement:** Users, developers, and AI agents only need the single compiled `bin/strata` executable and the C++ engine.
- **Unified Single Binary:** `bin/strata` provides all operational modes through dedicated subcommands:
  - `strata setup`: Hardware probe, interactive/scripted installer, HF downloader, and configuration writer (replacing `setup.py`).
  - `strata pack`: Fast concurrent GGUF parser and tensor packer for dense/expert packs (replacing `tools/iq_pack.py` and `tools/mtp_pack.py`).
  - `strata mcp`: High-performance JSON-RPC 2.0 Model Context Protocol server over stdio or HTTP/SSE (replacing `tools/strata_mcp.py`).
  - `strata bench`: Native benchmarks for tokens/sec, TTFT, long-context latency, and memory telemetry (replacing `tools/benchmark_*.py`).
  - `strata serve`: Existing high-performance inference server & Web UI.

---

## 2. Package Decomposition & Responsibilities

```
cmd/
  strata/
    main.go                   # Master CLI dispatcher: serve, setup, pack, mcp, bench, math, verify

pkg/
  gguf/
    reader.go                 # Binary GGUF v2/v3 reader: headers, metadata KVs, tensor descriptors
    writer.go                 # GGUF file writer for conversion and MTP layer packing
    types.go                  # GGML tensor types (F32, F16, Q4_K, IQ3_XXS, MXFP4, etc.)
    reader_test.go            # Binary round-trip and header validation tests

  pack/
    packer.go                 # Native pack generator: index.txt, dense.bin, native_experts.txt
    experts.go                # Blob extraction [gate | up | down] into experts.bin
    conversions.go            # FP32 <-> BF16 / FP16 exact narrowing/widening and conversions.json
    draft_layer.go            # MTP draft vocabulary pruning and draft layer packing (mtp_pack.go)
    packer_test.go            # Unit and fixture tests for tensor layout parity

  mcp/
    server.go                 # JSON-RPC 2.0 stdio / SSE transport protocol handler
    tools.go                  # Tools implementation: hardware, models, setup, status, bench, logs
    schema.go                 # MCP schema definitions compatible with Anthropic/OpenAI agents
    server_test.go            # Automated JSON-RPC protocol roundtrip tests

  installer/
    hardware.go               # Host detection: Linux /proc/meminfo, Windows GlobalMemoryStatusEx, nvidia-smi, ROCm rocm-smi
    wizard.go                 # Interactive terminal wizard and CLI flags (--family, --model, --context, etc.)
    downloader.go             # Parallel HTTP range downloader with resume, SHA-256 verification, and terminal progress
    config_writer.go          # Config generator: strata-<tag>.json and launcher scripts (run.sh / run.bat)
    engine_builder.go         # C++ engine compile trigger or prebuilt release extraction
    installer_test.go         # Extended test suite matching Python test_setup_* cases

  bench/
    engine.go                 # Benchmark runner for inference engine and HTTP API endpoints
    metrics.go                # Statistical calculator: TTFT, prompt tokens/sec, generation tokens/sec, p50/p95/p99
    bench_test.go             # Benchmark execution tests
```

---

## 3. Detailed Component Designs

### 3.1. Native GGUF Parser (`pkg/gguf`)
- **Binary Format Compliance**: Supports GGUF magic `0x46554747` (`GGUF`), versions 2 and 3.
- **Zero-Copy Tensor Slicing**: Uses memory mapping (`syscall.Mmap` / `golang.org/x/exp/mmap`) or offset file seeks to slice tensor blobs without loading multi-gigabyte models into system heap.
- **Metadata KV Reader**: Decodes string, int8..uint64, float32/64, bool, array, and nested arrays for tokenizer configs and tensor mappings.

### 3.2. Native Model Packager (`pkg/pack`)
- Replaces `tools/iq_pack.py` and `tools/mtp_pack.py`.
- **Expert Extraction**: Scans GGUF shards for MoE layers (`blk.N.ffn_gate_exps.weight`, `blk.N.ffn_up_exps.weight`, `blk.N.ffn_down_exps.weight`), extracts raw slice bytes, and writes `native_experts.txt` and `experts.bin`.
- **Dense Packing**: Identifies attention and router weights, performs exact BF16/F16 conversion for engine form compliance, and produces `dense.bin` + `conversions.json`.
- **MTP Drafting**: Reads `per_layer_token_embd.weight` and vocabulary tokens, generates localized sub-vocabularies (`cjk`, `en`, `fr`, `cyrillic`), and generates runtime draft weights.

### 3.3. Installer & Setup Engine (`pkg/installer`)
- Replaces `setup.py` (5,115 lines):
  - **Hardware Engine**: Reads physical RAM, available VRAM across all NVIDIA / AMD GPUs, AVX/AVX2/AVX-512 CPU flags.
  - **Catalog & Recommendations**: Evaluates VRAM + RAM budgets across Qwen, Swift, Coder, Unsloth, MiMo, and OpenAI GPT-OSS families.
  - **Download Manager**: Native concurrent streaming HTTP client with resume support (`Range` headers), SHA-256 verification against pinned revision hashes, and real-time ANSI terminal progress bars.
  - **Output Generation**: Writes standard `strata-<family>-<quant>.json` run configuration and shell scripts (`run.sh` / `run.bat`).

### 3.4. Model Context Protocol Server (`pkg/mcp`)
- Replaces `tools/strata_mcp.py`:
  - **JSON-RPC 2.0 Engine**: Compliant with Anthropic Claude Desktop, Cursor, and Gemini tool specifications over standard input/output.
  - **Tools Exposed**:
    - `strata_hardware`: Report CPU, system RAM, GPUs, and driver status.
    - `strata_models`: List supported and locally installed models with memory requirements.
    - `strata_setup`: Execute unattended non-interactive installation and pack generation.
    - `strata_status`: Check health, memory residency, and telemetry of running engine.
    - `strata_benchmark`: Trigger automated generation benchmarks.

---

## 4. Verification & Testing Strategy

1. **GGUF & Pack Parity**:
   - Compare binary output and text files (`native_experts.txt`, `dense.bin`, `index.txt`) generated by `strata pack` against outputs from `tools/iq_pack.py` using test fixtures.
2. **Setup Behavioral Parity**:
   - Convert Python test suites (`test_setup_amd.py`, `test_setup_choices.py`, `test_setup_risk.py`, etc.) into Go tests under `pkg/installer/` to assert exact compatibility.
3. **MCP Specification Tests**:
   - Run standard JSON-RPC request-response integration test suites against `strata mcp`.
4. **End-to-End Workflow**:
   - Test full sequence on clean system without Python:
     `bin/strata setup --yes --family openai --model GPT_OSS_20B` -> `bin/strata serve` -> API request.
