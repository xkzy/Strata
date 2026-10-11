# Python to Golang End-to-End Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Completely replace all Python tools, setup scripts, GGUF packagers, and MCP servers with native, zero-dependency Golang packages unified into `bin/strata`.

**Architecture:** Implement high-performance native Go packages (`pkg/gguf`, `pkg/pack`, `pkg/mcp`, `pkg/installer`, `pkg/bench`) and dispatch them through unified subcommands in `cmd/strata/main.go` (`strata setup`, `strata pack`, `strata mcp`, `strata bench`, `strata serve`).

**Tech Stack:** Go 1.22+, `syscall` / standard library, zero external Python runtime dependencies.

## Global Constraints
- Zero runtime Python requirement for Strata installation, packing, serving, or MCP tooling.
- Single unified binary: all commands must be accessible via `bin/strata <subcommand>`.
- Strict binary format compliance for GGUF v2/v3 and Strata pack formats (`index.txt`, `dense.bin`, `native_experts.txt`, `experts.bin`).
- All Go packages must pass `go test -v ./...` without external internet access during testing.

---

### Task 1: Native GGUF Binary Reader & Tensor Parser (`pkg/gguf`)

**Files:**
- Create: `pkg/gguf/types.go`
- Create: `pkg/gguf/reader.go`
- Create: `pkg/gguf/writer.go`
- Test: `pkg/gguf/reader_test.go`

**Interfaces:**
- Produces:
  - `type GGUFFile struct { Version uint32; Metadata map[string]any; Tensors []TensorInfo; ... }`
  - `func Open(path string) (*GGUFFile, error)`
  - `func (g *GGUFFile) GetTensor(name string) (*TensorInfo, bool)`
  - `func (g *GGUFFile) ReadTensorBytes(t *TensorInfo) ([]byte, error)`

- [x] **Step 1: Write failing unit test for GGUF binary decoding**

Create `pkg/gguf/reader_test.go` with synthetic GGUF v3 binary stream asserting magic `GGUF`, version 3, tensor count, and key-value metadata parsing.

- [x] **Step 2: Run test to verify it fails**

Run: `go test ./pkg/gguf -v`  
Expected: Compilation failure (package `pkg/gguf` does not exist).

- [x] **Step 3: Implement GGUF types and binary reader**

Implement `pkg/gguf/types.go` and `pkg/gguf/reader.go`:
- Magic validation (`0x46554747`).
- Metadata key-value types: uint8..uint64, int8..int64, float32, float64, bool, string, array.
- Tensor descriptor table parsing: name string, number of dimensions, dimensions `uint64[]`, type `uint32`, byte offset `uint64`.
- Memory-safe tensor data reader.

- [x] **Step 4: Run test to verify it passes**

Run: `go test ./pkg/gguf -v`  
Expected: PASS.

- [x] **Step 5: Implement GGUF writer for conversions and MTP drafting**

Implement `pkg/gguf/writer.go` and test round-trip metadata writing and tensor serialization in `reader_test.go`.

- [x] **Step 6: Run full package tests**

Run: `go test ./pkg/gguf -v`  
Expected: PASS.

---

### Task 2: Native Model Packager (`pkg/pack`)

**Files:**
- Create: `pkg/pack/types.go`
- Create: `pkg/pack/conversions.go`
- Create: `pkg/pack/experts.go`
- Create: `pkg/pack/packer.go`
- Create: `pkg/pack/draft_layer.go`
- Test: `pkg/pack/packer_test.go`
- Modify: `cmd/strata/main.go` (register `strata pack`)

**Interfaces:**
- Consumes: `pkg/gguf.GGUFFile`
- Produces:
  - `type PackOptions struct { GGUFPath string; OutDir string; BasePack string; ExpertsBin bool; CompatBF16 bool }`
  - `func PackModel(opts PackOptions) error`
  - `func GenerateDraftVocab(vocabTokens []string, script string) ([]int32, error)`

- [x] **Step 1: Write failing test for tensor conversion and pack generation**

Create `pkg/pack/packer_test.go` testing exact BF16/FP16 conversion, MoE slice ordering, `native_experts.txt`, and `index.txt` output generation.

- [x] **Step 2: Run test to verify it fails**

Run: `go test ./pkg/pack -v`  
Expected: Compilation failure.

- [x] **Step 3: Implement BF16/F16 conversions and MoE blob packing**

Implement `pkg/pack/conversions.go` and `pkg/pack/experts.go`:
- Float conversions matching Strata's `FORM` table (routers, projections as BF16, norms as F32).
- Expert slicing `[gate | up | down]` with per-layer blob tracking.
- Output generation of `index.txt`, `native_experts.txt`, `conversions.json`, and optional `experts.bin`.

- [x] **Step 4: Implement MTP draft vocab generator (`pkg/pack/draft_layer.go`)**

Port `draft_vocab.py` and `mtp_pack.py` into Go for vocabulary pruning (`cjk`, `en`, `fr`, `cyrillic`).

- [x] **Step 5: Wire `strata pack` into `cmd/strata/main.go`**

Expose `strata pack --gguf <file> --out <dir> [--experts-bin] [--compat-bf16]` CLI command.

- [x] **Step 6: Run tests and verify**

Run: `go test ./pkg/pack -v`  
Expected: PASS.

---

### Task 3: Native Model Context Protocol (MCP) Server (`pkg/mcp`)

**Files:**
- Create: `pkg/mcp/protocol.go`
- Create: `pkg/mcp/server.go`
- Create: `pkg/mcp/tools.go`
- Test: `pkg/mcp/server_test.go`
- Modify: `cmd/strata/main.go` (register `strata mcp`)

**Interfaces:**
- Produces:
  - `type Server struct { ... }`
  - `func NewServer() *Server`
  - `func (s *Server) ServeStdio(ctx context.Context) error`
  - `func (s *Server) ServeHTTP(addr string) error`

- [x] **Step 1: Write failing test for MCP JSON-RPC protocol**

Create `pkg/mcp/server_test.go` verifying `initialize`, `tools/list`, and `tools/call` for `strata_hardware`, `strata_models`, and `strata_status`.

- [x] **Step 2: Run test to verify it fails**

Run: `go test ./pkg/mcp -v`  
Expected: Compilation failure.

- [x] **Step 3: Implement JSON-RPC 2.0 transport and tool dispatcher**

Implement `pkg/mcp/protocol.go`, `pkg/mcp/server.go`, and `pkg/mcp/tools.go`:
- Read/write newline-delimited JSON-RPC 2.0 messages over `os.Stdin` / `os.Stdout`.
- Tools:
  - `strata_hardware`: Host telemetry.
  - `strata_models`: Model catalog and sizes.
  - `strata_setup`: Automated installation trigger.
  - `strata_status`: Server health and residency metrics.
  - `strata_benchmark`: Performance benchmark execution.

- [x] **Step 4: Wire `strata mcp` into `cmd/strata/main.go`**

Expose `strata mcp [--stdio|--http]` in `cmd/strata/main.go`.

- [x] **Step 5: Run tests and verify**

Run: `go test ./pkg/mcp -v`  
Expected: PASS.

---

### Task 4: Native Full Installer & Hardware Setup Engine (`pkg/installer`)

**Files:**
- Modify: `pkg/installer/installer.go`
- Create: `pkg/installer/wizard.go`
- Create: `pkg/installer/downloader.go`
- Create: `pkg/installer/config_writer.go`
- Modify: `pkg/installer/installer_test.go`
- Modify: `cmd/strata/main.go` (update `strata setup`)

**Interfaces:**
- Produces:
  - `func RunInteractiveWizard() error`
  - `func RunNonInteractiveSetup(flags SetupFlags) error`
  - `func DownloadModelFiles(repo string, files []string, destDir string) error`
  - `func WriteRunConfig(cfg RunConfig, path string) error`

- [x] **Step 1: Write failing test for config and script generation**

Add tests to `pkg/installer/installer_test.go` verifying run config output (`strata-<model>.json`), launcher scripts (`run.sh` / `run.bat`), and budget calculation.

- [x] **Step 2: Run test to verify it fails**

Run: `go test ./pkg/installer -v`  
Expected: Failure on missing wizard/config functions.

- [x] **Step 3: Implement streaming downloader with resume and progress**

Implement `pkg/installer/downloader.go`:
- HTTP chunked streaming with `Range: bytes=X-` support.
- Real-time ANSI progress bar with download speed (MB/s) and ETA.
- Post-download SHA-256 verification against pinned table.

- [x] **Step 4: Implement interactive wizard and configuration writer**

Implement `pkg/installer/wizard.go` and `pkg/installer/config_writer.go`:
- Context length prompts, VRAM budgeting, low-RAM SSD streaming calculations.
- Config output matching engine requirements (`--native`, `--pack`, `--expert-cache auto`, `--prefill auto`, `--resident-budget-gib`).

- [x] **Step 5: Wire full `strata setup` CLI in `cmd/strata/main.go`**

Support all options: `--family`, `--model`, `--context`, `--low-ram`, `--resident-budget-gib`, `--yes`, `--check`.

- [x] **Step 6: Run tests and verify**

Run: `go test ./pkg/installer -v`  
Expected: PASS.

---

### Task 5: Native Benchmarking Tool (`pkg/bench`)

**Files:**
- Create: `pkg/bench/bench.go`
- Create: `pkg/bench/metrics.go`
- Test: `pkg/bench/bench_test.go`
- Modify: `cmd/strata/main.go` (register `strata bench`)

**Interfaces:**
- Produces:
  - `type BenchmarkReport struct { TTFTMs float64; PromptTokensPerSec float64; GenTokensPerSec float64; ... }`
  - `func RunBenchmark(opts BenchmarkOptions) (*BenchmarkReport, error)`

- [x] **Step 1: Write failing test for benchmark engine**

Create `pkg/bench/bench_test.go` verifying latency calculation, percentile computation (p50/p95/p99), and token throughput.

- [x] **Step 2: Implement benchmark runner and metrics calculator**

Implement `pkg/bench/bench.go` and `pkg/bench/metrics.go`.

- [x] **Step 3: Wire `strata bench` in `cmd/strata/main.go`**

Expose `strata bench [--port 8080] [--prompt "..."] [--tokens 512]` subcommand.

- [x] **Step 4: Run tests and verify**

Run: `go test ./pkg/bench -v`  
Expected: PASS.

---

### Task 6: Master CLI Unification & Zero-Python Verification

**Files:**
- Modify: `cmd/strata/main.go`
- Modify: `setup.sh`
- Modify: `START-HERE.bat`
- Test: Full integration test

- [x] **Step 1: Update `setup.sh` and `START-HERE.bat`**

Ensure shell and batch scripts build and execute `bin/strata` natively without any Python checks or invocation.

- [x] **Step 2: Run all Go test suites across the repository**

Run: `go test -v ./pkg/...`  
Expected: All packages pass.

- [x] **Step 3: Rebuild native binaries**

Run: `go build -o bin/strata ./cmd/strata && go build -o bin/strata-server ./cmd/strata-server`  
Expected: Clean build, zero warnings.

- [x] **Step 4: Execute verification commands**

Run `bin/strata --version` and `bin/strata setup --check`. Verify zero Python processes or imports.
