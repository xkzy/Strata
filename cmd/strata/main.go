// cmd/strata/main.go - Unified Strata CLI & Engine Driver in Go
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"strata/pkg/bench"
	"strata/pkg/hallucination"
	"strata/pkg/installer"
	"strata/pkg/logicverifier"
	"strata/pkg/mathruntime"
	"strata/pkg/mcp"
	"strata/pkg/pack"
	"strata/pkg/server"
	"strata/pkg/telemetry"
	"strata/pkg/tokenizer"
)

const version = "1.0.0"

func main() {
	if len(os.Args) < 2 {
		configs, _ := filepath.Glob("strata-*.json")
		if len(configs) > 0 {
			runServer(nil)
			return
		}
		runSetup(nil)
		return
	}

	command := os.Args[1]

	switch command {
	case "serve", "server":
		runServer(os.Args[2:])
	case "setup":
		runSetup(os.Args[2:])
	case "pack":
		runPack(os.Args[2:])
	case "tokenize":
		runTokenize(os.Args[2:])
	case "profile", "make-profile":
		runProfile(os.Args[2:])
	case "calibrate":
		runCalibrate(os.Args[2:])
	case "needle", "needle-bench":
		runNeedle(os.Args[2:])
	case "mcp":
		runMCP(os.Args[2:])
	case "bench", "benchmark":
		runBench(os.Args[2:])
	case "math":
		runMath(os.Args[2:])
	case "verify":
		runVerify(os.Args[2:])
	case "check-hallucination", "hallucination":
		runCheckHallucination(os.Args[2:])
	case "telemetry", "status":
		runTelemetry(os.Args[2:])
	case "version", "--version", "-v":
		fmt.Printf("Strata Native AI Server & Mathematical Engine v%s (linux/amd64)\n", version)
	case "help", "--help", "-h":
		printUsage()
	default:
		// Default to serve if flag is passed
		if len(command) > 0 && command[0] == '-' {
			runServer(os.Args[1:])
			return
		}
		fmt.Printf("Unknown command '%s'. Run 'strata help' for usage.\n", command)
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("======================================================================")
	fmt.Println("   STRATA UNIFIED AI ENGINE, MATHEMATICAL CAS & SERVER (GOLANG)       ")
	fmt.Println("======================================================================")
	fmt.Println("Usage: strata <command> [arguments]")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  serve               Start the high-performance AI inference server & Web UI")
	fmt.Println("  setup               Detect hardware & configure/download optimal model")
	fmt.Println("  pack                Prepare GGUF weights, extract MoE expert blobs into Strata pack")
	fmt.Println("  profile             Create or update expert routing profile (STRP) from trace")
	fmt.Println("  calibrate           Calibrate PCIe transfer fraction and speculation settings")
	fmt.Println("  needle              Run needle-in-a-haystack long-context recall benchmark")
	fmt.Println("  mcp                 Start JSON-RPC Model Context Protocol server for AI tools")
	fmt.Println("  bench               Benchmark inference token throughput, TTFT, and latency")
	fmt.Println("  math                Evaluate a mathematical expression deterministically")
	fmt.Println("  verify              Verify logical deductions, constraints, SI units, schemas")
	fmt.Println("  check-hallucination Detect and verify factual claims against ground truth")
	fmt.Println("  telemetry           Print current CPU, RAM, GPU/VRAM hardware telemetry")
	fmt.Println("  version             Print version information")
	fmt.Println()
}

func runServer(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	port := fs.Int("port", 8080, "Port to listen on")
	host := fs.String("host", "127.0.0.1", "Address to listen on (anything but loopback requires --api-key)")
	rtMode := fs.String("rt", "auto", "C++ transparent runtime (verification, virtual context): auto, on or off")
	rtBinary := fs.String("rt-binary", "", "Path to strata_rt_server (default: looked for next to the engine)")
	window := fs.Int("window", 65536, "Physical context window the runtime fills per request (tokens)")
	modelName := fs.String("model", "", "Model identifier (default: from saved strata-config.json)")
	maxContext := fs.Int("max-context", 0, "Physical context limit (default: from saved strata-config.json)")
	virtualLimit := fs.Int("virtual-limit", 0, "Virtual context limit (default: from saved strata-config.json)")
	apiKey := fs.String("api-key", "", "API key for auth")
	modelPath := fs.String("model-path", "", "Path to model weights (.gguf / .bin)")
	binaryPath := fs.String("binary-path", "", "Path to C++ strata engine binary")
	configFile := fs.String("config", "", "Path to strata-<model>.json config")
	fs.Parse(args)

	cfg := server.ServerConfig{
		Port:          *port,
		ModelName:     *modelName,
		MaxContext:    *maxContext,
		VirtualLimit:  *virtualLimit,
		APIKey:        *apiKey,
		ModelPath:     *modelPath,
		BinaryPath:    *binaryPath,
		ConfigFile:    *configFile,
		BindHost:      *host,
		RuntimeMode:   *rtMode,
		RuntimeBinary: *rtBinary,
		WindowTokens:  *window,
	}

	if err := server.CheckBind(*host, *apiKey); err != nil {
		fmt.Fprintln(os.Stderr, "strata:", err)
		os.Exit(2)
	}
	srv := server.NewStrataServer(cfg)
	if err := srv.StartEngine(context.Background()); err != nil {
		fmt.Fprintf(os.Stderr, "strata: the model is not available: %v\n", err)
	} else {
		fmt.Fprintln(os.Stderr, "strata: loading the model in the background (requests get 503 until it is ready; see /health)")
	}
	handler := srv.Router()

	httpServer := &http.Server{
		Addr:         net.JoinHostPort(*host, fmt.Sprint(*port)),
		Handler:      handler,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 0, // replies stream for as long as the model thinks
	}

	fmt.Println("======================================================================")
	fmt.Println("   STRATA HIGH-PERFORMANCE AI SERVER & MATHEMATICAL ENGINE (GOLANG)   ")
	fmt.Println("======================================================================")
	fmt.Printf(" Model Name:        %s\n", srv.Config.ModelName)
	if srv.Config.ModelPath != "" {
		fmt.Printf(" Model Path:        %s\n", srv.Config.ModelPath)
	}
	fmt.Printf(" Web UI & API:      http://127.0.0.1:%d\n", srv.Config.Port)
	fmt.Printf(" Physical Context:  %d tokens\n", srv.Config.MaxContext)
	fmt.Printf(" Virtual Context:   %d tokens\n", srv.Config.VirtualLimit)
	fmt.Printf(" Math Runtime:      Active (Mathics CAS + FastNumeric Exact Rational)\n")
	fmt.Printf(" Logic Verifier:    Active (Tri-State Modus Ponens, SI Units, Bounds)\n")
	fmt.Println("======================================================================")

	go func() {
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	fmt.Println("\nShutting down Strata Go Server gracefully...")
}

func runMath(args []string) {
	if len(args) == 0 {
		fmt.Println("Usage: strata math \"<expression>\" [--op evaluate|simplify|expand|factor|determinant]")
		fmt.Println("       strata math --bench")
		return
	}
	if args[0] == "--bench" {
		mr := mathruntime.NewMathRuntime(nil)
		fmt.Println("Running comprehensive mathematical CAS and exact arithmetic benchmark...")
		report := mr.RunMathBenchmark(nil)
		out, _ := json.MarshalIndent(report, "", "  ")
		fmt.Println(string(out))
		return
	}
	expr := args[0]
	op := "evaluate"
	if len(args) >= 3 && args[1] == "--op" {
		op = args[2]
	}

	mr := mathruntime.NewMathRuntime(nil)
	req := mathruntime.MathRequest{
		Expression: expr,
		Operation:  mathruntime.MathOperation(op),
		Mode:       mathruntime.ModeExact,
	}
	res := mr.ProcessRequest(req)
	out, _ := json.MarshalIndent(res, "", "  ")
	fmt.Println(string(out))
}

func runVerify(args []string) {
	if len(args) == 0 {
		fmt.Println("Usage: strata verify \"<claim-or-expression>\"")
		return
	}
	text := args[0]
	lv := logicverifier.NewLogicVerifier()
	results := lv.VerifyText(text, "cli", "session-cli")
	out, _ := json.MarshalIndent(results, "", "  ")
	fmt.Println(string(out))
}

func runTelemetry(args []string) {
	tc := telemetry.NewTelemetryCollector()
	tc.Start()
	time.Sleep(1100 * time.Millisecond)
	tc.Stop()
	current := tc.GetCurrent()
	out, _ := json.MarshalIndent(current, "", "  ")
	fmt.Println(string(out))
}

func runSetup(args []string) {
	fs := flag.NewFlagSet("setup", flag.ExitOnError)
	check := fs.Bool("check", false, "Test PC specifications and report which models fit")
	family := fs.String("family", "openai", "Model family (openai, qwen, coder, swift, unsloth, mimo)")
	model := fs.String("model", "GPT_OSS_20B", "Model name or quantization size")
	contextLen := fs.Int("context", 0, "Context length tokens")
	lowRAM := fs.String("low-ram", "auto", "Low-RAM mode (auto, resident, off)")
	budget := fs.Int("resident-budget-gib", 0, "RAM budget for hot experts in GiB")
	yes := fs.Bool("yes", false, "Run unattended without confirmation prompts")
	noStart := fs.Bool("no-start", false, "Do not launch engine after setup")
	fs.Parse(args)

	flags := installer.SetupFlags{
		Check:             *check,
		Family:            *family,
		Model:             *model,
		Context:           *contextLen,
		LowRAM:            *lowRAM,
		ResidentBudgetGiB: *budget,
		Yes:               *yes,
		NoStart:           *noStart,
	}

	if err := installer.ExecuteSetup(flags); err != nil {
		log.Fatalf("Setup failed: %v", err)
	}
}

func runCheckHallucination(args []string) {
	if len(args) == 0 {
		fmt.Println("Usage: strata check-hallucination \"<text to verify>\" [--policy=MONITOR|STRICT|REGENERATE]")
		return
	}

	text := args[0]
	policy := hallucination.PolicyMonitor
	if len(args) > 1 {
		for _, arg := range args[1:] {
			if arg == "--strict" {
				policy = hallucination.PolicyStrict
			} else if arg == "--warn" {
				policy = hallucination.PolicyWarn
			} else if arg == "--regenerate" {
				policy = hallucination.PolicyRegenerate
			}
		}
	}

	runtime := hallucination.NewRuntime(".", hallucination.DefaultSafetyControllerOptions())
	report, err := runtime.EvaluateText(context.Background(), text, "cli-session", "cli-req", "default", policy)
	if err != nil {
		log.Fatalf("Verification error: %v", err)
	}

	fmt.Println("======================================================================")
	fmt.Println("   STRATA DETERMINISTIC HALLUCINATION & VERIFICATION AUDIT            ")
	fmt.Println("======================================================================")
	fmt.Printf(" Clean (Zero Contradictions): %v\n", report.IsClean)
	fmt.Printf(" Total Claims Extracted:     %d\n", report.Coverage.TotalClaims)
	fmt.Printf("   - Verified:               %d\n", report.Coverage.VerifiedCount)
	fmt.Printf("   - Supported:              %d\n", report.Coverage.SupportedCount)
	fmt.Printf("   - Contradicted:           %d\n", report.Coverage.ContradictedCount)
	fmt.Printf("   - Unsupported:            %d\n", report.Coverage.UnsupportedCount)
	fmt.Printf("   - Unknown / Opinions:     %d\n", report.Coverage.UnknownCount)
	fmt.Printf(" Evidence Coverage Ratio:    %.1f%%\n", report.Coverage.CoverageRatio*100.0)
	fmt.Printf(" Overall Risk Assessment:    %s\n", report.Coverage.OverallRisk)
	fmt.Printf(" Recommended Action:         %s\n", report.Coverage.RecommendedAction)
	fmt.Printf(" Audit Execution Time:       %.2f ms\n", report.ExecutionTimeMs)
	fmt.Println("----------------------------------------------------------------------")

	if len(report.Coverage.Explanations) > 0 {
		fmt.Println(" Extracted Claim Verifications:")
		for idx, exp := range report.Coverage.Explanations {
			fmt.Printf(" [%d] Status: %s | Verifier: %s\n", idx+1, exp.Status, exp.VerifierUsed)
			fmt.Printf("     Text: \"%s\"\n", exp.ClaimText)
			if len(exp.EvidenceDetails) > 0 {
				for _, ev := range exp.EvidenceDetails {
					fmt.Printf("     Evidence: %s\n", ev)
				}
			}
			if len(exp.Contradictions) > 0 {
				for _, c := range exp.Contradictions {
					fmt.Printf("     CONTRADICTION: %s\n", c)
				}
			}
			if exp.FailureReason != "" {
				fmt.Printf("     Reason: %s\n", exp.FailureReason)
			}
		}
	}
	fmt.Println("======================================================================")
}

func runPack(args []string) {
	fs := flag.NewFlagSet("pack", flag.ExitOnError)
	ggufPath := fs.String("gguf", "", "Path to source GGUF file")
	outDir := fs.String("out", "", "Output pack directory or destination GGUF")
	basePack := fs.String("base", "", "Base pack directory (share dense.bin)")
	expertsBin := fs.Bool("experts-bin", false, "Write all experts into one experts.bin file for low-RAM mapping")
	compatBF16 := fs.Bool("compat-bf16", false, "De-quantize small float projections into BF16 format")
	pleModel := fs.String("ple-model", "", "Checkpoint directory containing model.safetensors.index.json to pack PLE FP8")
	embdModel := fs.String("embd-model", "", "Checkpoint directory to pack token_embd BF16")
	mtpFetch := fs.Bool("mtp-fetch", false, "Fetch MTP draft block tensors via HTTP range requests")
	mtpRT := fs.String("mtp-rt", "", "Path to MTP GGUF to convert into runtime files (rt/experts.bin, rt/dense.bin)")
	fs.Parse(args)

	if *pleModel != "" {
		if *outDir == "" {
			log.Fatalf("Pack PLE requires --out <ple-fp8.gguf>")
		}
		fmt.Printf("Packing PLE FP8 table: %s -> %s\n", *pleModel, *outDir)
		if err := pack.PackPLEFP8(pack.PackPLEFP8Options{ModelDir: *pleModel, OutGGUF: *outDir}); err != nil {
			log.Fatalf("Pack PLE failed: %v", err)
		}
		fmt.Printf("PLE FP8 table created successfully: %s\n", *outDir)
		return
	}

	if *embdModel != "" {
		if *outDir == "" {
			log.Fatalf("Pack Embd requires --out <token-embd-bf16.gguf>")
		}
		fmt.Printf("Packing token embedding BF16: %s -> %s\n", *embdModel, *outDir)
		if err := pack.PackEmbdBF16(pack.PackEmbdBF16Options{ModelDir: *embdModel, OutGGUF: *outDir}); err != nil {
			log.Fatalf("Pack Embd failed: %v", err)
		}
		fmt.Printf("Token embedding GGUF created successfully: %s\n", *outDir)
		return
	}

	if *mtpFetch {
		if *outDir == "" {
			log.Fatalf("MTP fetch requires --out <dir>")
		}
		fmt.Printf("Fetching MTP tensors via HTTP range requests to %s...\n", *outDir)
		if err := pack.FetchMTPTensors(context.Background(), pack.MTPFetchOptions{OutDir: *outDir}); err != nil {
			log.Fatalf("MTP fetch failed: %v", err)
		}
		fmt.Printf("MTP tensors fetched successfully in %s\n", *outDir)
		return
	}

	if *mtpRT != "" {
		if *outDir == "" {
			log.Fatalf("MTP RT requires --out <dir>")
		}
		fmt.Printf("Building MTP runtime files: %s -> %s\n", *mtpRT, *outDir)
		if err := pack.BuildMTPRuntimeFiles(*mtpRT, *outDir); err != nil {
			log.Fatalf("MTP RT build failed: %v", err)
		}
		fmt.Printf("MTP runtime files created successfully in %s\n", *outDir)
		return
	}

	if *ggufPath == "" || *outDir == "" {
		fmt.Println("Usage: strata pack --gguf <model.gguf> --out <pack_dir> [--experts-bin] [--compat-bf16]")
		fmt.Println("       strata pack --ple-model <checkpoint_dir> --out <ple.gguf>")
		fmt.Println("       strata pack --embd-model <checkpoint_dir> --out <embd.gguf>")
		fmt.Println("       strata pack --mtp-fetch --out <mtp_dir>")
		fmt.Println("       strata pack --mtp-rt <mtp.gguf> --out <rt_dir>")
		os.Exit(1)
	}

	opts := pack.PackOptions{
		GGUFPath:   *ggufPath,
		OutDir:     *outDir,
		BasePack:   *basePack,
		ExpertsBin: *expertsBin,
		CompatBF16: *compatBF16,
	}

	fmt.Printf("Packaging GGUF model: %s -> %s\n", *ggufPath, *outDir)
	if err := pack.PackModel(opts); err != nil {
		log.Fatalf("Pack failed: %v", err)
	}
	fmt.Printf("Model prepared successfully in %s\n", *outDir)
}

func runMCP(args []string) {
	server := mcp.NewServer()
	ctx := context.Background()
	if err := server.ServeStdio(ctx); err != nil {
		log.Fatalf("MCP server error: %v", err)
	}
}

func runBench(args []string) {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	serverURL := fs.String("server", "http://127.0.0.1:8080", "Strata server base URL")
	prompt := fs.String("prompt", "Explain the concept of quantum entanglement and its implications for computing in 200 words.", "Benchmark prompt text")
	maxTokens := fs.Int("max-tokens", 128, "Maximum generated tokens")
	iterations := fs.Int("n", 3, "Number of benchmark iterations")
	batchSlots := fs.Int("batch", 0, "Run parallel multi-slot batching test with N slots")
	cacheTest := fs.Bool("cache", false, "Run conversation cache smoke and parity test")
	fs.Parse(args)

	if *batchSlots > 0 {
		fmt.Printf("Running parallel multi-slot batching test (%d slots) against %s...\n", *batchSlots, *serverURL)
		report, err := bench.RunBatchTest(context.Background(), bench.BatchTestOptions{
			ServerURL:   *serverURL,
			Concurrency: *batchSlots,
			MaxTokens:   *maxTokens,
		})
		if err != nil {
			log.Fatalf("Batch test failed: %v", err)
		}
		out, _ := json.MarshalIndent(report, "", "  ")
		fmt.Println(string(out))
		return
	}

	if *cacheTest {
		fmt.Printf("Running conversation cache smoke test against %s...\n", *serverURL)
		report, err := bench.RunCacheSmokeTest(context.Background(), *serverURL)
		if err != nil {
			log.Fatalf("Cache smoke test failed: %v", err)
		}
		out, _ := json.MarshalIndent(report, "", "  ")
		fmt.Println(string(out))
		return
	}

	opts := bench.BenchmarkOptions{
		ServerURL:  *serverURL,
		Prompt:     *prompt,
		MaxTokens:  *maxTokens,
		Iterations: *iterations,
	}

	fmt.Println("======================================================================")
	fmt.Println("   STRATA NATIVE INFERENCE PERFORMANCE BENCHMARK (GOLANG)             ")
	fmt.Println("======================================================================")
	fmt.Printf(" Server Endpoint:    %s\n", *serverURL)
	fmt.Printf(" Benchmark Runs:     %d\n", *iterations)
	fmt.Printf(" Max Output Tokens:  %d\n", *maxTokens)
	fmt.Println(" Executing runs...")

	report, err := bench.RunBenchmark(opts)
	if err != nil {
		log.Fatalf("Benchmark error: %v", err)
	}

	fmt.Println("----------------------------------------------------------------------")
	fmt.Printf(" Time To First Token (TTFT): %.2f ms\n", report.TTFTMs)
	fmt.Printf(" Generation Throughput:      %.2f tokens/sec\n", report.GenTokensPerSec)
	fmt.Printf(" Average Total Latency:      %.2f ms\n", report.TotalLatencyMs)
	fmt.Printf(" p50 Latency:                %.2f ms\n", report.P50LatencyMs)
	fmt.Printf(" p95 Latency:                %.2f ms\n", report.P95LatencyMs)
	fmt.Println("======================================================================")
}

func runProfile(args []string) {
	fs := flag.NewFlagSet("profile", flag.ExitOnError)
	base := fs.String("base", "data/expert-profile.bin", "Base ranking profile to keep first")
	noBase := fs.Bool("no-base", false, "Rank by routing traces only")
	reorder := fs.Bool("reorder", false, "Rank trace pairs before base")
	out := fs.String("out", "data/expert-profile.bin", "Output path for STRP profile")
	nExpert := fs.Int("n-expert", pack.DefaultNExpert, "Experts per layer (default 512)")
	nLayers := fs.Int("n-layers", pack.DefaultNLayers, "Number of layers (default 48)")
	fs.Parse(args)

	traces := fs.Args()
	freq := make(map[pack.ExpertPair]int)
	for _, t := range traces {
		tf, err := pack.ReadTrace(t, *nLayers, *nExpert)
		if err != nil {
			log.Printf("Warning: failed to read trace %s: %v", t, err)
			continue
		}
		for p, c := range tf {
			freq[p] += c
		}
	}

	var basePairs []pack.ExpertPair
	if !*noBase && *base != "" {
		bp, err := pack.ReadProfile(*base, *nLayers, *nExpert)
		if err != nil {
			log.Printf("Warning: could not read base profile %s: %v", *base, err)
		} else {
			basePairs = bp
		}
	}

	ranked, counts := pack.RankProfile(basePairs, freq, *nLayers, *nExpert, *noBase, *reorder)
	if err := pack.WriteProfile(*out, ranked, *nLayers, *nExpert); err != nil {
		log.Fatalf("Failed to write profile: %v", err)
	}
	fmt.Printf("Wrote %s: %d ranked pairs (%d from base, %d from traces, %d filled in)\n",
		*out, len(ranked), counts["base"], counts["trace"], counts["fill"])
}

func runCalibrate(args []string) {
	fs := flag.NewFlagSet("calibrate", flag.ExitOnError)
	serverURL := fs.String("server", "http://127.0.0.1:8080", "Strata server endpoint")
	apiKey := fs.String("api-key", "", "API key")
	fs.Parse(args)

	fmt.Printf("Tuning hardware parameters against %s...\n", *serverURL)
	ctx := context.Background()
	res, err := bench.RunCalibration(ctx, bench.CalibrationOptions{
		BaseURL: *serverURL,
		APIKey:  *apiKey,
	})
	if err != nil {
		log.Fatalf("Calibration failed: %v", err)
	}
	out, _ := json.MarshalIndent(res, "", "  ")
	fmt.Println(string(out))
}

func runNeedle(args []string) {
	fs := flag.NewFlagSet("needle", flag.ExitOnError)
	serverURL := fs.String("server", "http://127.0.0.1:8080", "Strata server endpoint")
	apiKey := fs.String("api-key", "", "API key")
	lengthsStr := fs.String("lengths", "4096,8192", "Comma-separated target context lengths")
	fs.Parse(args)

	lengths := []int{4096, 8192}
	if *lengthsStr != "" {
		lengths = nil
		for _, part := range strings.Split(*lengthsStr, ",") {
			var l int
			fmt.Sscanf(strings.TrimSpace(part), "%d", &l)
			if l > 0 {
				lengths = append(lengths, l)
			}
		}
	}

	fmt.Printf("Running needle-in-a-haystack recall test on %s...\n", *serverURL)
	ctx := context.Background()
	res, err := bench.RunNeedleEvaluation(ctx, bench.NeedleTestOptions{
		BaseURL: *serverURL,
		APIKey:  *apiKey,
		Lengths: lengths,
	})
	if err != nil {
		log.Fatalf("Needle test failed: %v", err)
	}

	for _, r := range res {
		status := "FAIL"
		if r.Found {
			status = "PASS"
		}
		fmt.Printf("[%s] Target %d tokens (actual: %d) at depth %d%%: needle '%s' (%v)\n",
			status, r.TargetLength, r.ActualTokens, r.DepthPercent, r.NeedleWord, r.Duration)
	}
}

func runTokenize(args []string) {
	fs := flag.NewFlagSet("tokenize", flag.ExitOnError)
	tokDir := fs.String("dir", "", "Path to tokenizer directory containing vocab.json")
	text := fs.String("text", "", "Text to encode")
	decodeIDs := fs.String("decode", "", "Comma-separated token IDs to decode")
	fs.Parse(args)

	dir := *tokDir
	if dir == "" {
		dataDir := installer.GetDataDir()
		candidates := []string{
			filepath.Join(dataDir, "packs", "coder-iq1_m", "tokenizer"),
			filepath.Join("data", "packs", "coder-iq1_m", "tokenizer"),
			"tokenizer",
		}
		for _, c := range candidates {
			if _, err := os.Stat(filepath.Join(c, "vocab.json")); err == nil {
				dir = c
				break
			}
		}
	}

	if dir == "" {
		log.Fatalf("Tokenizer directory not found. Please specify with --dir <path>")
	}

	tok, err := tokenizer.Load(dir)
	if err != nil {
		log.Fatalf("Failed to load tokenizer from %s: %v", dir, err)
	}

	if *decodeIDs != "" {
		var ids []int
		for _, part := range strings.Split(*decodeIDs, ",") {
			var id int
			fmt.Sscanf(strings.TrimSpace(part), "%d", &id)
			ids = append(ids, id)
		}
		decoded := tok.Decode(ids)
		fmt.Printf("Decoded text: %q\n", decoded)
		return
	}

	targetText := *text
	if targetText == "" && fs.NArg() > 0 {
		targetText = fs.Arg(0)
	}

	if targetText == "" {
		fmt.Println("Usage: strata tokenize [--dir <dir>] --text \"<text>\"")
		fmt.Println("       strata tokenize [--dir <dir>] --decode <id1,id2,...>")
		return
	}

	encoded := tok.Encode(targetText, true)
	fmt.Printf("Token IDs (%d): %v\n", len(encoded), encoded)
	decoded := tok.Decode(encoded)
	fmt.Printf("Round-trip text: %q (exact match: %t)\n", decoded, decoded == targetText)
}




