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
	"syscall"
	"time"

	"strata/pkg/hallucination"
	"strata/pkg/installer"
	"strata/pkg/logicverifier"
	"strata/pkg/mathruntime"
	"strata/pkg/server"
	"strata/pkg/telemetry"
)

const version = "1.0.0"

func main() {
	if len(os.Args) < 2 {
		printUsage()
		return
	}

	command := os.Args[1]

	switch command {
	case "serve", "server":
		runServer(os.Args[2:])
	case "setup":
		runSetup(os.Args[2:])
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
	window := fs.Int("window", 32768, "Physical context window the runtime fills per request (tokens)")
	modelName := fs.String("model", "Qwen3.8-Flash-Next", "Model identifier")
	maxContext := fs.Int("max-context", 32768, "Physical context limit")
	virtualLimit := fs.Int("virtual-limit", 2000000, "Virtual context limit")
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
	fmt.Printf(" Model Name:        %s\n", *modelName)
	fmt.Printf(" Web UI & API:      http://127.0.0.1:%d\n", *port)
	fmt.Printf(" Physical Context:  %d tokens\n", *maxContext)
	fmt.Printf(" Virtual Context:   %d tokens\n", *virtualLimit)
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
	fmt.Println("======================================================================")
	fmt.Println("   STRATA NATIVE HARDWARE DETECTION & SETUP (GOLANG)                  ")
	fmt.Println("======================================================================")

	specs := installer.DetectHardware()
	fmt.Printf(" CPU Cores:         %d\n", specs.CPUCores)
	fmt.Printf(" Total System RAM:  %.2f GB\n", specs.RAMTotalGB)
	fmt.Printf(" GPU Device:        %s (%s)\n", specs.GPUName, specs.GPUVendor)
	if specs.VRAMGB > 0 {
		fmt.Printf(" Dedicated VRAM:    %.2f GB\n", specs.VRAMGB)
	}

	model := installer.RecommendModel(specs)
	fmt.Printf("\n [Optimal Model Recommendation]\n")
	fmt.Printf(" Model:             %s\n", model.Name)
	fmt.Printf(" Description:       %s\n", model.Description)
	fmt.Printf(" Required RAM:      >= %.0f GB (System has %.1f GB)\n", model.MinRAMGB, specs.RAMTotalGB)
	fmt.Println("======================================================================")
	fmt.Println("Setup verified. To start inference server, run: strata serve")
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
