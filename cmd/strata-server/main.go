package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"strata/pkg/server"
)

func main() {
	port := flag.Int("port", 8080, "Port to listen on")
	host := flag.String("host", "127.0.0.1", "Address to listen on (anything but loopback requires --api-key)")
	rtMode := flag.String("rt", "auto", "C++ transparent runtime (verification, virtual context): auto, on or off")
	rtBinary := flag.String("rt-binary", "", "Path to strata_rt_server (default: looked for next to the engine)")
	window := flag.Int("window", 32768, "Physical context window the runtime fills per request (tokens)")
	modelName := flag.String("model", "Qwen3.8-Flash-Next", "Model name or identifier")
	maxContext := flag.Int("max-context", 32768, "Physical context limit in tokens")
	virtualLimit := flag.Int("virtual-limit", 2000000, "Virtual context limit in tokens")
	apiKey := flag.String("api-key", "", "Optional API Key for authentication")
	modelPath := flag.String("model-path", "", "Path to the model pack (legacy; prefer --config)")
	binaryPath := flag.String("binary-path", "", "Path to the C++ strata engine binary")
	configFile := flag.String("config", "", "Path to strata-<model>.json (engine command line)")
	apiMonitor := flag.Bool("api-monitor", false, "Keep the last 100 requests' prompts and answers in memory for /api-monitor (also \"api_monitor\": true in the run config)")
	flag.Parse()

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
		APIMonitor:    *apiMonitor,
	}

	if err := server.CheckBind(*host, *apiKey); err != nil {
		fmt.Fprintln(os.Stderr, "strata-server:", err)
		os.Exit(2)
	}
	strataServer := server.NewStrataServer(cfg)
	if err := strataServer.StartEngine(context.Background()); err != nil {
		fmt.Fprintf(os.Stderr, "strata: the model is not available: %v\n", err)
	} else {
		fmt.Fprintln(os.Stderr, "strata: loading the model in the background (requests get 503 until it is ready; see /health)")
	}
	handler := strataServer.Router()

	httpServer := &http.Server{
		Addr:         net.JoinHostPort(*host, fmt.Sprint(*port)),
		Handler:      handler,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 0, // replies stream for as long as the model thinks
	}

	fmt.Println("======================================================================")
	fmt.Printf("   STRATA HIGH-PERFORMANCE AI SERVER & MATHEMATICAL ENGINE (GOLANG)   \n")
	fmt.Println("======================================================================")
	fmt.Printf(" Model Name:        %s\n", *modelName)
	fmt.Printf(" Listening on:      http://%s\n", net.JoinHostPort(*host, fmt.Sprint(*port)))
	fmt.Printf(" Physical Context:  %d tokens\n", *maxContext)
	fmt.Printf(" Virtual Context:   %d tokens\n", *virtualLimit)
	fmt.Printf(" Math Runtime:      Mathics3 CAS + FastNumeric Exact Rational Active\n")
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
