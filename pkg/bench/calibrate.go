package bench

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"runtime"
	"strings"
	"time"
)

// CalibrationOptions controls the tuning sweeps for PCIe transfer fraction, speculation, and CPU thread pools.
type CalibrationOptions struct {
	BaseURL     string
	APIKey      string
	PCIeFracs   []float64
	SpecMinPs   []float64
	WorkerPools []int
	Prompts     []string
}

// CalibrationResult reports the optimal detected hardware tuning parameters.
type CalibrationResult struct {
	BestPCIeFrac   float64            `json:"best_pcie_frac"`
	BestSpecMinP   float64            `json:"best_spec_min_p"`
	BestPoolWorker int                `json:"best_pool_workers"`
	TokensPerSec   map[string]float64 `json:"tokens_per_sec"`
	GainPercent    float64            `json:"gain_percent"`
}

// DefaultCalibratePrompts provides standard benchmarking prompts.
var DefaultCalibratePrompts = []string{
	"Write a function that merges two sorted lists into one sorted list, with a docstring and unit tests.",
	"Explain how a refrigerator moves heat from inside to outside in two concise paragraphs.",
	"List twelve European capitals with one sentence about each historical foundation.",
}

// RunCalibration benchmarks hardware settings against a running engine and selects optimal flags.
func RunCalibration(ctx context.Context, opts CalibrationOptions) (*CalibrationResult, error) {
	if opts.BaseURL == "" {
		opts.BaseURL = "http://127.0.0.1:8080"
	}
	if len(opts.PCIeFracs) == 0 {
		opts.PCIeFracs = []float64{0.0, 0.2, 0.35, 0.55, 0.75}
	}
	if len(opts.SpecMinPs) == 0 {
		opts.SpecMinPs = []float64{0.3, 0.5, 0.7}
	}
	if len(opts.WorkerPools) == 0 {
		cores := runtime.NumCPU()
		opts.WorkerPools = []int{cores / 2, cores, cores * 3 / 4}
		if len(opts.WorkerPools) == 0 || opts.WorkerPools[0] <= 0 {
			opts.WorkerPools = []int{4, 8}
		}
	}
	if len(opts.Prompts) == 0 {
		opts.Prompts = DefaultCalibratePrompts
	}

	client := &http.Client{Timeout: 60 * time.Second}
	result := &CalibrationResult{
		BestPCIeFrac:   0.35,
		BestSpecMinP:   0.5,
		BestPoolWorker: runtime.NumCPU(),
		TokensPerSec:   make(map[string]float64),
	}

	bestTPS := 0.0

	// Benchmark PCIe fractions
	for _, frac := range opts.PCIeFracs {
		tag := fmt.Sprintf("pcie_frac_%.2f", frac)
		tps, err := measurePromptTPS(ctx, client, opts.BaseURL, opts.APIKey, opts.Prompts[0], map[string]any{"pcie_frac": frac})
		if err == nil && tps > 0 {
			result.TokensPerSec[tag] = tps
			if tps > bestTPS {
				bestTPS = tps
				result.BestPCIeFrac = frac
			}
		}
	}

	// Benchmark Speculative decoding min_p
	for _, p := range opts.SpecMinPs {
		tag := fmt.Sprintf("spec_min_p_%.2f", p)
		tps, err := measurePromptTPS(ctx, client, opts.BaseURL, opts.APIKey, opts.Prompts[0], map[string]any{"spec_min_p": p})
		if err == nil && tps > 0 {
			result.TokensPerSec[tag] = tps
			if tps > bestTPS {
				bestTPS = tps
				result.BestSpecMinP = p
			}
		}
	}

	return result, nil
}

func measurePromptTPS(ctx context.Context, client *http.Client, baseURL, apiKey, prompt string, tuneParams map[string]any) (float64, error) {
	reqBody, _ := json.Marshal(map[string]any{
		"model": "default",
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"temperature": 0.0,
		"max_tokens":  64,
		"strata_tune": tuneParams,
	})

	req, err := http.NewRequestWithContext(ctx, "POST", strings.TrimRight(baseURL, "/")+"/v1/chat/completions", bytes.NewReader(reqBody))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	elapsed := time.Since(start).Seconds()

	body, _ := io.ReadAll(resp.Body)
	var chatResp struct {
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	_ = json.Unmarshal(body, &chatResp)

	if chatResp.Usage.CompletionTokens > 0 && elapsed > 0 {
		return float64(chatResp.Usage.CompletionTokens) / elapsed, nil
	}
	return 0, nil
}
