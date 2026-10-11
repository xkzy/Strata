package bench

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"
)

// RunBenchmark executes performance benchmarking against a Strata API server
func RunBenchmark(opts BenchmarkOptions) (*BenchmarkReport, error) {
	if opts.ServerURL == "" {
		opts.ServerURL = "http://127.0.0.1:8080"
	}
	if opts.Prompt == "" {
		opts.Prompt = "Write a short poem about stars and space exploration."
	}
	if opts.Iterations <= 0 {
		opts.Iterations = 3
	}
	if opts.MaxTokens <= 0 {
		opts.MaxTokens = 128
	}

	endpoint := opts.ServerURL
	if !strings.HasSuffix(endpoint, "/v1/chat/completions") && !strings.Contains(endpoint, "/v1/") {
		endpoint = strings.TrimRight(endpoint, "/") + "/v1/chat/completions"
	}

	var ttftList []float64
	var totalLatencyList []float64
	totalGenTokens := 0
	totalPromptTokens := len(strings.Fields(opts.Prompt))

	client := &http.Client{Timeout: 120 * time.Second}

	for i := 0; i < opts.Iterations; i++ {
		reqBody := map[string]any{
			"model": opts.ModelName,
			"messages": []map[string]string{
				{"role": "user", "content": opts.Prompt},
			},
			"stream":     true,
			"max_tokens": opts.MaxTokens,
		}
		raw, _ := json.Marshal(reqBody)

		start := time.Now()
		req, err := http.NewRequest("POST", endpoint, bytes.NewReader(raw))
		if err != nil {
			return nil, fmt.Errorf("create req: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("do request: %w", err)
		}

		scanner := bufio.NewScanner(resp.Body)
		firstTokenReceived := false
		tokensInIter := 0
		var ttft time.Duration

		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "data: ") {
				payload := strings.TrimPrefix(line, "data: ")
				if strings.TrimSpace(payload) == "[DONE]" {
					break
				}
				if !firstTokenReceived {
					ttft = time.Since(start)
					firstTokenReceived = true
				}
				tokensInIter++
			}
		}
		resp.Body.Close()

		totalDuration := time.Since(start)
		if ttft == 0 {
			ttft = totalDuration
		}

		ttftList = append(ttftList, float64(ttft.Milliseconds()))
		totalLatencyList = append(totalLatencyList, float64(totalDuration.Milliseconds()))
		totalGenTokens += tokensInIter
	}

	sort.Float64s(totalLatencyList)
	avgTTFT := average(ttftList)
	avgTotal := average(totalLatencyList)

	genTokensPerSec := 0.0
	if avgTotal > 0 {
		genTokensPerSec = (float64(totalGenTokens) / float64(opts.Iterations)) / (avgTotal / 1000.0)
	}

	p50 := totalLatencyList[len(totalLatencyList)/2]
	p95 := totalLatencyList[int(float64(len(totalLatencyList))*0.95)]

	return &BenchmarkReport{
		Iterations:      opts.Iterations,
		PromptTokens:    totalPromptTokens,
		GenTokens:       totalGenTokens / opts.Iterations,
		TTFTMs:          avgTTFT,
		GenTokensPerSec: genTokensPerSec,
		TotalLatencyMs:  avgTotal,
		P50LatencyMs:    p50,
		P95LatencyMs:    p95,
	}, nil
}

func average(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}
