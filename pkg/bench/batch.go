package bench

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// BatchTestOptions defines parameters for parallel multi-slot decoding tests.
type BatchTestOptions struct {
	ServerURL   string
	Concurrency int
	Prompts     []string
	MaxTokens   int
	Timeout     time.Duration
}

// BatchSlotResult stores the response and performance of a single batch slot.
type BatchSlotResult struct {
	SlotID    int           `json:"slot_id"`
	Prompt    string        `json:"prompt"`
	Response  string        `json:"response"`
	Tokens    int           `json:"tokens"`
	Duration  time.Duration `json:"duration_ms"`
	TokensPerSec float64    `json:"tokens_per_sec"`
	Success   bool          `json:"success"`
	Error     string        `json:"error,omitempty"`
}

// BatchTestReport summarizes the aggregate throughput across parallel slots.
type BatchTestReport struct {
	Concurrency       int               `json:"concurrency"`
	Slots             []BatchSlotResult `json:"slots"`
	TotalTokens       int               `json:"total_tokens"`
	WallDuration      time.Duration     `json:"wall_duration_ms"`
	AggregateTokensPerSec float64       `json:"aggregate_tokens_per_sec"`
	AllSucceeded      bool              `json:"all_succeeded"`
}

// DefaultBatchPrompts provides varied questions for parallel evaluation.
var DefaultBatchPrompts = []string{
	"Explain the internal mechanics of a B-tree index in relational databases.",
	"Write a Python function to merge overlapping intervals with tests.",
	"Compare TCP and QUIC in terms of handshake latency and packet multiplexing.",
	"Summarize the history of computing pi from Archimedes to modern algorithms.",
	"Explain step by step how multi-head self-attention works.",
	"Describe the lifecycle and fusion mechanisms of a star like the Sun.",
	"Provide a step-by-step recipe for classic apple pie.",
	"Analyze the main historical economic causes of the French Revolution.",
}

// RunBatchTest executes concurrent generation requests across multiple slots.
func RunBatchTest(ctx context.Context, opts BatchTestOptions) (*BatchTestReport, error) {
	if opts.ServerURL == "" {
		opts.ServerURL = "http://127.0.0.1:8080"
	}
	if opts.Concurrency <= 0 {
		opts.Concurrency = 4
	}
	if len(opts.Prompts) == 0 {
		opts.Prompts = DefaultBatchPrompts
	}
	if opts.MaxTokens <= 0 {
		opts.MaxTokens = 64
	}
	if opts.Timeout <= 0 {
		opts.Timeout = 120 * time.Second
	}

	client := &http.Client{Timeout: opts.Timeout}
	report := &BatchTestReport{
		Concurrency:  opts.Concurrency,
		Slots:        make([]BatchSlotResult, opts.Concurrency),
		AllSucceeded: true,
	}

	var wg sync.WaitGroup
	startWall := time.Now()

	for slot := 0; slot < opts.Concurrency; slot++ {
		wg.Add(1)
		go func(s int) {
			defer wg.Done()
			prompt := opts.Prompts[s%len(opts.Prompts)]
			res := BatchSlotResult{
				SlotID: s,
				Prompt: prompt,
			}

			reqBody, _ := json.Marshal(map[string]any{
				"model": "default",
				"messages": []map[string]string{
					{"role": "user", "content": prompt},
				},
				"temperature": 0.0,
				"max_tokens":  opts.MaxTokens,
				"chat_template_kwargs": map[string]any{
					"enable_thinking": false,
				},
			})

			req, err := http.NewRequestWithContext(ctx, "POST", strings.TrimRight(opts.ServerURL, "/")+"/v1/chat/completions", bytes.NewReader(reqBody))
			if err != nil {
				res.Success = false
				res.Error = err.Error()
				report.Slots[s] = res
				return
			}
			req.Header.Set("Content-Type", "application/json")

			startSlot := time.Now()
			resp, err := client.Do(req)
			elapsed := time.Since(startSlot)
			res.Duration = elapsed

			if err != nil {
				res.Success = false
				res.Error = err.Error()
				report.Slots[s] = res
				return
			}
			defer resp.Body.Close()

			body, _ := io.ReadAll(resp.Body)
			var chatResp struct {
				Choices []struct {
					Message struct {
						Content string `json:"content"`
					} `json:"message"`
				} `json:"choices"`
				Usage struct {
					CompletionTokens int `json:"completion_tokens"`
				} `json:"usage"`
			}
			_ = json.Unmarshal(body, &chatResp)

			if len(chatResp.Choices) > 0 {
				res.Response = chatResp.Choices[0].Message.Content
				res.Tokens = chatResp.Usage.CompletionTokens
				if elapsed.Seconds() > 0 {
					res.TokensPerSec = float64(res.Tokens) / elapsed.Seconds()
				}
				res.Success = true
			} else {
				res.Success = false
				res.Error = fmt.Sprintf("invalid response body: %s", string(body))
			}

			report.Slots[s] = res
		}(slot)
	}

	wg.Wait()
	report.WallDuration = time.Since(startWall)

	for _, slot := range report.Slots {
		report.TotalTokens += slot.Tokens
		if !slot.Success {
			report.AllSucceeded = false
		}
	}

	if report.WallDuration.Seconds() > 0 {
		report.AggregateTokensPerSec = float64(report.TotalTokens) / report.WallDuration.Seconds()
	}

	return report, nil
}
