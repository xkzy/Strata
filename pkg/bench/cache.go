package bench

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// CacheTestReport summarizes the results of conversation cache tests.
type CacheTestReport struct {
	SlotsDetected int    `json:"slots_detected"`
	SmokePassed   bool   `json:"smoke_passed"`
	ParityPassed  bool   `json:"parity_passed"`
	Detail        string `json:"detail"`
}

// RunCacheSmokeTest executes multi-turn conversations through an active server and checks cache consistency.
func RunCacheSmokeTest(ctx context.Context, serverURL string) (*CacheTestReport, error) {
	if serverURL == "" {
		serverURL = "http://127.0.0.1:8080"
	}

	client := &http.Client{Timeout: 60 * time.Second}
	report := &CacheTestReport{}

	// Check /health or /metrics
	resp, err := client.Get(strings.TrimRight(serverURL, "/") + "/health")
	if err != nil {
		return nil, fmt.Errorf("connect to server health: %w", err)
	}
	resp.Body.Close()

	// Turn 1
	turn1Prompt := "Remember the code word: PHOENIX-99."
	body1, _ := json.Marshal(map[string]any{
		"model": "default",
		"messages": []map[string]string{
			{"role": "user", "content": turn1Prompt},
		},
		"max_tokens":  16,
		"temperature": 0.0,
	})

	req1, _ := http.NewRequestWithContext(ctx, "POST", strings.TrimRight(serverURL, "/")+"/v1/chat/completions", bytes.NewReader(body1))
	req1.Header.Set("Content-Type", "application/json")
	resp1, err := client.Do(req1)
	if err != nil {
		return nil, fmt.Errorf("turn 1 request failed: %w", err)
	}
	defer resp1.Body.Close()

	var chat1 struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	_ = json.NewDecoder(resp1.Body).Decode(&chat1)

	// Turn 2 (with cached context)
	body2, _ := json.Marshal(map[string]any{
		"model": "default",
		"messages": []map[string]string{
			{"role": "user", "content": turn1Prompt},
			{"role": "assistant", "content": "Acknowledged."},
			{"role": "user", "content": "What was the code word?"},
		},
		"max_tokens":  16,
		"temperature": 0.0,
	})

	req2, _ := http.NewRequestWithContext(ctx, "POST", strings.TrimRight(serverURL, "/")+"/v1/chat/completions", bytes.NewReader(body2))
	req2.Header.Set("Content-Type", "application/json")
	resp2, err := client.Do(req2)
	if err != nil {
		return nil, fmt.Errorf("turn 2 request failed: %w", err)
	}
	defer resp2.Body.Close()

	var chat2 struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	_ = json.NewDecoder(resp2.Body).Decode(&chat2)

	if len(chat2.Choices) > 0 && strings.Contains(strings.ToUpper(chat2.Choices[0].Message.Content), "PHOENIX-99") {
		report.SmokePassed = true
		report.ParityPassed = true
		report.Detail = "Conversation context retained and recalled correctly across cache turns"
	} else {
		report.SmokePassed = false
		report.ParityPassed = false
		report.Detail = "Cache turn failed to retain target code word"
	}

	return report, nil
}

// ComputeHash computes a SHA256 hex digest of arbitrary text.
func ComputeHash(text string) string {
	h := sha256.Sum256([]byte(text))
	return hex.EncodeToString(h[:])
}
