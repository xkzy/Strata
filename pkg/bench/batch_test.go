package bench

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestRunBatchTest(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{"content": "Batch response content."}},
			},
			"usage": map[string]int{"completion_tokens": 10},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer ts.Close()

	opts := BatchTestOptions{
		ServerURL:   ts.URL,
		Concurrency: 2,
		MaxTokens:   16,
		Timeout:     5 * time.Second,
	}

	report, err := RunBatchTest(context.Background(), opts)
	if err != nil {
		t.Fatalf("RunBatchTest failed: %v", err)
	}
	if !report.AllSucceeded {
		t.Errorf("expected all slots to succeed")
	}
	if report.TotalTokens != 20 {
		t.Errorf("expected 20 total tokens, got %d", report.TotalTokens)
	}
}

func TestRunCacheSmokeTest(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.Write([]byte(`{"status":"ok"}`))
			return
		}
		resp := map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{"content": "The code word was PHOENIX-99."}},
			},
			"usage": map[string]int{"completion_tokens": 8},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer ts.Close()

	report, err := RunCacheSmokeTest(context.Background(), ts.URL)
	if err != nil {
		t.Fatalf("RunCacheSmokeTest failed: %v", err)
	}
	if !report.SmokePassed {
		t.Errorf("expected smoke test to pass")
	}
}
