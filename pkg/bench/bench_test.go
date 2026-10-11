package bench

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestRunBenchmark_MockServer(t *testing.T) {
	// Mock OpenAI compatible streaming endpoint
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected flusher")
		}

		time.Sleep(10 * time.Millisecond) // simulate TTFT delay
		// First chunk
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"))
		flusher.Flush()

		// Second chunk
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\" world!\"}}]}\n\n"))
		flusher.Flush()

		w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}))
	defer ts.Close()

	opts := BenchmarkOptions{
		ServerURL:   ts.URL,
		Prompt:      "Say hello",
		MaxTokens:   16,
		Concurrence: 1,
		Iterations:  2,
	}

	report, err := RunBenchmark(opts)
	if err != nil {
		t.Fatalf("RunBenchmark failed: %v", err)
	}

	if report.Iterations != 2 {
		t.Errorf("expected 2 iterations, got %d", report.Iterations)
	}
	if report.TTFTMs <= 0 {
		t.Errorf("expected positive TTFT, got %f", report.TTFTMs)
	}
	if report.GenTokensPerSec <= 0 {
		t.Errorf("expected positive tokens/sec, got %f", report.GenTokensPerSec)
	}

	out, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("failed to marshal report: %v", err)
	}
	if len(out) == 0 {
		t.Errorf("empty report JSON")
	}
}
