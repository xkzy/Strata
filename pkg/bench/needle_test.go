package bench

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestNeedleEvaluation(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{"content": "The secret password code is amber."}},
			},
			"usage": map[string]int{"prompt_tokens": 128},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer ts.Close()

	opts := NeedleTestOptions{
		BaseURL:     ts.URL,
		Lengths:     []int{256},
		Depths:      []int{50},
		NeedleWords: []string{"amber"},
		Timeout:     5 * time.Second,
	}

	res, err := RunNeedleEvaluation(context.Background(), opts)
	if err != nil {
		t.Fatalf("needle evaluation failed: %v", err)
	}
	if len(res) != 1 {
		t.Fatalf("expected 1 result, got %d", len(res))
	}
	if !res[0].Found {
		t.Errorf("expected needle 'amber' to be found in response %q", res[0].Response)
	}
}

func TestCalibration(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"usage": map[string]int{"completion_tokens": 50},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer ts.Close()

	opts := CalibrationOptions{
		BaseURL:   ts.URL,
		PCIeFracs: []float64{0.2, 0.5},
		SpecMinPs: []float64{0.5},
	}

	res, err := RunCalibration(context.Background(), opts)
	if err != nil {
		t.Fatalf("calibration failed: %v", err)
	}
	if res.BestPCIeFrac <= 0 {
		t.Errorf("expected positive best pcie frac, got %f", res.BestPCIeFrac)
	}
}

func TestGenerateHaystack(t *testing.T) {
	haystack := GenerateHaystack(500, "secretword", 50)
	if !strings.Contains(haystack, "secretword") {
		t.Errorf("haystack did not contain needle word")
	}
}
