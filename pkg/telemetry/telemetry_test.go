// pkg/telemetry/telemetry_test.go - Unit Tests for Telemetry Collector
package telemetry

import (
	"testing"
	"time"
)

func TestTelemetryCollector_SampleAndHistory(t *testing.T) {
	tc := NewTelemetryCollector()
	tc.Start()
	defer tc.Stop()

	time.Sleep(1100 * time.Millisecond)

	current := tc.GetCurrent()
	if current.Timestamp == 0 {
		t.Fatalf("expected non-zero timestamp in telemetry reading")
	}

	history := tc.GetHistory()
	if len(history) == 0 {
		t.Fatalf("expected at least 1 reading in history")
	}
}

// historySlice returns one sparkline history from a Snapshot, failing the test if the key is absent or not a
// []float64 (the web app reads history.prefill_tok_s_mean the same way it reads history.tok_s).
func historySlice(t *testing.T, tc *TelemetryCollector, key string) []float64 {
	t.Helper()
	hist, ok := tc.Snapshot()["history"].(map[string]interface{})
	if !ok {
		t.Fatalf("snapshot has no history map")
	}
	v, ok := hist[key].([]float64)
	if !ok {
		t.Fatalf("history[%q] is %T, want []float64", key, hist[key])
	}
	return v
}

func TestPrefillHistoryFollowsItsSource(t *testing.T) {
	tc := NewTelemetryCollector()
	tc.SetPrefillSource(func() float64 { return 42.5 })
	tc.sampleOnce()
	tc.sampleOnce()

	h := historySlice(t, tc, "prefill_tok_s_mean")
	if len(h) != 2 {
		t.Fatalf("one history entry per sample: %v", h)
	}
	if h[0] != 42.5 || h[1] != 42.5 {
		t.Errorf("history must follow the source: %v", h)
	}

	// An idle request (source 0) keeps the sparkline going at zero, as the decode history does.
	tc.SetPrefillSource(func() float64 { return 0 })
	tc.sampleOnce()
	if h := historySlice(t, tc, "prefill_tok_s_mean"); len(h) != 3 || h[2] != 0 {
		t.Errorf("idle sample: %v", h)
	}
}

func TestPrefillHistoryWithoutASourceIsZero(t *testing.T) {
	tc := NewTelemetryCollector()
	tc.sampleOnce()
	if h := historySlice(t, tc, "prefill_tok_s_mean"); len(h) != 1 || h[0] != 0 {
		t.Errorf("no source, zero expected: %v", h)
	}
}

func TestBothSpeedHistoriesAreBounded(t *testing.T) {
	tc := NewTelemetryCollector()
	tc.SetTokSource(func() float64 { return 7 })
	tc.SetPrefillSource(func() float64 { return 9 })
	for i := 0; i < HistoryCapacity+5; i++ {
		tc.mu.Lock()
		tc.pushHistory(&tc.historyTokS, 7)
		tc.pushHistory(&tc.historyPrefill, 9)
		tc.mu.Unlock()
	}
	if h := historySlice(t, tc, "tok_s"); len(h) != HistoryCapacity {
		t.Errorf("tok_s: %d entries", len(h))
	}
	if h := historySlice(t, tc, "prefill_tok_s_mean"); len(h) != HistoryCapacity {
		t.Errorf("prefill: %d entries", len(h))
	}
}

func TestRAMSampleSaysWhetherItIsSystemWide(t *testing.T) {
	tc := NewTelemetryCollector()
	tc.sampleOnce()
	now, ok := tc.Snapshot()["now"].(map[string]interface{})
	if !ok {
		t.Fatal("snapshot has no now map")
	}
	system, ok := now["ram_system"].(bool)
	if !ok {
		t.Fatalf("ram_system must be present and boolean: %#v", now["ram_system"])
	}
	if system {
		if _, ok := now["ram_total"].(uint64); !ok {
			t.Fatalf("system RAM without a total: %#v", now["ram_total"])
		}
	}
}
