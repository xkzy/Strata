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
