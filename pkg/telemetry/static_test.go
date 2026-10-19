package telemetry

import "testing"

// The static probe is cached for the process, but each collector owns its own map: a caller that edits one
// collector's view must not change what the next collector reports.
func TestStaticFactsAreSharedAndCopied(t *testing.T) {
	a, b := NewTelemetryCollector(), NewTelemetryCollector()
	if a.static["gpu_name"] != b.static["gpu_name"] || a.static["threads"] != b.static["threads"] {
		t.Errorf("collectors disagree on static hardware: %v vs %v", a.static, b.static)
	}
	a.static["gpu_name"] = "mutated"
	a.static["cores"] = 0
	c := NewTelemetryCollector()
	if c.static["gpu_name"] == "mutated" || c.static["cores"] == 0 {
		t.Errorf("the process-wide facts leaked into a collector: %v", c.static)
	}
	if _, ok := c.static["gpu_count"]; !ok {
		t.Errorf("static facts missing after the copy: %v", c.static)
	}
}

// Building a collector used to fork nvidia-smi (or lspci) every time. One probe per process is what the server
// tests and a multi-model server pay.
func BenchmarkNewCollector(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = NewTelemetryCollector()
	}
}
