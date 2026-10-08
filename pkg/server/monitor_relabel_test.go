package server

import "testing"

// The runtime stops reading a reply that hit max_tokens, so the engine reports a cancel; the request's final finish
// reason (what the API returns) replaces it in the Monitor.
func TestRelabelSetsTheFinalFinishReason(t *testing.T) {
	var m requestMonitor
	m.init()
	r := m.begin(10, 3)
	r.token()
	r.finish("cancel", 3, 0)
	r.relabel("length")
	reqs, _, totals := m.view(false)
	if len(reqs) != 1 || reqs[0]["finish"] != "length" {
		t.Fatalf("finish after relabel: %v", reqs)
	}
	if totals["requests"] != 1 {
		t.Errorf("relabel must not count the request again: %v", totals)
	}
	r.relabel("tool_calls") // shown like the engine's stop, as in finish
	if reqs, _, _ = m.view(false); reqs[0]["finish"] != "stop" {
		t.Errorf("tool_calls is a stop: %v", reqs[0]["finish"])
	}
	// a request that never finished has nothing to relabel
	r2 := m.begin(1, 1)
	r2.relabel("length")
}
