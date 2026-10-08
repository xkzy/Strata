package server

import "testing"

// While the engine starts the Monitor must say "loading", and after a failed start "error"; "unloaded" is only
// for an engine that is stopped. A running request keeps its own state.
func TestLiveStateFollowsTheEngine(t *testing.T) {
	var m requestMonitor
	m.init()
	for engine, want := range map[string]string{"stopped": "unloaded", "starting": "loading", "failed": "error", "ready": "idle"} {
		live := m.live(engine == "ready")
		overlayEngineState(live, engine)
		if live["state"] != want {
			t.Errorf("engine %q: state %v, want %q", engine, live["state"], want)
		}
	}
	m.begin(10, 10)
	live := m.live(true)
	overlayEngineState(live, "starting")
	if live["state"] != "reading" {
		t.Errorf("a running request must keep its state, got %v", live["state"])
	}
}
