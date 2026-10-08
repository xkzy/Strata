package server

import (
	"encoding/json"
	"io"
	"net/http"
	"testing"
)

// With the C++ runtime active, every request goes through it. The Monitor must see those requests too: its Recent
// requests table, the live state and the Context card read the request monitor.
func TestRuntimeRequestsReachTheMonitor(t *testing.T) {
	ts, _ := startRTServer(t, "Thinking about it.\n</think>\n\nHello from the runtime.")
	resp, body := postJSON(t, ts.URL+"/v1/chat/completions", `{"messages":[{"role":"user","content":"hi there"}],"max_tokens":32}`)
	if resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, body)
	}
	r, err := http.Get(ts.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	raw, _ := io.ReadAll(r.Body)
	var m struct {
		Requests []map[string]interface{} `json:"requests"`
		Totals   map[string]interface{}   `json:"totals"`
		Live     map[string]interface{}   `json:"live"`
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	if len(m.Requests) == 0 {
		t.Fatalf("the runtime request is missing from Recent requests: %s", raw)
	}
	got := m.Requests[0]
	if intOf(got["output_tokens"]) <= 0 || intOf(got["prompt_tokens"]) <= 0 || got["finish"] != "stop" {
		t.Errorf("the recorded request: %v", got)
	}
	if intOf(m.Totals["requests"]) < 1 {
		t.Errorf("totals: %v", m.Totals)
	}
	if m.Live["state"] != "idle" {
		t.Errorf("the request is over, the state must be idle, got %v", m.Live["state"])
	}
}
