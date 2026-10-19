package server

import (
	"net/http/httptest"
	"testing"
	"time"
)

func TestMonitorTracksARequest(t *testing.T) {
	var m requestMonitor
	m.init()
	if live := m.live(true); live["state"] != "idle" {
		t.Fatalf("no request: %v", live)
	}
	r := m.begin(100, 256)
	if live := m.live(true); live["state"] != "reading" || live["prompt_tokens"] != 100 || live["queued"] != 0 {
		t.Fatalf("before the first token: %v", live)
	}
	r.token()
	time.Sleep(20 * time.Millisecond)
	r.token()
	live := m.live(true)
	if live["state"] != "generating" || live["generated"] != 2 || live["max_tokens"] != 256 {
		t.Fatalf("while generating: %v", live)
	}
	if v, _ := live["tok_s"].(float64); v <= 0 {
		t.Errorf("tok_s must be positive while generating: %v", live["tok_s"])
	}
	r.finish("stop", 2, 0)
	r.finish("error", 9, 0) // only the first end counts
	if live := m.live(true); live["state"] != "idle" {
		t.Errorf("after the end: %v", live)
	}
	reqs, kept, totals := m.view(false)
	if kept != 1 || len(reqs) != 1 {
		t.Fatalf("history: %d %v", kept, reqs)
	}
	got := reqs[0]
	if got["finish"] != "stop" || got["prompt_tokens"] != 100 || got["output_tokens"] != 2 || got["time"] == nil {
		t.Errorf("record: %v", got)
	}
	if totals["requests"] != 1 || totals["prompt_tokens"] != 100 || totals["output_tokens"] != 2 || totals["since"] == nil {
		t.Errorf("totals: %v", totals)
	}
}

func TestMonitorKeepsNewestFirstAndBounded(t *testing.T) {
	var m requestMonitor
	m.init()
	for i := 0; i < monitorKeep+5; i++ {
		r := m.begin(i, 1)
		r.finish("stop", 1, 0)
	}
	all, kept, _ := m.view(true)
	if kept != monitorKeep || len(all) != monitorKeep || all[0]["prompt_tokens"] != monitorKeep+4 {
		t.Errorf("kept=%d len=%d first=%v", kept, len(all), all[0]["prompt_tokens"])
	}
	if recent, _, _ := m.view(false); len(recent) != 12 {
		t.Errorf("the page shows the last 12 unless asked for all: %d", len(recent))
	}
}

func TestMetricsFeedsTheMonitorView(t *testing.T) {
	srv := setupTestServer()
	r := srv.mon.begin(50, 10)
	r.token()
	r.finish("length", 1, 0)
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()
	// The collector samples once a second, but every server this test binary builds starts its own collector and
	// each sample runs nvidia-smi (0.03 s alone, 0.7 s for 50 at once on this PC), so the first sample can arrive
	// later than a fixed wait. Poll for it; the assertions below are unchanged.
	var m map[string]interface{}
	for deadline := time.Now().Add(10 * time.Second); ; {
		_, m = getJSON(t, ts.URL+"/metrics")
		if hw, _ := m["hardware"].(map[string]interface{}); hw["cpu"] != nil || time.Now().After(deadline) {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	live, _ := m["live"].(map[string]interface{})
	if live["state"] == nil {
		t.Errorf("live: %v", live)
	}
	if reqs, _ := m["requests"].([]interface{}); len(reqs) != 1 {
		t.Errorf("requests: %v", m["requests"])
	}
	if m["requests_kept"] != float64(1) {
		t.Errorf("requests_kept: %v", m["requests_kept"])
	}
	if tot, _ := m["totals"].(map[string]interface{}); tot["requests"] != float64(1) {
		t.Errorf("totals: %v", m["totals"])
	}
	hw, _ := m["hardware"].(map[string]interface{})
	if hw["cpu"] == nil { // the page reads hardware.cpu and history.cpu (not cpu_util)
		t.Errorf("hardware.cpu missing: %v", hw)
	}
	h, _ := m["history"].(map[string]interface{})
	if h["cpu"] == nil {
		t.Errorf("history.cpu missing: %v", h)
	}
	if _, ok := h["prefill_tok_s_mean"].([]interface{}); !ok { // the web app draws sp-prefill from this key
		t.Errorf("history.prefill_tok_s_mean missing or not a list: %v", h["prefill_tok_s_mean"])
	}
}

func TestChatRequestShowsInTheMonitor(t *testing.T) {
	ts, _ := startChatServer(t, "Hello from the model!")
	resp, body := postJSON(t, ts.URL+"/v1/chat/completions", `{"model":"x","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, body)
	}
	var m map[string]interface{}
	deadline := time.Now().Add(3 * time.Second)
	for {
		_, m = getJSON(t, ts.URL+"/metrics")
		if reqs, _ := m["requests"].([]interface{}); len(reqs) == 1 || time.Now().After(deadline) {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	reqs, _ := m["requests"].([]interface{})
	if len(reqs) != 1 {
		t.Fatalf("the finished request must be listed: %v", m["requests"])
	}
	r := reqs[0].(map[string]interface{})
	if r["finish"] != "stop" || r["prompt_tokens"].(float64) <= 0 || r["output_tokens"].(float64) <= 0 {
		t.Errorf("record: %v", r)
	}
	if live := m["live"].(map[string]interface{}); live["state"] != "idle" {
		t.Errorf("live after the request: %v", live)
	}
	if tot := m["totals"].(map[string]interface{}); tot["requests"] != float64(1) || tot["output_tokens"].(float64) <= 0 {
		t.Errorf("totals: %v", tot)
	}
}

func TestMonitorReportsPrefillSpeed(t *testing.T) {
	var m requestMonitor
	m.init()
	if v := m.currentPrefillTokS(); v != 0 {
		t.Errorf("idle: the sparkline source must be 0, got %v", v)
	}
	r := m.begin(1000, 64)
	if v := m.live(true)["prefill_tok_s_mean"]; v != nil {
		t.Errorf("nothing read yet, no speed expected: %v", v)
	}
	time.Sleep(20 * time.Millisecond)
	r.promptProgress(500, 1000)
	if v, _ := m.live(true)["prefill_tok_s_mean"].(float64); v <= 0 {
		t.Errorf("reading with progress must report a speed: %v", m.live(true))
	}
	if v := m.currentPrefillTokS(); v <= 0 {
		t.Errorf("the sparkline source must follow the request: %v", v)
	}
	r.token()
	if v, _ := m.live(true)["prefill_tok_s_mean"].(float64); v <= 0 {
		t.Errorf("generating must keep the prompt's speed: %v", m.live(true))
	}
	r.finish("stop", 1, 0)
	if v := m.currentPrefillTokS(); v != 0 {
		t.Errorf("after the request: the sparkline source must be 0, got %v", v)
	}
}

func TestRecentStatsUsesFinishedRequests(t *testing.T) {
	var m requestMonitor
	m.init()
	for i := 0; i < 4; i++ {
		r := m.begin(1000+i*100, 100)
		r.token()
		r.token()
		r.finish("stop", 2, 50+float64(i))
	}
	st := m.recentStats(10 * time.Second)
	if st.Samples != 4 || st.PromptP95 != 1300 {
		t.Errorf("prompt p95: %+v", st)
	}
	if st.DecodeTokSMean <= 0 || st.OutputP95 != 2 {
		t.Errorf("decode/output: %+v", st)
	}
	if m.recentStats(time.Nanosecond).Samples != 0 {
		t.Error("a window with no finished requests must be empty")
	}
}
