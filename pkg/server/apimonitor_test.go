package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// The standalone page (pkg/frontend/web/monitor.js) polls /v1/status and /api/requests, shows one request's input,
// output and reasoning, and has Load / Unload buttons.

func apiRecords(t *testing.T, base string) []interface{} {
	t.Helper()
	code, got := getJSON(t, base+"/api/requests")
	if code != 200 {
		t.Fatalf("GET /api/requests = %d %v", code, got)
	}
	list, _ := got["requests"].([]interface{})
	return list
}

func waitRecords(t *testing.T, base string, n int) []interface{} {
	t.Helper()
	var list []interface{}
	for deadline := time.Now().Add(3 * time.Second); time.Now().Before(deadline); time.Sleep(30 * time.Millisecond) {
		if list = apiRecords(t, base); len(list) >= n {
			if st, _ := list[0].(map[string]interface{})["state"].(string); st != "generating" {
				return list
			}
		}
	}
	return list
}

func TestAPIMonitorIsOffByDefault(t *testing.T) {
	ts := httptest.NewServer(setupTestServer().Router())
	defer ts.Close()
	for _, p := range []string{"/api/requests", "/monitor", "/api-monitor"} {
		resp, _ := http.Get(ts.URL + p)
		if resp.StatusCode != 404 {
			t.Errorf("GET %s with the monitor off = %d, want 404 (nothing is kept, no page)", p, resp.StatusCode)
		}
	}
}

func TestAPIMonitorKeepsARequest(t *testing.T) {
	t.Setenv("STRATA_API_MONITOR", "1")
	ts, _ := startChatServer(t, "Thinking it over.\n</think>\n\nHello from the model!")
	if resp, body := postJSON(t, ts.URL+"/v1/chat/completions", `{"model":"x","messages":[{"role":"user","content":"hi there"}]}`); resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, body)
	}
	list := waitRecords(t, ts.URL, 1)
	if len(list) != 1 {
		t.Fatalf("records: %v", list)
	}
	r := list[0].(map[string]interface{})
	if r["state"] != "completed" || r["path"] != "/v1/chat/completions" || r["http_status"] != float64(200) || r["stream"] != false {
		t.Errorf("summary: %v", r)
	}
	for _, heavy := range []string{"input", "output", "reasoning", "response"} {
		if _, ok := r[heavy]; ok {
			t.Errorf("the list must not carry %q", heavy)
		}
	}
	if _, ok := r["wallclock_s"].(float64); !ok || r["started_at"] == nil {
		t.Errorf("wallclock_s / started_at missing: %v", r)
	}
	if sc, _ := r["scope"].(map[string]interface{}); sc["session_id"] == nil {
		t.Errorf("scope: %v", r["scope"])
	}
	_, d := getJSON(t, ts.URL+"/api/requests?id="+r["id"].(string))
	if !strings.Contains(d["input"].(string), "hi there") || d["output"] != "Hello from the model!" || d["reasoning"] != "Thinking it over." {
		t.Errorf("detail: input=%v output=%q reasoning=%q", d["input"], d["output"], d["reasoning"])
	}
	if u, _ := d["usage"].(map[string]interface{}); u["completion_tokens"] == nil {
		t.Errorf("usage: %v", d["usage"])
	}
	if d["response"] == nil || d["first_token_s"] == nil {
		t.Errorf("response / first_token_s: %v %v", d["response"], d["first_token_s"])
	}
	resp, _ := http.Get(ts.URL + "/api/requests?id=nope")
	if resp.StatusCode != 404 {
		t.Errorf("an unknown id must be 404, got %d", resp.StatusCode)
	}
}

func TestAPIMonitorKeepsAStreamedRequest(t *testing.T) {
	t.Setenv("STRATA_API_MONITOR", "1")
	ts, _ := startChatServer(t, "Thinking it over.\n</think>\n\nHello from the model!")
	if resp, body := postJSON(t, ts.URL+"/v1/chat/completions", `{"model":"x","stream":true,"messages":[{"role":"user","content":"hi"}]}`); resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, body)
	}
	list := waitRecords(t, ts.URL, 1)
	r := list[0].(map[string]interface{})
	if r["state"] != "completed" || r["stream"] != true {
		t.Fatalf("summary: %v", r)
	}
	_, d := getJSON(t, ts.URL+"/api/requests?id="+r["id"].(string))
	if d["output"] != "Hello from the model!" || d["reasoning"] != "Thinking it over." {
		t.Errorf("streamed capture: output=%q reasoning=%q", d["output"], d["reasoning"])
	}
}

func TestAPIMonitorKeepsAFailedRequest(t *testing.T) {
	t.Setenv("STRATA_API_MONITOR", "1")
	ts := httptest.NewServer(setupTestServer().Router()) // no model
	defer ts.Close()
	postJSON(t, ts.URL+"/v1/chat/completions", `{"model":"x","messages":[{"role":"user","content":"hi"}]}`)
	list := waitRecords(t, ts.URL, 1)
	r := list[0].(map[string]interface{})
	if r["state"] != "error" || r["http_status"] != float64(503) {
		t.Fatalf("summary: %v", r)
	}
	_, d := getJSON(t, ts.URL+"/api/requests?id="+r["id"].(string))
	if e, _ := d["error"].(map[string]interface{}); e["message"] == nil {
		t.Errorf("error: %v", d["error"])
	}
	if d["first_token_s"] != nil {
		t.Errorf("a failed request has no first token: %v", d["first_token_s"])
	}
}

func TestV1StatusFeedsTheStandalonePage(t *testing.T) {
	ts, _ := startChatServer(t, "ok")
	_, st := getJSON(t, ts.URL+"/v1/status")
	for _, k := range []string{"model", "architecture", "swa", "runtime_stats", "activity", "machine", "loaded", "cache_max_tokens", "auto_load"} {
		if st[k] == nil {
			t.Errorf("/v1/status lacks %q: %v", k, st)
		}
	}
	if a, _ := st["activity"].(map[string]interface{}); a["in_flight"] != float64(0) {
		t.Errorf("activity: %v", st["activity"])
	}
	if st["loaded"] != true {
		t.Errorf("loaded: %v", st["loaded"])
	}
}

func TestLoadAndUnload(t *testing.T) {
	ts, _ := startChatServer(t, "back again")
	post := func(path, ctype string) (int, map[string]interface{}) {
		req, _ := http.NewRequest("POST", ts.URL+path, strings.NewReader("{}"))
		req.Header.Set("Content-Type", ctype)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var out map[string]interface{}
		_ = json.NewDecoder(resp.Body).Decode(&out)
		return resp.StatusCode, out
	}
	if code, _ := post("/unload", "text/plain"); code == 200 {
		t.Error("a form post must not unload the model")
	}
	if code, out := post("/unload", "application/json"); code != 200 || out["status"] != "unloaded" {
		t.Fatalf("unload: %d %v", code, out)
	}
	if _, st := getJSON(t, ts.URL+"/v1/status"); st["loaded"] != false {
		t.Errorf("after unload: %v", st["loaded"])
	}
	if code, out := post("/load", "application/json"); code != 200 || out["status"] != "loaded" {
		t.Fatalf("load: %d %v", code, out)
	}
	if _, st := getJSON(t, ts.URL+"/v1/status"); st["loaded"] != true {
		t.Errorf("after load: %v", st["loaded"])
	}
	resp, body := postJSON(t, ts.URL+"/v1/chat/completions", `{"model":"x","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 200 || !strings.Contains(string(body), "back again") {
		t.Errorf("a loaded model must answer: %d %s", resp.StatusCode, body)
	}
}

func TestUnloadRefusesWhileARequestRuns(t *testing.T) {
	srv := setupTestServer()
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()
	rq := srv.mon.begin(10, 10)
	defer rq.finish("stop", 0, 0)
	resp, err := http.Post(ts.URL+"/unload", "application/json", strings.NewReader("{}"))
	if err != nil || resp.StatusCode != 409 {
		t.Errorf("unload while busy = %v %v, want 409", resp, err)
	}
}

func TestPageAssetsLoadWithoutAnAPIKey(t *testing.T) {
	t.Setenv("STRATA_API_MONITOR", "1")
	srv := NewStrataServer(ServerConfig{Port: 8080, APIKey: "secret-token-123", ModelName: "m", MaxContext: 4096})
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()
	// a browser fetches the page and its scripts before the user can type a key
	for _, p := range []string{"/", "/web/app.js", "/web/app.css", "/web/tokens.css", "/web/sprite.svg", "/monitor", "/web/monitor.js"} {
		if resp, _ := http.Get(ts.URL + p); resp.StatusCode != 200 {
			t.Errorf("GET %s with an API key set = %d, want 200", p, resp.StatusCode)
		}
	}
	for _, p := range []string{"/metrics", "/v1/status", "/api/requests", "/settings"} {
		if resp, _ := http.Get(ts.URL + p); resp.StatusCode != 401 {
			t.Errorf("GET %s without the key = %d, want 401", p, resp.StatusCode)
		}
	}
}
