package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The web app (pkg/frontend/web/app.js) fetches these paths and reads JSON: a 404 or the page's HTML breaks a view.

func getJSON(t *testing.T, url string) (int, map[string]interface{}) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	var out map[string]interface{}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("GET %s: not JSON (%s): %.80s", url, resp.Header.Get("Content-Type"), body)
	}
	return resp.StatusCode, out
}

func TestWebUISettingsRoundTrip(t *testing.T) {
	ts := httptest.NewServer(setupTestServer().Router())
	defer ts.Close()
	code, got := getJSON(t, ts.URL+"/settings")
	if code != 200 || got["shared"] != false {
		t.Fatalf("GET /settings = %d %v", code, got)
	}
	resp, err := http.Post(ts.URL+"/settings", "application/json",
		strings.NewReader(`{"defaults":{"reasoning_effort":"low","temperature":0.5,"top_k":20,"max_tokens":256}}`))
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("POST /settings: %v %v", resp, err)
	}
	_, got = getJSON(t, ts.URL+"/settings")
	d, _ := got["defaults"].(map[string]interface{})
	if got["shared"] != true || d["reasoning_effort"] != "low" || d["max_tokens"] != float64(256) {
		t.Errorf("not kept: %v", got)
	}
	bad, _ := http.Post(ts.URL+"/settings", "application/json", strings.NewReader(`{"defaults":{"temperature":9}}`))
	if bad.StatusCode != 400 {
		t.Errorf("a bad value must be a 400, got %d", bad.StatusCode)
	}
	resp, _ = http.Post(ts.URL+"/settings", "application/json", strings.NewReader(`{"defaults":null}`))
	_, got = getJSON(t, ts.URL+"/settings")
	if resp.StatusCode != 200 || got["shared"] != false {
		t.Errorf("null must switch sharing off: %v", got)
	}
}

func TestSharedDefaultsFillWhatARequestLeavesOut(t *testing.T) {
	srv := setupTestServer()
	if _, err := srv.setShared(map[string]interface{}{"reasoning_effort": "low", "temperature": 0.5, "max_tokens": float64(256)}); err != nil {
		t.Fatal(err)
	}
	var req oaRequest
	srv.applySharedOA(&req)
	if req.ReasoningEffort != "low" || req.Temperature == nil || *req.Temperature != 0.5 || req.MaxTokens == nil || *req.MaxTokens != 256 {
		t.Errorf("defaults not applied: %+v", req)
	}
	own := 0.9
	req2 := oaRequest{Temperature: &own, ReasoningEffort: "high"}
	srv.applySharedOA(&req2)
	if *req2.Temperature != 0.9 || req2.ReasoningEffort != "high" {
		t.Errorf("a request's own values must win: %+v", req2)
	}
}

func TestWebUIRunConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "strata-m.json")
	if err := os.WriteFile(path, []byte(`{"exe":"e","args":["--max-context","4096"],"api_key":"k"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := ServerConfig{Port: 8080, ModelName: "m", MaxContext: 4096, VirtualLimit: 1 << 20, ConfigFile: path}
	ts := httptest.NewServer(NewStrataServer(cfg).Router())
	defer ts.Close()

	code, got := getJSON(t, ts.URL+"/config")
	keys, _ := got["keys"].([]interface{})
	if code != 200 || got["file"] != "strata-m.json" || len(keys) == 0 {
		t.Fatalf("GET /config must list the editable keys: %d %v", code, got)
	}
	resp, err := http.Post(ts.URL+"/config", "application/json", strings.NewReader(`{"set":{"power_policy":"LOW_LATENCY"}}`))
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("POST /config: %v %v", resp, err)
	}
	var post map[string]interface{}
	_ = json.NewDecoder(resp.Body).Decode(&post)
	if c, _ := post["changed"].([]interface{}); len(c) != 1 || c[0] != "power_policy" {
		t.Errorf("changed = %v", post["changed"])
	}
	raw, _ := os.ReadFile(path)
	if !strings.Contains(string(raw), `"LOW_LATENCY"`) || !strings.Contains(string(raw), `"api_key"`) {
		t.Errorf("file = %s", raw)
	}
	if _, err := os.Stat(path + ".bak"); err != nil {
		t.Errorf("the earlier file must be kept: %v", err)
	}
	bad, _ := http.Post(ts.URL+"/config", "application/json", strings.NewReader(`{"set":{"api_key":"x"}}`))
	if bad.StatusCode != 400 {
		t.Errorf("a key outside the list must be a 400, got %d", bad.StatusCode)
	}
}

// A plain form post from any site (text/plain, form-urlencoded) must not change the run config or the shared settings:
// only JSON, which a page elsewhere cannot send without a CORS preflight this server never grants.
func TestStateChangingPostsNeedJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "strata-m.json")
	if err := os.WriteFile(path, []byte(`{"exe":"e","args":["--x","1"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(NewStrataServer(ServerConfig{Port: 8080, ModelName: "m", MaxContext: 4096, ConfigFile: path}).Router())
	defer ts.Close()
	for _, c := range []struct{ path, body string }{
		{"/config", `{"set":{"power_policy":"LOW_LATENCY"}}`},
		{"/settings", `{"defaults":{"temperature":0.5}}`},
	} {
		for _, ctype := range []string{"text/plain", "application/x-www-form-urlencoded", ""} {
			req, _ := http.NewRequest("POST", ts.URL+c.path, strings.NewReader(c.body))
			if ctype != "" {
				req.Header.Set("Content-Type", ctype)
			}
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			resp.Body.Close()
			if resp.StatusCode != http.StatusUnsupportedMediaType {
				t.Errorf("POST %s as %q = %d, want 415", c.path, ctype, resp.StatusCode)
			}
		}
	}
	if raw, _ := os.ReadFile(path); strings.Contains(string(raw), "LOW_LATENCY") {
		t.Error("a refused post changed the run config")
	}
	if _, got := getJSON(t, ts.URL+"/settings"); got["shared"] != false {
		t.Errorf("a refused post changed the shared settings: %v", got)
	}
}

func TestWebUIRunConfigMissing(t *testing.T) {
	ts := httptest.NewServer(setupTestServer().Router())
	defer ts.Close()
	resp, _ := http.Get(ts.URL + "/config")
	if resp.StatusCode != 404 {
		t.Errorf("no run config: want 404 (the view hides its card), got %d", resp.StatusCode)
	}
}
