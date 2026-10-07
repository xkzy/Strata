package server

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// These tests run the real C++ runtime (strata_rt_server) between the API and a fake engine.
func rtBinary(t *testing.T) string {
	t.Helper()
	for _, p := range []string{os.Getenv("STRATA_RT_BIN"), "../../build/strata_rt_server", "../../engine/strata_rt_server"} {
		if p == "" {
			continue
		}
		if abs, err := filepath.Abs(p); err == nil {
			if st, err := os.Stat(abs); err == nil && !st.IsDir() {
				return abs
			}
		}
	}
	t.Skip("strata_rt_server is not built (cmake --build build --target strata_rt_server)")
	return ""
}

func startRTServer(t *testing.T, reply string, extraEnv ...string) (*httptest.Server, string) {
	t.Helper()
	bin := rtBinary(t)
	t.Setenv("STRATA_CONTEXT_DIR", t.TempDir())
	dir := t.TempDir()
	pack := filepath.Join(dir, "pack")
	writeTinyPack(t, filepath.Join(pack, "tokenizer"))
	exe, _ := os.Executable()
	promptFile := filepath.Join(dir, "prompt.txt")
	env := map[string]string{"STRATA_FAKE_ENGINE": "1", "STRATA_FAKE_REPLY": reply, "STRATA_FAKE_PROMPT_FILE": promptFile, "STRATA_FAKE_CONTINUE": "1", "STRATA_FAKE_CTX": "16384"}
	for _, kv := range extraEnv {
		p := strings.SplitN(kv, "=", 2)
		env[p[0]] = p[1]
	}
	spec := map[string]interface{}{"exe": exe, "args": []string{"-test.run=XXX", "--pack", pack, "--max-context", "16384"}, "env": env}
	sb, _ := json.Marshal(spec)
	cfgPath := filepath.Join(dir, "strata-test.json")
	if err := os.WriteFile(cfgPath, sb, 0o644); err != nil {
		t.Fatal(err)
	}
	srv := NewStrataServer(ServerConfig{ModelName: "test-model", MaxContext: 16384, ConfigFile: cfgPath, RuntimeMode: "on", RuntimeBinary: bin, WindowTokens: 2048})
	if err := srv.StartEngine(context.Background()); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := srv.EngineIPC.WaitReady(ctx); err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(srv.Router())
	t.Cleanup(func() { ts.Close(); srv.EngineIPC.Stop(); srv.rt.client.Close() })
	return ts, promptFile
}

func chatContent(t *testing.T, b []byte) (content, reasoning string, trace []map[string]interface{}) {
	t.Helper()
	var out struct {
		Choices []struct {
			Message struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
		Trace []map[string]interface{} `json:"strata_trace"`
	}
	if err := json.Unmarshal(b, &out); err != nil || len(out.Choices) == 0 {
		t.Fatalf("bad response: %s", b)
	}
	return out.Choices[0].Message.Content, out.Choices[0].Message.ReasoningContent, out.Trace
}

func postDebug(t *testing.T, url, body string, hdr map[string]string) (*http.Response, []byte) {
	t.Helper()
	req, _ := http.NewRequest("POST", url, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var sb strings.Builder
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<24)
	for sc.Scan() {
		sb.WriteString(sc.Text() + "\n")
	}
	return resp, []byte(sb.String())
}

func TestRuntimeCorrectsAWrongAnswerInternally(t *testing.T) {
	ts, promptFile := startRTServer(t, "checking the product\n</think>\n\nThe product is 37 * 19 = 713.",
		"STRATA_FAKE_REPLY2=checking the product\n</think>\n\n37 * 19 = 703.")
	resp, b := postDebug(t, ts.URL+"/v1/chat/completions", `{"messages":[{"role":"user","content":"What is 37 * 19?"}]}`, map[string]string{"X-Strata-Debug": "1"})
	if resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	content, reasoning, trace := chatContent(t, b)
	if !strings.Contains(content, "703") || strings.Contains(content, "713") {
		t.Fatalf("the wrong value reached the client: %q", content)
	}
	if reasoning != "checking the product" {
		t.Fatalf("reasoning should pass through untouched: %q", reasoning)
	}
	kinds := ""
	for _, e := range trace {
		kinds += e["kind"].(string) + " "
	}
	if !strings.Contains(kinds, "recovery") || !strings.Contains(kinds, "claim") {
		t.Fatalf("the trace should show the internal check and recovery: %s", kinds)
	}
	raw, _ := os.ReadFile(promptFile)
	if !strings.Contains(string(raw), "Verified facts for this answer") || !strings.Contains(string(raw), "703") {
		t.Fatalf("the regeneration prompt should carry the correction (as ordinary context):\n%s", raw)
	}
	// the request itself was a plain chat request: no tool, no special prompt
	if !strings.Contains(string(raw), "<|im_start|>user\nWhat is 37 * 19?<|im_end|>") {
		t.Fatalf("user turn missing:\n%s", raw)
	}
	// metrics are available to the operator
	mresp, err := http.Get(ts.URL + "/v1/strata/runtime/metrics")
	if err != nil {
		t.Fatal(err)
	}
	var m struct {
		Metrics map[string]float64 `json:"metrics"`
	}
	json.NewDecoder(mresp.Body).Decode(&m)
	if m.Metrics["regenerations"] != 1 || m.Metrics["contradictions"] < 1 {
		t.Fatalf("metrics: %+v", m.Metrics)
	}
}

func TestRuntimeStreamsAndNeverLeaksTheWrongValue(t *testing.T) {
	ts, _ := startRTServer(t, "hmm\n</think>\n\nSure. 37 * 19 = 713.", "STRATA_FAKE_REPLY2=hmm\n</think>\n\nSure. 37 * 19 = 703.")
	resp, b := postDebug(t, ts.URL+"/v1/chat/completions", `{"messages":[{"role":"user","content":"37 * 19?"}],"stream":true}`, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	var content, reasoning string
	for _, line := range strings.Split(string(b), "\n") {
		if !strings.HasPrefix(line, "data: {") {
			continue
		}
		var ch struct {
			Choices []struct {
				Delta struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		json.Unmarshal([]byte(line[6:]), &ch)
		if len(ch.Choices) > 0 {
			content += ch.Choices[0].Delta.Content
			reasoning += ch.Choices[0].Delta.ReasoningContent
		}
	}
	if strings.Contains(content, "713") || !strings.Contains(content, "703") || reasoning != "hmm" || !strings.Contains(string(b), "[DONE]") {
		t.Fatalf("stream content=%q reasoning=%q", content, reasoning)
	}
}

func TestRuntimeVirtualContextRecallsAcrossRequests(t *testing.T) {
	ts, _ := startRTServer(t, "</think>\n\nI do not have that.", "STRATA_FAKE_IF_PROMPT_HAS=code 4321", "STRATA_FAKE_IF_REPLY=</think>\n\nThe code was 4321.")
	sess := map[string]string{"X-Session-ID": "chat-A"}
	post := func(body string, h map[string]string) string {
		resp, b := postDebug(t, ts.URL+"/v1/chat/completions", body, h)
		if resp.StatusCode != 200 {
			t.Fatalf("%d %s", resp.StatusCode, b)
		}
		c, _, _ := chatContent(t, b)
		return c
	}
	post(`{"messages":[{"role":"user","content":"Decision: the deploy code 4321 is final."}],"reasoning_effort":"none"}`, sess)
	for i := 0; i < 40; i++ { // push it far out of the (tiny) window
		post(`{"messages":[{"role":"user","content":"Status note `+strings.Repeat("filler words go here ", 6)+`"}],"reasoning_effort":"none"}`, sess)
	}
	// the agent sends only its latest message: the runtime still has the whole session
	got := post(`{"messages":[{"role":"user","content":"What was the deploy code we discussed earlier?"}],"reasoning_effort":"none"}`, sess)
	if !strings.Contains(got, "4321") {
		t.Fatalf("the earlier decision was not paged back in: %q", got)
	}
	// another session cannot see it
	other := post(`{"messages":[{"role":"user","content":"What was the deploy code we discussed earlier?"}],"reasoning_effort":"none"}`, map[string]string{"X-Session-ID": "chat-B"})
	if strings.Contains(other, "4321") {
		t.Fatalf("sessions leaked: %q", other)
	}
}

func TestRuntimePagesAnOversizedPromptInsteadOfFailing(t *testing.T) {
	ts, _ := startRTServer(t, "</think>\n\nok")
	big := strings.Repeat("a long paragraph of ordinary words that fills the context. ", 400) // ~6000 tokens for a 2048-token model
	resp, b := postDebug(t, ts.URL+"/v1/chat/completions", `{"messages":[{"role":"user","content":"`+big+` Summarize."}],"reasoning_effort":"none"}`, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("an oversized request should be served from a working set, got %d %s", resp.StatusCode, b)
	}
}

func TestRuntimeToolCallsPassThrough(t *testing.T) {
	reply := "</think>\n\nLet me look.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/hosts\n</parameter>\n<parameter=port>\n9090\n</parameter>\n</function>\n</tool_call>"
	ts, _ := startRTServer(t, reply)
	body := `{"messages":[{"role":"user","content":"open the hosts file"}],"tools":[{"type":"function","function":{"name":"read_file","description":"d","parameters":{"type":"object","properties":{"path":{"type":"string"},"port":{"type":"integer"}}}}}]}`
	resp, b := postDebug(t, ts.URL+"/v1/chat/completions", body, nil)
	if resp.StatusCode != 200 || !strings.Contains(string(b), `"finish_reason":"tool_calls"`) || !strings.Contains(string(b), `read_file`) {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	mresp, _ := http.Get(ts.URL + "/v1/strata/runtime/metrics")
	var m struct {
		Metrics map[string]float64 `json:"metrics"`
	}
	json.NewDecoder(mresp.Body).Decode(&m)
	if m.Metrics["claims_checked"] != 0 || m.Metrics["regenerations"] != 0 {
		t.Fatalf("a tool call is not a claim to verify: %+v", m.Metrics)
	}
}

func TestRuntimeHealthReportsItself(t *testing.T) {
	ts, _ := startRTServer(t, "</think>\n\nok")
	postDebug(t, ts.URL+"/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}]}`, nil)
	resp, err := http.Get(ts.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	var h struct {
		Runtime struct {
			Active  bool   `json:"active"`
			Healthy bool   `json:"healthy"`
			Mode    string `json:"mode"`
		} `json:"runtime"`
	}
	json.NewDecoder(resp.Body).Decode(&h)
	if !h.Runtime.Active || !h.Runtime.Healthy || h.Runtime.Mode != "on" {
		t.Fatalf("%+v", h.Runtime)
	}
}

func TestAnonymousCallersNeverShareAContext(t *testing.T) {
	ts, _ := startRTServer(t, "</think>\n\nI do not have that.", "STRATA_FAKE_IF_PROMPT_HAS=code 4321", "STRATA_FAKE_IF_REPLY=</think>\n\nThe code was 4321.")
	ask := func(q string) string {
		resp, b := postDebug(t, ts.URL+"/v1/chat/completions", `{"messages":[{"role":"user","content":"`+q+`"}],"reasoning_effort":"none"}`, nil)
		if resp.StatusCode != 200 {
			t.Fatalf("%d %s", resp.StatusCode, b)
		}
		c, _, _ := chatContent(t, b)
		return c
	}
	// two unrelated anonymous callers happen to open with the same words
	ask("Decision: the deploy code 4321 is final.")
	ask("Decision: the deploy code 4321 is final. What was the deploy code we discussed earlier?")
	// a third, anonymous, asks about it: nothing may be recalled from the first
	prompts := ask("What was the deploy code we discussed earlier?")
	if strings.Contains(prompts, "4321") {
		t.Fatalf("an anonymous caller recalled another anonymous caller's context: %q", prompts)
	}
	mresp, _ := http.Get(ts.URL + "/v1/strata/runtime/metrics")
	var m struct {
		Metrics map[string]float64 `json:"metrics"`
	}
	json.NewDecoder(mresp.Body).Decode(&m)
	if m.Metrics["requests"] != 3 {
		t.Fatalf("metrics: %+v", m.Metrics)
	}
	entries, _ := os.ReadDir(os.Getenv("STRATA_CONTEXT_DIR"))
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".manifest") {
			t.Fatalf("an anonymous request left a stored context behind: %s", e.Name())
		}
	}
}
