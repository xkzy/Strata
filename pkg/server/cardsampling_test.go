package server

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRequestWithoutSamplingGetsTheModelCard(t *testing.T) {
	srv := NewStrataServer(ServerConfig{Port: 0})
	think, err := (&oaRequest{Messages: []oaMessage{{Role: "user", Content: []byte(`"hi"`)}}}).toSpec(srv)
	if err != nil {
		t.Fatal(err)
	}
	if g := think.Sampling; g.Temperature != 1.0 || g.TopP != 0.95 || g.TopK != 20 || g.PresencePenalty != 0 {
		t.Errorf("thinking: %+v", g)
	}
	off := false
	plain, _ := (&oaRequest{Messages: think2(), EnableThinking: &off}).toSpec(srv)
	if g := plain.Sampling; g.Temperature != 0.7 || g.TopP != 0.8 || g.PresencePenalty != 1.5 {
		t.Errorf("not thinking: %+v", g)
	}
	zero := 0.0
	greedy, _ := (&oaRequest{Messages: think2(), Temperature: &zero}).toSpec(srv)
	if greedy.Sampling.Temperature != 0 {
		t.Errorf("temperature 0 must stay greedy: %+v", greedy.Sampling)
	}
}

func TestConfigSamplingBlockBeatsTheCard(t *testing.T) {
	srv := NewStrataServer(ServerConfig{Port: 0})
	srv.eng.samplingSet = true // a block of zeros: greedy, asked for
	if g := srv.defaultSampling(true); g.Temperature != 0 || g.TopK != 0 {
		t.Errorf("an explicit config block must win over the card: %+v", g)
	}
}

func think2() []oaMessage { return []oaMessage{{Role: "user", Content: []byte(`"hi"`)}} }

func TestToolArgumentsInHistoryAreKeptNotRefused(t *testing.T) {
	for in, want := range map[string]string{
		`"{\"a\":1}"`: `{"a":1}`,
		`{"a":1}`:     `{"a":1}`,
		`"{\"a\": 1"`: `{"arguments":"{\"a\": 1"}`, // cut off by max_tokens
		`"[1,2]"`:     `{"arguments":"[1,2]"}`,
		`"  "`:        ``,
		`null`:        ``,
	} {
		got, err := toolArguments([]byte(in))
		if err != nil || string(got) != want {
			t.Errorf("%s: got %q, %v; want %q", in, got, err, want)
		}
	}
}

func TestSeveralAPIKeys(t *testing.T) {
	srv := NewStrataServer(ServerConfig{Port: 0, APIKey: "k1, k2 ,,k3"})
	for key, want := range map[string]bool{"k1": true, "k2": true, "k3": true, "k": false, "": false, "k1,k2": false} {
		r, _ := http.NewRequest("GET", "/v1/models", nil)
		r.Header.Set("Authorization", "Bearer "+key)
		if got := srv.authorized(r); got != want {
			t.Errorf("Bearer %q: %v, want %v", key, got, want)
		}
		r2, _ := http.NewRequest("GET", "/v1/models", nil)
		r2.Header.Set("X-Api-Key", key)
		if got := srv.authorized(r2); got != want && key != "" {
			t.Errorf("x-api-key %q: %v, want %v", key, got, want)
		}
	}
}

func TestSplitJoinedArgs(t *testing.T) {
	got := splitJoinedArgs([]string{"--pack", "p", "--ple-io mmap", "--x=1", "--prompt", "--not a flag", "-v", "plain value"})
	want := []string{"--pack", "p", "--ple-io", "mmap", "--x", "1", "--prompt", "--not a flag", "-v", "plain value"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Errorf("got %q\nwant %q", got, want)
	}
}

func TestEngineSpecJSONMistakeNamesThePlace(t *testing.T) {
	path := filepath.Join(t.TempDir(), "strata-x.json")
	os.WriteFile(path, []byte("{\n  \"args\": [\"--pack\" \"p\"]\n}"), 0o600)
	_, err := readEngineSpec(path)
	if err == nil || !strings.Contains(err.Error(), "line 2") {
		t.Errorf("error = %v", err)
	}
}
