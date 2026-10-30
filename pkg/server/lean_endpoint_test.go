package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func TestMathProveEndpoint(t *testing.T) {
	srv := setupTestServer()
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()

	post := func(payload string) map[string]interface{} {
		resp, err := http.Post(ts.URL+"/v1/strata/math/evaluate", "application/json", strings.NewReader(payload))
		if err != nil || resp.StatusCode != http.StatusOK {
			t.Fatalf("prove request failed: %v", err)
		}
		var res map[string]interface{}
		json.NewDecoder(resp.Body).Decode(&res)
		return res
	}
	if res := post(`{"operation":"prove","expression":"2^10 = 1024"}`); res["exact_result"] != "proved" || res["backend_name"] != "Lean4-Kernel-Go" {
		t.Errorf("proved expected: %v", res)
	}
	res := post(`{"operation":"prove","expression":"2 + 2 = 5"}`)
	if res["exact_result"] != "refuted" {
		t.Errorf("refuted expected: %v", res)
	}
	if res := post(`{"operation":"prove","expression":"-3 < 2 ∧ 3 - 5 = -2"}`); res["exact_result"] != "proved" {
		t.Errorf("an Int claim should be proved: %v", res)
	}
	if md, _ := res["metadata"].(map[string]interface{}); md["proof_term"] == nil {
		t.Errorf("the certificate should be in metadata: %v", res)
	}

	resp, err := http.Get(ts.URL + "/v1/strata/math/backends")
	if err != nil {
		t.Fatal(err)
	}
	var b map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&b)
	if b["lean_available"] != true {
		t.Errorf("backends: %v", b)
	}
	// The prelude admits nothing: the field is an empty list, not a missing one, so a client can tell "none" from
	// "this build does not say".
	if ax, ok := b["lean_admitted_axioms"].([]interface{}); !ok || len(ax) != 0 {
		t.Errorf("no axiom should be admitted: %v", b["lean_admitted_axioms"])
	}
}

func TestInterceptedLeanVerdictIsNotAnEquation(t *testing.T) {
	ts, promptFile := startChatServer(t, "ok")
	body := `{"messages":[{"role":"user","content":"example : 2 + 2 = 5 := by decide"}]}`
	if resp, b := postJSON(t, ts.URL+"/v1/chat/completions", body); resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	raw, _ := os.ReadFile(promptFile)
	if !strings.Contains(string(raw), "[Lean kernel: 2 + 2 = 5 => refuted]") {
		t.Fatalf("prompt lacks the Lean verdict: %s", raw)
	}
}
