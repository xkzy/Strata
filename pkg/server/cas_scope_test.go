package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// POST /v1/strata/cas/solve: the caller's tenant and session come from the request's scope (X-Tenant-ID / X-Session-ID
// headers, as in every other handler), never from the body. The session keys the math runtime's variables, and the
// tenant picks backends, so a body that names another caller's session would read or overwrite that caller's variables.

func casSolve(t *testing.T, base, session, body string) string {
	t.Helper()
	req, _ := http.NewRequest("POST", base+"/v1/strata/cas/solve", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if session != "" {
		req.Header.Set("X-Session-ID", session)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("not JSON: %v", err)
	}
	b, _ := json.Marshal(out)
	return string(b)
}

func TestCASSolveIgnoresSessionAndTenantFromTheBody(t *testing.T) {
	ts := httptest.NewServer(setupTestServer().Router())
	defer ts.Close()

	// Alice stores a variable in her own session (named by header) and reads it back there.
	casSolve(t, ts.URL, "alice-session", `{"expression":"secret_x = 7"}`)
	own := casSolve(t, ts.URL, "alice-session", `{"expression":"secret_x"}`)
	if !strings.Contains(own, `"7"`) {
		t.Fatalf("a caller's own session must keep its variables: %s", own)
	}

	// Mallory names Alice's session (and the sage tenant) in the body: that must not reach Alice's variable ...
	stolen := casSolve(t, ts.URL, "mallory-session", `{"expression":"secret_x","session_id":"alice-session","tenant_id":"sage"}`)
	if strings.Contains(stolen, `"7"`) {
		t.Errorf("session_id in the body reached another caller's variable: %s", stolen)
	}

	// ... nor write into it.
	casSolve(t, ts.URL, "mallory-session", `{"expression":"secret_x = 99","session_id":"alice-session"}`)
	after := casSolve(t, ts.URL, "alice-session", `{"expression":"secret_x"}`)
	if strings.Contains(after, "99") || !strings.Contains(after, `"7"`) {
		t.Errorf("session_id in the body overwrote another caller's variable: %s", after)
	}
}
