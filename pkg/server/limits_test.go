package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// The math, theorem, CAS, verification and control endpoints take small JSON; they read the body into memory, so an
// unbounded body is a way to exhaust the server's memory. The model APIs keep their own (much larger) limit: a chat
// request legitimately carries a whole conversation.

func postBody(t *testing.T, url, body string) int {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	resp.Body.Close()
	return resp.StatusCode
}

func TestNonModelEndpointsCapRequestBodies(t *testing.T) {
	ts := httptest.NewServer(setupTestServer().Router())
	defer ts.Close()
	pad := strings.Repeat("a", 2<<20) // 2 MiB
	for _, path := range []string{
		"/v1/strata/cas/solve", "/v1/strata/math/evaluate", "/v1/strata/math/verify",
		"/v1/strata/math/theorems/verify", "/v1/strata/verify", "/v1/strata/hallucination/evaluate",
	} {
		body := `{"expression":"1+1","name":"wilson","claim":"1+1=2","args":[],"pad":"` + pad + `"}`
		if code := postBody(t, ts.URL+path, body); code != http.StatusBadRequest && code != http.StatusRequestEntityTooLarge {
			t.Errorf("POST %s with a 2 MiB body = %d, want 400 or 413", path, code)
		}
	}
	for path, body := range map[string]string{
		"/v1/strata/cas/solve":     `{"expression":"2+2"}`,
		"/v1/strata/math/evaluate": `{"expression":"2+2"}`,
	} {
		if code := postBody(t, ts.URL+path, body); code != http.StatusOK {
			t.Errorf("POST %s with a small body = %d, want 200", path, code)
		}
	}
}

func TestModelAPIsKeepTheirLargeBodyLimit(t *testing.T) {
	ts := httptest.NewServer(setupTestServer().Router())
	defer ts.Close()
	// the bulk goes in a field the request ignores: this tests the body limit, not how fast a long prompt is rendered
	body := `{"model":"x","messages":[{"role":"user","content":"hi"}],"pad":"` + strings.Repeat("a", 2<<20) + `"}`
	// no engine in this test: a 503 means the body was read and the request reached the model layer
	if code := postBody(t, ts.URL+"/v1/chat/completions", body); code != http.StatusServiceUnavailable {
		t.Errorf("a 2 MiB chat request = %d, want 503 (read in full, no engine)", code)
	}
}
