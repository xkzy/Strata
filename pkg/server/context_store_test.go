package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func doReq(t *testing.T, method, url, body string, headers map[string]string) (int, []byte) {
	t.Helper()
	req, _ := http.NewRequest(method, url, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var b strings.Builder
	buf := make([]byte, 4096)
	for {
		n, err := resp.Body.Read(buf)
		b.Write(buf[:n])
		if err != nil {
			break
		}
	}
	return resp.StatusCode, []byte(b.String())
}

// /v1/strata/context/query used to return the audit records of every tenant.
func TestContextQueryOnlyReturnsTheCallersOwnAudit(t *testing.T) {
	ts := httptest.NewServer(setupTestServer().Router())
	defer ts.Close()
	claim := `{"type":"arithmetic","expression":"12 * 12","claimed_value":"150"}`
	if code, b := doReq(t, "POST", ts.URL+"/v1/strata/verify", claim, map[string]string{"X-Tenant-ID": "tenantA", "X-Session-ID": "sA"}); code != 200 {
		t.Fatalf("verify: %d %s", code, b)
	}
	query := func(tenant, session string) int {
		code, b := doReq(t, "POST", ts.URL+"/v1/strata/context/query", `{"query":"arithmetic expected 144 got 150","top_k":5}`,
			map[string]string{"X-Tenant-ID": tenant, "X-Session-ID": session})
		if code != 200 {
			t.Fatalf("query: %d %s", code, b)
		}
		var out struct {
			Hits []interface{} `json:"hits"`
		}
		json.Unmarshal(b, &out)
		return len(out.Hits)
	}
	if n := query("tenantA", "sA"); n == 0 {
		t.Error("the caller must find its own audit record")
	}
	if n := query("tenantB", "sB"); n != 0 {
		t.Errorf("another tenant read %d audit records of tenantA", n)
	}
}

func TestMemoryCanBeClearedByItsOwner(t *testing.T) {
	ts, promptFile := startChatServer(t, "ok")
	chatAs(t, ts.URL, "s1", "My dog is called Biscuit and he is a brown labrador.")
	// another session cannot clear it
	code, b := doReq(t, "DELETE", ts.URL+"/v1/strata/memory", "", map[string]string{"X-Session-ID": "s2"})
	if code != 200 || !strings.Contains(string(b), `"forgotten":0`) {
		t.Fatalf("another session's DELETE: %d %s", code, b)
	}
	code, b = doReq(t, "DELETE", ts.URL+"/v1/strata/memory", "", map[string]string{"X-Session-ID": "s1"})
	if code != 200 || strings.Contains(string(b), `"forgotten":0`) {
		t.Fatalf("own DELETE: %d %s", code, b)
	}
	chatAs(t, ts.URL, "s1", "Remind me, what was the name of my brown labrador dog?")
	if p := promptOf(t, promptFile); strings.Contains(p, "Biscuit") {
		t.Errorf("a cleared session still recalled:\n%s", p)
	}
	if code, _ := doReq(t, "GET", ts.URL+"/v1/strata/memory", "", nil); code == 200 {
		t.Error("only DELETE clears; GET must not")
	}
}

// Two callers with different credentials must not share a session id's memory.
func TestRecallKeyIncludesTheCredential(t *testing.T) {
	ts, promptFile := startChatServer(t, "ok")
	chatAs(t, ts.URL, "shared", "My dog is called Biscuit and he is a brown labrador.", "Authorization", "Bearer key-one")
	chatAs(t, ts.URL, "shared", "Remind me, what was the name of my brown labrador dog?", "Authorization", "Bearer key-two")
	if p := promptOf(t, promptFile); strings.Contains(p, "Biscuit") {
		t.Errorf("a caller with another credential recalled the turn:\n%s", p)
	}
	chatAs(t, ts.URL, "shared", "Remind me, what was the name of my brown labrador dog?", "Authorization", "Bearer key-one")
	if p := promptOf(t, promptFile); !strings.Contains(p, "Biscuit") {
		t.Errorf("the same credential must still recall:\n%s", p)
	}
}
