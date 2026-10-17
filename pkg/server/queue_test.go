package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"strata/pkg/scheduler"
)

func TestRequestTicket(t *testing.T) {
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r.RemoteAddr = "10.0.0.7:5555"
	if tk := requestTicket(r); tk.Tenant != "10.0.0.7" || tk.Priority != scheduler.DefaultPriority {
		t.Errorf("anonymous caller: %+v", tk)
	}
	r.Header.Set("X-Tenant-ID", "acme")
	r.Header.Set("X-User-ID", "ann")
	r.Header.Set("X-Strata-Priority", "9")
	if tk := requestTicket(r); tk.Tenant != "acme/ann" || tk.Priority != 9 {
		t.Errorf("headers: %+v", tk)
	}
	r.Header.Set("Authorization", "Bearer secret-key")
	tk := requestTicket(r)
	if tk.Tenant == "" || tk.Tenant == "acme/ann" || tk.Tenant == "10.0.0.7" {
		t.Errorf("a key should name the caller: %+v", tk)
	}
	if len(tk.Tenant) != 12 { // 6 bytes of a hash in hex, never the key
		t.Errorf("tenant %q is not the key's short hash", tk.Tenant)
	}
	r.Header.Set("X-Strata-Priority", "nope")
	if requestTicket(r).Priority != scheduler.DefaultPriority {
		t.Error("a bad priority falls back to the default")
	}
}

func TestQueueFullIs429WithRetryAfter(t *testing.T) {
	w := httptest.NewRecorder()
	writeAPIError(w, &scheduler.ErrQueueFull{Depth: 32, RetryAfter: 17})
	if w.Code != 429 || w.Header().Get("Retry-After") != "17" {
		t.Errorf("late refusal: %d, Retry-After %q", w.Code, w.Header().Get("Retry-After"))
	}
	w = httptest.NewRecorder()
	writeAPIError(w, &apiError{Status: 429, Type: "rate_limit_exceeded", Msg: "busy"})
	if w.Code != 429 || w.Header().Get("Retry-After") != "5" {
		t.Errorf("pre-check refusal: %d, Retry-After %q", w.Code, w.Header().Get("Retry-After"))
	}
	w = httptest.NewRecorder()
	writeAPIError(w, &apiError{Status: 503, Msg: "loading"})
	if w.Header().Get("Retry-After") != "5" {
		t.Error("503 keeps its Retry-After")
	}
	if _, ok := queueFullError(http.ErrAbortHandler); ok {
		t.Error("another error is not a queue refusal")
	}
}
