package router

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func fake(reply string) (*Backend, *int) {
	n := new(int)
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/models") {
			return
		}
		*n++
		w.Write([]byte(`{"choices":[{"finish_reason":"stop","message":{"content":"` + reply + `"}}]}`))
	}))
	return &Backend{Name: reply, URL: s.URL, Start: func() error { return nil }, Stop: func() error { return nil }}, n
}

func ask(r *Router, hdr string) string {
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{}`))
	if hdr != "" {
		req.Header.Set(escalateHeader, hdr)
	}
	w := httptest.NewRecorder()
	r.Chat(w, req)
	return w.Body.String()
}

func TestEscalateOnRepeatThenReturn(t *testing.T) {
	fast, nf := fake("same")
	dense, nd := fake("better")
	r := &Router{Fast: fast, Dense: dense, DenseRequests: 2, Wait: func(*Backend) error { return nil }}
	ask(r, "")
	if got := ask(r, ""); !strings.Contains(got, "better") { // fast repeated itself
		t.Fatalf("expected dense reply, got %s", got)
	}
	if *nf != 2 || *nd != 1 {
		t.Fatalf("fast %d dense %d", *nf, *nd)
	}
	ask(r, "") // dense, second of its 2
	if got := ask(r, ""); !strings.Contains(got, "same") {
		t.Fatalf("expected fast again, got %s", got)
	}
}

func TestClientEscalate(t *testing.T) {
	fast, _ := fake("a")
	dense, _ := fake("b")
	r := &Router{Fast: fast, Dense: dense, DenseRequests: 1, Wait: func(*Backend) error { return nil }}
	if got := ask(r, "1"); !strings.Contains(got, `"b"`) {
		t.Fatalf("got %s", got)
	}
}

func TestClientBackToFast(t *testing.T) {
	fast, _ := fake("a")
	dense, _ := fake("b")
	r := &Router{Fast: fast, Dense: dense, DenseRequests: 10, Wait: func(*Backend) error { return nil }}
	ask(r, "1")
	if got := ask(r, "0"); !strings.Contains(got, `"a"`) {
		t.Fatalf("got %s", got)
	}
}
