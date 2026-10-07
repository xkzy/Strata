package rtclient

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

// The test binary doubles as a fake sidecar (STRATA_FAKE_RT=1): it answers config with ready and every request with
// output of an earlier request id first (what an abandoned request leaves behind), then the real answer.
func TestMain(m *testing.M) {
	if os.Getenv("STRATA_FAKE_RT") == "1" {
		fakeSidecar()
		return
	}
	os.Exit(m.Run())
}

func fakeSidecar() {
	r := bufio.NewReader(os.Stdin)
	out := func(v map[string]any) { b, _ := json.Marshal(v); fmt.Println(string(b)) }
	for {
		line, err := r.ReadBytes('\n')
		if err != nil {
			return
		}
		var m map[string]any
		if json.Unmarshal(line, &m) != nil {
			continue
		}
		switch m["op"] {
		case "config":
			out(map[string]any{"op": "ready"})
		case "request":
			if os.Getenv("STRATA_FAKE_MODE") == "hang" {
				continue // never answers, ignores cancel
			}
			id := m["id"].(float64)
			out(map[string]any{"op": "emit", "id": id - 1, "text": "STALE"})
			out(map[string]any{"op": "done", "id": id - 1, "finish": "stale"})
			out(map[string]any{"op": "emit", "id": id, "text": "OK"})
			out(map[string]any{"op": "done", "id": id, "finish": "stop"})
		case "quit":
			return
		}
	}
}

func TestRunIgnoresMessagesOfOtherRequests(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Skip(err)
	}
	t.Setenv("STRATA_FAKE_RT", "1")
	c := New(Config{Binary: exe, StartTimeout: 10 * time.Second})
	defer c.Close()
	for i := 0; i < 3; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		var got strings.Builder
		d, err := c.Run(ctx, Request{Messages: []Message{{Role: "user", Content: "hi"}}}, nil, func(s string) { got.WriteString(s) })
		cancel()
		if err != nil {
			t.Fatalf("run %d: %v", i, err)
		}
		if got.String() != "OK" {
			t.Fatalf("run %d: delivered %q, want only the current request's text", i, got.String())
		}
		if d.Finish != "stop" {
			t.Fatalf("run %d: finish %q came from another request", i, d.Finish)
		}
	}
}

func TestStuckCancelKillsSidecarAndNextRequestIsFresh(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Skip(err)
	}
	t.Setenv("STRATA_FAKE_RT", "1")
	t.Setenv("STRATA_FAKE_MODE", "hang")
	c := New(Config{Binary: exe, StartTimeout: 10 * time.Second, DrainTimeout: 200 * time.Millisecond})
	defer c.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	_, err = c.Run(ctx, Request{Messages: []Message{{Role: "user", Content: "hi"}}}, nil, nil)
	ae, ok := err.(*APIError)
	if !ok || ae.Status != 504 {
		t.Fatalf("want a 504 after the drain timeout, got %v", err)
	}
	// the stuck sidecar is gone; the next request starts a fresh one (which now behaves)
	t.Setenv("STRATA_FAKE_MODE", "")
	var got strings.Builder
	ctx2, cancel2 := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel2()
	d, err := c.Run(ctx2, Request{Messages: []Message{{Role: "user", Content: "hi"}}}, nil, func(s string) { got.WriteString(s) })
	if err != nil || d.Finish != "stop" || got.String() != "OK" {
		t.Fatalf("fresh sidecar: err=%v done=%+v text=%q", err, d, got.String())
	}
}
