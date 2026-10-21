package server

import (
	"testing"

	"strata/pkg/rtclient"
)

func TestAttemptNotesNameTheLoopRetries(t *testing.T) {
	tr := []rtclient.TraceEvent{
		{Kind: "generation", Detail: "attempt 1 starts: 50746 prompt tokens, temperature 0.000000"},
		{Kind: "recovery", Detail: "loop: Periodic generation loop detected (period 3 repeated 5 times)."},
		{Kind: "generation", Detail: "attempt 1 ends: 139 tokens, restart requested"},
		{Kind: "generation", Detail: "attempt 2 starts: 50884 prompt tokens, temperature 0.000000"},
		{Kind: "generation", Detail: "attempt 2 ends: 154 tokens, restart requested"},
		{Kind: "recovery", Detail: "loop persisted: stopping safely"},
		{Kind: "generation", Detail: "attempt 3 ends: 169 tokens, engine finished"},
		{Kind: "generation", Detail: "attempt 4 ends: 10 tokens, engine finished"},
	}
	got := attemptNotes(tr)
	want := map[int]string{
		1: "runtime retry after attempt 1: loop: Periodic generation loop detected (period 3 repeated 5 times)",
		2: "runtime restarted the generation after attempt 2",
		3: "runtime: loop persisted: stopping safely",
	}
	if len(got) != len(want) {
		t.Fatalf("got %v", got)
	}
	for n, w := range want {
		if got[n] != w {
			t.Errorf("attempt %d: got %q, want %q", n, got[n], w)
		}
	}
}

func TestMonitorRecordCarriesIDAndNoteInEitherOrder(t *testing.T) {
	var m requestMonitor
	m.init()
	a := m.begin(10, 100)
	a.link("abc123")
	a.finish("cancel", 3, 0)
	a.setNote("runtime retry after attempt 1: loop") // after finish
	b := m.begin(10, 100)
	b.setNote("late") // before finish
	b.link("abc123")
	b.finish("cancel", 3, 0)
	for _, r := range []*monitoredRequest{a, b} {
		if r.rec["id"] != "abc123" || r.rec["note"] == nil {
			t.Errorf("record lacks id or note: %v", r.rec)
		}
	}
}
