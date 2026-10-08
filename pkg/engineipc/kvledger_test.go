package engineipc

import (
	"testing"
	"time"
)

var t0 = time.Unix(1000, 0)

func TestParseKVEventValid(t *testing.T) {
	ev, ok := ParseKVEvent("KVEV v1 restore tier=checkpoint tokens=0-9276 bytes=123456 ms=51.2", t0)
	if !ok {
		t.Fatal("valid line rejected")
	}
	if ev.Event != "restore" || ev.Tier != "checkpoint" || ev.FirstTok != 0 || ev.LastTok != 9276 ||
		ev.Bytes != 123456 || ev.MS != 51.2 || !ev.At.Equal(t0) {
		t.Errorf("fields: %+v", ev)
	}
}

func TestParseKVEventToleratesExtraKeys(t *testing.T) {
	if _, ok := ParseKVEvent("KVEV v1 park tier=parked tokens=1-5 bytes=10 ms=1.5 future=x", t0); !ok {
		t.Error("an extra key on v1 must be tolerated")
	}
}

func TestParseKVEventRejects(t *testing.T) {
	for _, line := range []string{
		"",
		"strata serve: conversation cache: restored 5 tokens",
		"KVEV v2 restore tier=live tokens=0-5 bytes=1 ms=1",
		"KVEV v1 explode tier=live tokens=0-5 bytes=1 ms=1",
		"KVEV v1 restore tier=live tokens=0-5 bytes=abc ms=1",
		"KVEV v1 restore tier=live tokens=9-5 bytes=1 ms=1",
		"KVEV v1 restore tier=live tokens=0-5 bytes=1",
		"KVEV v1 restore tier=live tokens=0-5 bytes=-1 ms=1",
		"KVEV v1 restore tier=disk tokens=0-5 bytes=1 ms=1",
		"KVEV v1 restore tier=live tokens=0-5 bytes=1 ms=NaN",
	} {
		if ev, ok := ParseKVEvent(line, t0); ok {
			t.Errorf("accepted %q as %+v", line, ev)
		}
	}
}

func TestKVLedgerBoundedAndTotals(t *testing.T) {
	var l KVLedger
	if tot := l.Totals(); tot.LastEvent != nil || tot.Parks != 0 {
		t.Errorf("empty ledger: %+v", tot)
	}
	for i := 0; i < 600; i++ {
		ev := KVEvent{Event: "restore", Tier: "parked", LastTok: int64(i), MS: 1, At: t0.Add(time.Duration(i) * time.Second)}
		if i%3 == 0 {
			ev.Event = "park"
		}
		l.Add(ev)
	}
	evs := l.Events()
	if len(evs) != 256 {
		t.Fatalf("ledger holds %d events, want 256", len(evs))
	}
	if evs[0].LastTok != 600-256 || evs[255].LastTok != 599 {
		t.Errorf("order/trim: first=%d last=%d", evs[0].LastTok, evs[255].LastTok)
	}
	tot := l.Totals()
	if tot.Parks != 200 || tot.Restores != 400 || tot.LastEvent == nil || tot.LastEvent.LastTok != 599 {
		t.Errorf("totals: %+v", tot)
	}
	evs[0].Event = "changed"
	if l.Events()[0].Event == "changed" {
		t.Error("Events() aliases internal storage")
	}
}

func TestNoteStderrLineFeedsLedger(t *testing.T) {
	eo := &EngineOrchestrator{}
	eo.noteStderrLine("strata serve: conversation cache: restored 5 tokens (live) in 1.0 ms", t0)
	eo.noteStderrLine("KVEV v1 park tier=parked tokens=0-100 bytes=4096 ms=2.5", t0)
	eo.noteStderrLine("KVEV v1 garbage", t0)
	evs := eo.KVLedger().Events()
	if len(evs) != 1 || evs[0].Event != "park" || evs[0].Bytes != 4096 {
		t.Fatalf("ledger after stderr lines: %+v", evs)
	}
	if got := eo.logTailLen(); got != 3 {
		t.Errorf("every line still goes to the log tail, got %d", got)
	}
}

func TestParseKVEventParkedKeys(t *testing.T) {
	ev, ok := ParseKVEvent("KVEV v1 park tier=parked tokens=0-100 bytes=4096 ms=2.5 parked=2 parked_bytes=8192", t0)
	if !ok || !ev.HasParked || ev.Parked != 2 || ev.ParkedBytes != 8192 {
		t.Fatalf("parked keys: ok=%v %+v", ok, ev)
	}
	plain, _ := ParseKVEvent("KVEV v1 park tier=parked tokens=0-100 bytes=4096 ms=2.5", t0)
	if plain.HasParked {
		t.Error("a line without the keys must not claim a parked count")
	}
	for _, bad := range []string{"parked=x parked_bytes=1", "parked=1 parked_bytes=-5", "parked=1"} {
		if _, ok := ParseKVEvent("KVEV v1 park tier=parked tokens=0-1 bytes=1 ms=1 "+bad, t0); !ok {
			continue // a malformed optional key may drop the whole line or just the keys, but never invent a count
		}
		ev, _ := ParseKVEvent("KVEV v1 park tier=parked tokens=0-1 bytes=1 ms=1 "+bad, t0)
		if ev.HasParked && (ev.Parked < 0 || ev.ParkedBytes < 0) {
			t.Errorf("%q gave a negative count: %+v", bad, ev)
		}
	}
}

func TestKVLedgerTracksParkedState(t *testing.T) {
	var l KVLedger
	if tot := l.Totals(); tot.HasParked {
		t.Error("no parked state before any event reports it")
	}
	l.Add(KVEvent{Event: "park", HasParked: true, Parked: 2, ParkedBytes: 500})
	l.Add(KVEvent{Event: "checkpoint"}) // does not report it: the last known value stays
	l.Add(KVEvent{Event: "restore", HasParked: true, Parked: 1, ParkedBytes: 200})
	if tot := l.Totals(); !tot.HasParked || tot.Parked != 1 || tot.ParkedBytes != 200 {
		t.Errorf("parked state = %+v", tot)
	}
}
