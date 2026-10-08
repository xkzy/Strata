// pkg/engineipc/kvledger.go - the engine's KV cache events, as measured by the engine.
//
// The engine prints one line per cache event on stderr:
//
//	KVEV v1 <park|restore|checkpoint> tier=<live|checkpoint|parked> tokens=<first>-<last> bytes=<n> ms=<float>
//
// The version prefix lets a newer engine add events or versions without breaking an older server: a line this file
// does not understand is ignored.
package engineipc

import (
	"math"
	"strconv"
	"strings"
	"sync"
	"time"
)

// kvLedgerKeep is how many recent events the ledger retains (the totals count all of them).
const kvLedgerKeep = 256

// KVEvent is one measured cache event.
type KVEvent struct {
	Event    string // park, restore or checkpoint
	Tier     string // live, checkpoint or parked
	FirstTok int64
	LastTok  int64
	Bytes    int64
	MS       float64
	At       time.Time // when the server read the line

	// The cache's state after the event, when the engine reports it (park and restore lines carry
	// "parked=<n> parked_bytes=<n>"); HasParked is false when it does not.
	HasParked   bool
	Parked      int64
	ParkedBytes int64
}

// ParseKVEvent reads one engine line. ok is false for anything that is not a well-formed v1 event.
func ParseKVEvent(line string, at time.Time) (KVEvent, bool) {
	f := strings.Fields(line)
	if len(f) < 3 || f[0] != "KVEV" || f[1] != "v1" {
		return KVEvent{}, false
	}
	ev := KVEvent{Event: f[2], At: at}
	switch ev.Event {
	case "park", "restore", "checkpoint":
	default:
		return KVEvent{}, false
	}
	var haveTier, haveTokens, haveBytes, haveMS, haveParked, haveParkedBytes bool
	for _, kv := range f[3:] {
		k, v, found := strings.Cut(kv, "=")
		if !found {
			continue
		}
		switch k {
		case "tier":
			switch v {
			case "live", "checkpoint", "parked":
				ev.Tier, haveTier = v, true
			default:
				return KVEvent{}, false
			}
		case "tokens":
			a, b, found := strings.Cut(v, "-")
			first, err1 := strconv.ParseInt(a, 10, 64)
			last, err2 := strconv.ParseInt(b, 10, 64)
			if !found || err1 != nil || err2 != nil || first < 0 || last < first {
				return KVEvent{}, false
			}
			ev.FirstTok, ev.LastTok, haveTokens = first, last, true
		case "bytes":
			n, err := strconv.ParseInt(v, 10, 64)
			if err != nil || n < 0 {
				return KVEvent{}, false
			}
			ev.Bytes, haveBytes = n, true
		case "parked", "parked_bytes": // optional: a malformed one is ignored, never turned into a count
			if n, err := strconv.ParseInt(v, 10, 64); err == nil && n >= 0 {
				if k == "parked" {
					ev.Parked, haveParked = n, true
				} else {
					ev.ParkedBytes, haveParkedBytes = n, true
				}
			}
		case "ms":
			x, err := strconv.ParseFloat(v, 64)
			if err != nil || math.IsNaN(x) || math.IsInf(x, 0) || x < 0 {
				return KVEvent{}, false
			}
			ev.MS, haveMS = x, true
		}
	}
	if !haveTier || !haveTokens || !haveBytes || !haveMS {
		return KVEvent{}, false
	}
	ev.HasParked = haveParked && haveParkedBytes
	if !ev.HasParked {
		ev.Parked, ev.ParkedBytes = 0, 0
	}
	return ev, true
}

// KVTotals counts every event seen since the engine started (not only the retained ones).
type KVTotals struct {
	Parks, Restores, Checkpoints int
	LastEvent                    *KVEvent
	// The parked conversations and their bytes after the latest event that reported them.
	HasParked           bool
	Parked, ParkedBytes int64
}

// KVLedger keeps the most recent events and running totals. The zero value is ready to use.
type KVLedger struct {
	mu     sync.Mutex
	events []KVEvent
	tot    KVTotals
}

// Add records one event.
func (l *KVLedger) Add(ev KVEvent) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.events = append(l.events, ev)
	if len(l.events) > kvLedgerKeep {
		l.events = append([]KVEvent(nil), l.events[len(l.events)-kvLedgerKeep:]...)
	}
	switch ev.Event {
	case "park":
		l.tot.Parks++
	case "restore":
		l.tot.Restores++
	case "checkpoint":
		l.tot.Checkpoints++
	}
	if ev.HasParked {
		l.tot.HasParked, l.tot.Parked, l.tot.ParkedBytes = true, ev.Parked, ev.ParkedBytes
	}
	last := ev
	l.tot.LastEvent = &last
}

// Events returns a copy of the retained events, oldest first.
func (l *KVLedger) Events() []KVEvent {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]KVEvent(nil), l.events...)
}

// Totals returns the running counts and the last event.
func (l *KVLedger) Totals() KVTotals {
	l.mu.Lock()
	defer l.mu.Unlock()
	t := l.tot
	if t.LastEvent != nil {
		last := *t.LastEvent
		t.LastEvent = &last
	}
	return t
}

// noteStderrLine keeps the line in the log tail (for error messages) and records it when it is a KVEV event.
func (eo *EngineOrchestrator) noteStderrLine(line string, at time.Time) {
	eo.mu.Lock()
	eo.logTail = append(eo.logTail, line)
	if len(eo.logTail) > 30 {
		eo.logTail = eo.logTail[len(eo.logTail)-30:]
	}
	eo.mu.Unlock()
	if ev, ok := ParseKVEvent(line, at); ok {
		eo.kv.Add(ev)
	}
}

// KVLedger is the engine's measured cache events.
func (eo *EngineOrchestrator) KVLedger() *KVLedger { return &eo.kv }

func (eo *EngineOrchestrator) logTailLen() int {
	eo.mu.RLock()
	defer eo.mu.RUnlock()
	return len(eo.logTail)
}
