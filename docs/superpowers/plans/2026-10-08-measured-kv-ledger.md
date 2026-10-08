# Measured KV Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show, per request, what a chat switch really costs in KV restore ms, prefill ms and tokens reused (with an estimate of prefill time saved), from measured engine events, and stop presenting the page table's invented numbers as measurements.

**Architecture:** The engine prints one additive `KVEV v1 ...` line per cache event on stderr. The Go orchestrator already pumps engine stderr (`pumpStderr`); it parses those lines into a bounded in-memory ledger. The `/metrics` handler joins ledger events to each request's time window and adds `restore_ms`, `prefill_ms`, `saved_ms`, plus ledger totals under `conversation_cache`. The runtime's `virtual_kv` metrics get a `"source": "modeled"` label.

**Tech Stack:** Go (`pkg/engineipc`, `pkg/server`), C++/CUDA engine (`src/program/generate.cpp`), C++ runtime (`src/rt/rt_server.cpp`), Python test harness (`tools/parking_test.py`).

**Spec:** `docs/superpowers/specs/2026-10-08-measured-kv-ledger-design.md`

## Global Constraints

- Cache behavior is unchanged: no change to what the engine caches, evicts, restores or parks.
- The new engine line is exactly `KVEV v1 <event> tier=<live|checkpoint|parked> tokens=<first>-<last> bytes=<n> ms=<float>`; `event` is `park`, `restore` or `checkpoint`.
- A missing value is `null` in JSON, never `0`. Unknown or malformed `KVEV` lines are ignored; parsing never blocks the request path.
- Prefill ms and saved ms are estimates and are labeled `estimate` in the docs and the JSON keys' comments.
- Docs style: plain words, measured numbers with the machine they were measured on, no claim without a measurement.
- The working tree already has many uncommitted changes by the user (including `pkg/engineipc/orchestrator.go`, `pkg/server/server.go`, `pkg/server/monitor.go`). New logic goes in NEW files. Commit steps stage only the new files, and for edited existing files use `git add -p` to stage only this plan's hunks. Never `git add -A` or `git commit -a`.
- Commit messages end with: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`

## Review Focus

- A `KVEV` line with a newer version (`v2`) or extra unknown `key=value` pairs: ignored (v2) or tolerated (extra keys on v1). Pinned in Task 1.
- A line with non-numeric `bytes=`/`ms=` or a reversed token range: ignored, never a zero-valued event. Pinned in Task 1.
- Hundreds of events with no requests (idle engine): the ledger stays bounded at 256 events. Pinned in Task 1.
- A request that overlaps no restore event (fresh prompt, or cache off): `restore_ms` is `null`, `saved_ms` is `null`, not `0`. Pinned in Task 3.
- A request that read zero tokens (full reuse) or has no `prompt_ms`: no division by zero; `saved_ms` is `null`. Pinned in Task 3.
- The annotated requests must not mutate the monitor's stored maps (data race with the monitor). Pinned in Task 3.

---

### Task 1: KV event parser and ledger

**Files:**
- Create: `pkg/engineipc/kvledger.go`
- Test: `pkg/engineipc/kvledger_test.go`

**Interfaces:**
- Produces:
  - `type KVEvent struct { Event, Tier string; FirstTok, LastTok, Bytes int64; MS float64; At time.Time }`
  - `func ParseKVEvent(line string, at time.Time) (KVEvent, bool)`
  - `type KVLedger struct{ ... }` with zero value ready to use
  - `func (l *KVLedger) Add(ev KVEvent)`
  - `func (l *KVLedger) Events() []KVEvent` (copy, oldest first)
  - `type KVTotals struct { Parks, Restores, Checkpoints int; LastEvent *KVEvent }`
  - `func (l *KVLedger) Totals() KVTotals`

- [ ] **Step 1: Write the failing tests**

```go
// pkg/engineipc/kvledger_test.go
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
		"KVEV v2 restore tier=live tokens=0-5 bytes=1 ms=1",       // newer version
		"KVEV v1 explode tier=live tokens=0-5 bytes=1 ms=1",       // unknown event
		"KVEV v1 restore tier=live tokens=0-5 bytes=abc ms=1",     // non-numeric
		"KVEV v1 restore tier=live tokens=9-5 bytes=1 ms=1",       // reversed range
		"KVEV v1 restore tier=live tokens=0-5 bytes=1",            // missing ms
		"KVEV v1 restore tier=live tokens=0-5 bytes=-1 ms=1",      // negative bytes
		"KVEV v1 restore tier=disk tokens=0-5 bytes=1 ms=1",       // unknown tier
		"KVEV v1 restore tier=live tokens=0-5 bytes=1 ms=NaN",     // not a number we can show
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
	tot := l.Totals() // totals count every event ever added, not just the retained ones
	if tot.Parks != 200 || tot.Restores != 400 || tot.LastEvent == nil || tot.LastEvent.LastTok != 599 {
		t.Errorf("totals: %+v", tot)
	}
	evs[0].Event = "changed" // the copy must not alias the ledger
	if l.Events()[0].Event == "changed" {
		t.Error("Events() aliases internal storage")
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./pkg/engineipc -run 'KV' -v`
Expected: FAIL to compile (`undefined: ParseKVEvent`, `KVLedger`).

- [ ] **Step 3: Write the implementation**

```go
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
	var haveTier, haveTokens, haveBytes, haveMS bool
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
	return ev, true
}

// KVTotals counts every event seen since the engine started (not only the retained ones).
type KVTotals struct {
	Parks, Restores, Checkpoints int
	LastEvent                    *KVEvent
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `go test ./pkg/engineipc -run 'KV' -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit (new files only)**

```bash
git add pkg/engineipc/kvledger.go pkg/engineipc/kvledger_test.go
git commit -m "feat(engineipc): parse the engine's KVEV cache events into a bounded ledger" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Feed the ledger from the engine's stderr

**Files:**
- Modify: `pkg/engineipc/orchestrator.go` (struct `EngineOrchestrator` near line 245; `pumpStderr` near line 388)
- Test: `pkg/engineipc/kvledger_test.go` (append)

**Interfaces:**
- Consumes: `ParseKVEvent`, `KVLedger` from Task 1.
- Produces: `func (eo *EngineOrchestrator) KVLedger() *KVLedger`, and a package-level `func (eo *EngineOrchestrator) noteStderrLine(line string, at time.Time)` that `pumpStderr` calls for every line.

- [ ] **Step 1: Write the failing test** (append to `kvledger_test.go`)

```go
func TestNoteStderrLineFeedsLedger(t *testing.T) {
	eo := &EngineOrchestrator{}
	eo.noteStderrLine("strata serve: conversation cache: restored 5 tokens (live) in 1.0 ms", t0) // free text: ignored
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./pkg/engineipc -run NoteStderr -v`
Expected: FAIL to compile (`noteStderrLine`, `KVLedger`, `logTailLen` undefined).

- [ ] **Step 3: Implement**

In `orchestrator.go`, add a field to `EngineOrchestrator` (after `logTail []string`):

```go
	kv         KVLedger // the engine's KVEV cache events (see kvledger.go)
```

Replace the body of the `for sc.Scan()` loop in `pumpStderr` with a call to a new method, and add the methods (put the methods in `kvledger.go` to keep `orchestrator.go` edits small):

```go
// in orchestrator.go, pumpStderr loop body becomes:
	for sc.Scan() {
		line := sc.Text()
		fmt.Fprintln(os.Stderr, line)
		eo.noteStderrLine(line, time.Now())
	}
```

```go
// appended to kvledger.go
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
```

If `time` is not yet imported in `orchestrator.go`, it already is (it uses `time`); verify with `go build ./pkg/engineipc`.

- [ ] **Step 4: Run to verify it passes**

Run: `go test ./pkg/engineipc -v 2>&1 | tail -20`
Expected: PASS, including the pre-existing engineipc tests.

- [ ] **Step 5: Commit (hunks only)**

```bash
git add pkg/engineipc/kvledger.go pkg/engineipc/kvledger_test.go
git add -p pkg/engineipc/orchestrator.go   # stage ONLY the kv field and the pumpStderr loop body
git commit -m "feat(engineipc): record KVEV lines from the engine's stderr" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Cost split in `/metrics`

**Files:**
- Create: `pkg/server/kvcost.go`
- Test: `pkg/server/kvcost_test.go`
- Modify: `pkg/server/server.go` (the metrics handler near line 545-560), `pkg/server/enginemetrics.go` (`conversationCacheView`)

**Interfaces:**
- Consumes: `engineipc.KVEvent`, `engineipc.KVTotals`.
- Produces:
  - `func annotateKVCost(requests []map[string]interface{}, events []engineipc.KVEvent) []map[string]interface{}` returns NEW maps (the input is not mutated) each with `restore_ms`, `prefill_ms`, `saved_ms` (each a `float64` or `nil`).
  - `func kvLedgerView(t engineipc.KVTotals) map[string]interface{}` returns `{"parks","restores","checkpoints","last_event"}`.

Request maps (from `monitor.go`) carry: `"time"` (float64 unix seconds of start), `"prompt_tokens"` (int), `"reused"` (int or nil), `"prompt_ms"` (float64 or nil). `prompt_ms` runs from request start to first token, so it includes any restore.

- [ ] **Step 1: Write the failing tests**

```go
// pkg/server/kvcost_test.go
package server

import (
	"testing"
	"time"

	"strata/pkg/engineipc"
)

func req(start float64, promptTokens, reused int, promptMS interface{}) map[string]interface{} {
	var r interface{}
	if reused >= 0 {
		r = reused
	}
	return map[string]interface{}{"time": start, "prompt_tokens": promptTokens, "reused": r, "prompt_ms": promptMS}
}

func at(sec float64) time.Time { return time.Unix(0, int64(sec*1e9)) }

func TestAnnotateKVCostSplit(t *testing.T) {
	// request starts at t=100 s, first token 1.0 s later (prompt_ms 1000): 50 ms restore, 950 ms reading 800 of 1000 tokens
	reqs := []map[string]interface{}{req(100, 1000, 200, 1000.0)}
	evs := []engineipc.KVEvent{{Event: "restore", MS: 50, At: at(100.05)}}
	out := annotateKVCost(reqs, evs)
	if out[0]["restore_ms"] != 50.0 {
		t.Errorf("restore_ms = %v", out[0]["restore_ms"])
	}
	if out[0]["prefill_ms"] != 950.0 {
		t.Errorf("prefill_ms = %v", out[0]["prefill_ms"])
	}
	// 950 ms for 800 read tokens = 1.1875 ms/token; 200 reused tokens would have cost 237.5 ms
	if got := out[0]["saved_ms"].(float64); got < 237.4 || got > 237.6 {
		t.Errorf("saved_ms = %v", got)
	}
	if _, mutated := reqs[0]["restore_ms"]; mutated {
		t.Error("the input maps must not be mutated")
	}
}

func TestAnnotateKVCostNullsNotZeros(t *testing.T) {
	cases := map[string]map[string]interface{}{
		"no events":       req(100, 500, 0, 400.0),
		"no prompt_ms":    req(100, 500, 0, nil),
		"nothing read":    req(100, 500, 500, 30.0),
		"reused unknown":  req(100, 500, -1, 400.0),
		"zero prompt ms":  req(100, 500, 100, 0.0),
	}
	for name, r := range cases {
		out := annotateKVCost([]map[string]interface{}{r}, nil)
		if out[0]["restore_ms"] != nil || out[0]["saved_ms"] != nil {
			t.Errorf("%s: restore_ms=%v saved_ms=%v, want nil/nil", name, out[0]["restore_ms"], out[0]["saved_ms"])
		}
	}
}

func TestAnnotateKVCostIgnoresOtherWindows(t *testing.T) {
	reqs := []map[string]interface{}{req(100, 1000, 200, 1000.0)}
	evs := []engineipc.KVEvent{
		{Event: "restore", MS: 70, At: at(50)},    // before the request
		{Event: "restore", MS: 80, At: at(105)},   // after its first token
		{Event: "park", MS: 90, At: at(100.5)},    // a park is not a restore
		{Event: "restore", MS: 10, At: at(100.9)}, // inside
	}
	out := annotateKVCost(reqs, evs)
	if out[0]["restore_ms"] != 10.0 {
		t.Errorf("restore_ms = %v, want only the event inside the window", out[0]["restore_ms"])
	}
}

func TestKVLedgerViewEmpty(t *testing.T) {
	v := kvLedgerView(engineipc.KVTotals{})
	if v["last_event"] != nil || v["restores"] != 0 {
		t.Errorf("empty totals: %v", v)
	}
}
```

Check the module path first: `head -1 go.mod` and replace `strata/pkg/engineipc` in the import with the real module path.

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./pkg/server -run 'KVCost|KVLedgerView' -v`
Expected: FAIL to compile (`annotateKVCost` undefined).

- [ ] **Step 3: Implement**

```go
// pkg/server/kvcost.go - the three costs of a chat switch, from the engine's measured KV events.
package server

import "strata/pkg/engineipc"

// annotateKVCost returns copies of the request records with:
//
//	restore_ms  measured: the engine's KV restore events inside the request's start..first-token window
//	prefill_ms  estimate: prompt_ms minus restore_ms (time spent reading the prompt tokens that were not reused)
//	saved_ms    estimate: reused tokens priced at this request's own prefill rate
//
// A value that cannot be known is nil, never 0. The input maps (the monitor's) are not modified.
func annotateKVCost(requests []map[string]interface{}, events []engineipc.KVEvent) []map[string]interface{} {
	out := make([]map[string]interface{}, len(requests))
	for i, r := range requests {
		c := make(map[string]interface{}, len(r)+3)
		for k, v := range r {
			c[k] = v
		}
		c["restore_ms"], c["prefill_ms"], c["saved_ms"] = nil, nil, nil
		out[i] = c

		start, okStart := r["time"].(float64)
		promptMS, okMS := r["prompt_ms"].(float64)
		if !okStart || !okMS || promptMS <= 0 {
			continue
		}
		restore, seen := 0.0, false
		endSec := start + promptMS/1000
		for _, ev := range events {
			if ev.Event != "restore" {
				continue
			}
			t := float64(ev.At.UnixNano()) / 1e9
			if t >= start && t <= endSec {
				restore += ev.MS
				seen = true
			}
		}
		if seen {
			c["restore_ms"] = restore
		}
		prefill := promptMS - restore
		total, read := asInt(r["prompt_tokens"]), 0
		reused, haveReused := r["reused"].(int)
		if haveReused {
			read = total - reused
		}
		if seen || (haveReused && read > 0) {
			if prefill < 0 {
				prefill = 0
			}
			c["prefill_ms"] = prefill
		}
		if haveReused && read > 0 && reused > 0 && prefill > 0 {
			c["saved_ms"] = float64(reused) * prefill / float64(read)
		}
	}
	return out
}

// kvLedgerView is the ledger's totals for /metrics. last_event is nil until the engine has reported one.
func kvLedgerView(t engineipc.KVTotals) map[string]interface{} {
	v := map[string]interface{}{"parks": t.Parks, "restores": t.Restores, "checkpoints": t.Checkpoints, "last_event": nil}
	if e := t.LastEvent; e != nil {
		v["last_event"] = map[string]interface{}{
			"event": e.Event, "tier": e.Tier, "tokens": e.LastTok - e.FirstTok, "bytes": e.Bytes, "ms": e.MS,
			"at": float64(e.At.UnixNano()) / 1e9,
		}
	}
	return v
}
```

Wire it into the handler in `server.go` (after `requests, kept, totals := s.mon.view(...)` and the `engInfo` block):

```go
	var kvTotals engineipc.KVTotals
	if s.EngineIPC != nil {
		requests = annotateKVCost(requests, s.EngineIPC.KVLedger().Events())
		kvTotals = s.EngineIPC.KVLedger().Totals()
	}
```

and in the response map change `conversationCacheView(engInfo, requests, totals)` to add the ledger after it is built:

```go
	cc := conversationCacheView(engInfo, requests, totals)
	cc["ledger"] = kvLedgerView(kvTotals)
```

then use `"conversation_cache": cc`. Also fill the card's `parks`/`restores`/`last_event` placeholders that `conversationCacheView` currently leaves at `0`/`nil`: set them from `kvTotals` inside the handler (`cc["parks"] = kvTotals.Parks`, `cc["restores"] = kvTotals.Restores`) only when `s.EngineIPC != nil`.

- [ ] **Step 4: Run to verify it passes**

Run: `go test ./pkg/server -run 'KVCost|KVLedgerView|ConversationCacheView' -v && go build ./... && go vet ./pkg/server ./pkg/engineipc`
Expected: PASS, build and vet clean.

- [ ] **Step 5: Commit (new files + hunks)**

```bash
git add pkg/server/kvcost.go pkg/server/kvcost_test.go
git add -p pkg/server/server.go   # only the kvTotals/annotateKVCost/cc hunks
git commit -m "feat(server): per-request KV restore/prefill/saved split and ledger totals in /metrics" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Engine prints the KVEV lines

**Files:**
- Modify: `src/program/generate.cpp`: the park print (~line 6491), the restore print (~line 8387), and the checkpoint save (`checkpoint_save`, ~line 1321 and its callers' token count).
- Test: `tools/parking_test.py` (extend)

**Interfaces:**
- Produces: stderr lines in the format of Global Constraints, which Task 1 parses.

The engine needs a CUDA build to verify (`ninja -C build` per `HANDOFF.md`). Parked entries and restores already compute `live.size()`, `snapshot_bytes`, the elapsed ms and (for restore) `resume`, `from_live`; reuse those values, add nothing new to the cache path.

- [ ] **Step 1: Write the failing test** (extend `tools/parking_test.py` after the existing `restored = re.findall(...)` line in `run()`)

```python
    kvev = re.findall(r"^KVEV v1 (park|restore|checkpoint) tier=(live|checkpoint|parked) tokens=(\d+)-(\d+) bytes=(\d+) ms=([\d.]+)$",
                      log, re.M)
    if with_b:
        kinds = {k[0] for k in kvev}
        assert {"park", "restore"} <= kinds, f"no KVEV park/restore line in the engine log (saw {kinds})"
        for ev, tier, a, b, nbytes, ms in kvev:
            assert int(b) >= int(a) and int(nbytes) > 0 and float(ms) >= 0, f"implausible KVEV line: {ev} {a}-{b} {nbytes} {ms}"
        print(f"KVEV lines: {len(kvev)} ({', '.join(sorted(kinds))})", flush=True)
```

- [ ] **Step 2: Run to verify it fails**

Run (needs a model and GPU; use the command in the file's docstring): `python3 tools/parking_test.py --exe engine/strata --config <config> --extra "--conversation-cache-mib 8192 --conversation-cache-slots 4 --pcie-frac 0"`
Expected: FAIL with `no KVEV park/restore line in the engine log`.

- [ ] **Step 3: Implement**

Park, directly after the existing `std::fprintf(stderr, "strata serve: conversation cache: %s %zu tokens ...` call, inside the same block, only when `stored`:

```cpp
                if (stored)
                    std::fprintf(stderr, "KVEV v1 park tier=parked tokens=0-%zu bytes=%zu ms=%.1f\n", live.size(), snapshot_bytes,
                                 std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
```

Restore, directly after the existing `restored %lld tokens (%s) in %.1f ms` print. Use the conversation cache's byte count that the restore knows; if the snapshot's byte size is not in scope there, capture `const size_t restored_bytes = incoming->bytes();` before `incoming.reset()` (same block, a few lines above) and print it:

```cpp
                std::fprintf(stderr, "KVEV v1 restore tier=%s tokens=0-%lld bytes=%zu ms=%.1f\n",
                             from_live ? "live" : "checkpoint", (long long) resume, restored_bytes,
                             std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
```

Checkpoint: in `checkpoint_save` callers' lambda `checkpoint_at` (~line 6562), time the `checkpoint_save(c, ss, g)` call and print on success:

```cpp
                const Clock::time_point ck0 = Clock::now();
                if (!gpu_sync_ok() || !checkpoint_save(c, ss, g)) return false;
                std::fprintf(stderr, "KVEV v1 checkpoint tier=checkpoint tokens=0-%lld bytes=%zu ms=%.1f\n", (long long) done, c.bytes(),
                             std::chrono::duration<double, std::milli>(Clock::now() - ck0).count());
```

Read the surrounding lines first and keep the variable names the code actually uses (`done`, `c`); if `done` is named differently there, use the token position the checkpoint is taken at. Do not touch the multi-stage `part` branch beyond the same one-line print per part, and do not print when the save fails.

- [ ] **Step 4: Build and run**

Run: `ninja -C build strata 2>&1 | tail -5` (the target name may differ; use the one `HANDOFF.md` lists), then the `parking_test.py` command from Step 2.
Expected: PASS, prints `KVEV lines: N (checkpoint, park, restore)`, and the existing exactness check (follow-up decodes identical tokens) still passes. If no GPU or model is available, say so in the report and mark this task unverified.

- [ ] **Step 5: Commit (hunks only)**

```bash
git add -p src/program/generate.cpp tools/parking_test.py   # only the three KVEV prints and the test addition
git commit -m "feat(engine): print measured KVEV cache events (park, restore, checkpoint)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Label the page table's metrics as modeled

**Files:**
- Modify: `src/rt/rt_server.cpp` (`kv_metrics_json`, line 224)
- Test: `pkg/server/rt_test.go` (extend `TestRuntimeMetricsReportsVirtualKV`, near line 299)

- [ ] **Step 1: Write the failing test.** In the existing test, after the `total_pages` check, add:

```go
	if res.Metrics.VirtualKV["source"] != "modeled" {
		t.Errorf("virtual_kv must say it is modeled, got source=%v", res.Metrics.VirtualKV["source"])
	}
```

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./pkg/server -run TestRuntimeMetricsReportsVirtualKV -v`
Expected: FAIL (`source=<nil>`). This test starts the runtime server; if it needs `build/strata_rt_server`, run `ninja -C build strata_rt_server` first.

- [ ] **Step 3: Implement.** In `kv_metrics_json`, after `j.set("page_faults", ...)`:

```cpp
    // The runtime's pages describe logical context pages with estimated sizes and costs, not the engine's real
    // KV; the engine's measured numbers are under conversation_cache.ledger in the server's /metrics.
    j.set("source", Json::string("modeled"));
```

- [ ] **Step 4: Run to verify it passes**

Run: `ninja -C build strata_rt_server test_kv_page_table && ./build/test_kv_page_table && go test ./pkg/server -run TestRuntimeMetricsReportsVirtualKV -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -p src/rt/rt_server.cpp pkg/server/rt_test.go   # only this task's hunks
git commit -m "fix(rt): label virtual_kv metrics as modeled, not measured" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Show it in the Monitor

**Files:**
- Modify: `pkg/frontend/web/app.js`, `pkg/frontend/web/index.html` (the Conversation cache card; find it with `grep -n -i "conversation cache\|conversation_cache" pkg/frontend/web/app.js pkg/frontend/web/index.html`)

- [ ] **Step 1:** Locate the card's render code. Add three rows to the per-request line and the card: last restore (`conversation_cache.ledger.last_event`), and for the newest request `restore_ms`, `prefill_ms`, `saved_ms`, each rendered as `n/a` when `null` (never `0`) and with `~` before the two estimates (`prefill_ms`, `saved_ms`).
- [ ] **Step 2:** Run `./bin/strata serve` against a model, send two alternating conversations (the Step 2 scenario of Task 7), and open `http://127.0.0.1:8080#monitor`. Confirm the numbers match `curl -s 127.0.0.1:8080/metrics | python3 -m json.tool` (`conversation_cache.ledger`, `requests[0].restore_ms`).
- [ ] **Step 3:** Commit with `git add -p pkg/frontend/web/app.js pkg/frontend/web/index.html`.

If no model or GPU is available, report this task as unverified in the browser; the JSON from Task 3 is still verified by its tests.

---

### Task 7: Measure and document

**Files:**
- Modify: `docs/DETAILS.md` (the Conversation cache section, after the Monitor-card paragraph near line 743)

- [ ] **Step 1: Two-agent scenario.** With the cache on (`--conversation-cache-mib 8192 --conversation-cache-slots 4`), alternate two conversations A and B of at least 8,000 tokens each, six switches. For each request record `restore_ms`, `prefill_ms`, `saved_ms`, `reused`, `prompt_tokens` from `/metrics`.
- [ ] **Step 2: Long single chat.** One conversation grown over 10 turns past 16K tokens; record the same fields per turn.
- [ ] **Step 3: Write the section.** Plain words, one small table per scenario, the machine (CPU, GPU, RAM), model, engine flags and engine version, "single runs", and an explicit "not measured" list: prefill-rate estimate assumes a uniform rate per token (attention cost grows with length), concurrency, NVMe session-file restore, the layer-split case unless it was run.
- [ ] **Step 4: State which of the follow-up projects (cost-aware eviction + NVMe spill, mid-prompt shared checkpoints, prefetch) the numbers support**, in two sentences, and nothing more.
- [ ] **Step 5: Commit** `docs/DETAILS.md` with `git add -p`.

---

## Self-review

- **Spec coverage:** goal 1 (three costs per request in Monitor and `/metrics`) → Tasks 3, 6; goal 2 (page table honest) → Task 5, chosen branch is the "label as modeled" option, and feeding the table real data is left out because the table and engine are different processes; goal 3 (measured run in docs) → Task 7; goal 4 (no behavior change) → engine edits are print-only (Task 4). The spec's `reuse` event is not a separate line: reuse is already in `RESUME`/`REUSED`/`DONE` and the request record's `reused`, so `restore`, `park` and `checkpoint` are the three new events.
- **Placeholder scan:** none; Tasks 4 and 6 name exact insertion points and say what to do if a variable name differs.
- **Type consistency:** `KVEvent`, `KVLedger`, `KVTotals`, `ParseKVEvent`, `annotateKVCost`, `kvLedgerView`, `noteStderrLine`, `KVLedger()` are used with the same signatures in every task.
- **Review Focus:** each line is pinned in a test in Tasks 1 and 3.
