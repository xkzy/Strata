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
	// a cold read in the same list sets the price: 1000 tokens in 2000 ms = 2 ms/token
	reqs := []map[string]interface{}{req(100, 1000, 200, 1000.0), req(50, 1000, -1, 2000.0)}
	evs := []engineipc.KVEvent{{Event: "restore", MS: 50, At: at(100.05)}}
	out := annotateKVCost(reqs, evs)
	if out[0]["restore_ms"] != 50.0 {
		t.Errorf("restore_ms = %v", out[0]["restore_ms"])
	}
	if out[0]["prefill_ms"] != 950.0 {
		t.Errorf("prefill_ms = %v", out[0]["prefill_ms"])
	}
	if got := out[0]["saved_ms"].(float64); got != 400.0 { // 200 reused tokens at the cold 2 ms/token
		t.Errorf("saved_ms = %v, want 400", got)
	}
	if _, mutated := reqs[0]["restore_ms"]; mutated {
		t.Error("the input maps must not be mutated")
	}
}

func TestAnnotateKVCostNullsNotZeros(t *testing.T) {
	cases := map[string]map[string]interface{}{
		"no events":      req(100, 500, 0, 400.0),
		"no prompt_ms":   req(100, 500, 0, nil),
		"nothing read":   req(100, 500, 500, 30.0),
		"reused unknown": req(100, 500, -1, 400.0),
		"zero prompt ms": req(100, 500, 100, 0.0),
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
		{Event: "restore", MS: 70, At: at(50)},
		{Event: "restore", MS: 80, At: at(105)},
		{Event: "park", MS: 90, At: at(100.5)},
		{Event: "restore", MS: 10, At: at(100.9)},
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

func TestAnnotateKVCostReusedFromRestoreEvent(t *testing.T) {
	// the monitor leaves reused nil; the engine's restore event says 200 tokens came back
	reqs := []map[string]interface{}{req(100, 1000, -1, 1000.0), req(50, 1000, -1, 2000.0)}
	evs := []engineipc.KVEvent{{Event: "restore", MS: 50, LastTok: 200, At: at(100.05)}}
	out := annotateKVCost(reqs, evs)
	if got, ok := out[0]["saved_ms"].(float64); !ok || got != 400.0 {
		t.Errorf("saved_ms = %v, want 400 from the restore event's 200 tokens", out[0]["saved_ms"])
	}
}

// The numbers a live run produced (RTX 4060, Coder IQ1_M): a restore request that read ~5 tokens took 709 ms in all;
// its own rate would call 3,075 reused tokens 140 s of saving. The cold reads in the same run priced a token at ~2.5 ms.
func TestAnnotateKVCostLiveRunRegression(t *testing.T) {
	reqs := []map[string]interface{}{
		req(300, 3080, -1, 709.4),  // restored 3075 tokens in 22.6 ms
		req(200, 3060, -1, 7592.7), // cold
		req(100, 1552, -1, 4154.6), // cold
	}
	evs := []engineipc.KVEvent{{Event: "restore", MS: 22.6, LastTok: 3075, At: at(300.1)}}
	out := annotateKVCost(reqs, evs)
	got, ok := out[0]["saved_ms"].(float64)
	if !ok || got < 7000 || got > 9000 {
		t.Errorf("saved_ms = %v, want ~7.6 s (3075 tokens at ~2.5 ms), not minutes", out[0]["saved_ms"])
	}
}

func TestAnnotateKVCostNoColdReadNoSavedEstimate(t *testing.T) {
	reqs := []map[string]interface{}{req(100, 1000, -1, 1000.0)}
	evs := []engineipc.KVEvent{{Event: "restore", MS: 50, LastTok: 200, At: at(100.05)}}
	if out := annotateKVCost(reqs, evs); out[0]["saved_ms"] != nil {
		t.Errorf("saved_ms = %v, want nil when no cold read prices a token", out[0]["saved_ms"])
	}
}

func TestAnnotateKVCostFillsReusedFromRestore(t *testing.T) {
	reqs := []map[string]interface{}{req(100, 1000, -1, 1000.0), req(50, 1000, -1, 2000.0)}
	evs := []engineipc.KVEvent{{Event: "restore", MS: 50, LastTok: 200, At: at(100.05)}}
	out := annotateKVCost(reqs, evs)
	if out[0]["reused"] != 200 {
		t.Errorf("reused = %v, want 200 from the restore event", out[0]["reused"])
	}
	if out[1]["reused"] != nil {
		t.Errorf("a cold read must stay nil, got %v", out[1]["reused"])
	}
	if reqs[0]["reused"] != nil {
		t.Error("the input must not be mutated")
	}
}
