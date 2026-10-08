// pkg/server/kvcost.go - the three costs of a chat switch, from the engine's measured KV events.
package server

import "strata/pkg/engineipc"

// annotateKVCost returns copies of the request records with:
//
//	restore_ms  measured: the engine's KV restore events inside the request's start..first-token window
//	prefill_ms  estimate: prompt_ms minus restore_ms (time spent reading the prompt tokens that were not reused)
//	saved_ms    estimate: reused tokens priced at the cold-read rate (see coldReadMSPerToken); nil without one
//
// Reused tokens come from the record's "reused" when it has one, else from the largest restored token count of the
// window's restore events. A value that cannot be known is nil, never 0. The input maps (the monitor's) are not
// modified.
func annotateKVCost(requests []map[string]interface{}, events []engineipc.KVEvent) []map[string]interface{} {
	out := make([]map[string]interface{}, len(requests))
	coldRate := coldReadMSPerToken(requests, events)
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
		var restoredTokens int64
		endSec := start + promptMS/1000
		for _, ev := range events {
			if ev.Event != "restore" {
				continue
			}
			t := float64(ev.At.UnixNano()) / 1e9
			if t >= start && t <= endSec {
				restore += ev.MS
				seen = true
				if ev.LastTok > restoredTokens {
					restoredTokens = ev.LastTok
				}
			}
		}
		if seen {
			c["restore_ms"] = restore
		}
		total := asInt(r["prompt_tokens"])
		reused, haveReused := 0, false
		if r["reused"] != nil {
			reused, haveReused = asInt(r["reused"]), true
		} else if seen {
			reused, haveReused = int(restoredTokens), true
			c["reused"] = reused // the monitor records none; the engine's restore says how much came back
		}
		read := total - reused
		if !haveReused || read <= 0 {
			continue
		}
		prefill := promptMS - restore
		if prefill < 0 {
			prefill = 0
		}
		c["prefill_ms"] = prefill
		if reused > 0 && coldRate > 0 {
			c["saved_ms"] = float64(reused) * coldRate
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

// coldMinTokens is the shortest prompt whose time is mostly reading: a shorter one is mostly fixed overhead.
const coldMinTokens = 256

// coldReadMSPerToken prices one prompt token read from scratch: the highest time-per-token among the requests that
// restored nothing and had no reuse recorded (a cold read). The request being priced cannot price itself: one that
// read only a few tokens takes mostly fixed first-token time, and its own rate made a 0.7 s request "save" 140 s in
// a live run. The highest, not the average, because a live continuation that left no event also looks cold and is
// far cheaper; so the estimate errs low, except when one cold read was slow for another reason. 0 means no cold read.
func coldReadMSPerToken(requests []map[string]interface{}, events []engineipc.KVEvent) float64 {
	best := 0.0
	for _, r := range requests {
		start, okStart := r["time"].(float64)
		promptMS, okMS := r["prompt_ms"].(float64)
		total := asInt(r["prompt_tokens"])
		if !okStart || !okMS || promptMS <= 0 || r["reused"] != nil || total < coldMinTokens {
			continue
		}
		restored := false
		for _, ev := range events {
			t := float64(ev.At.UnixNano()) / 1e9
			if ev.Event == "restore" && t >= start && t <= start+promptMS/1000 {
				restored = true
				break
			}
		}
		if restored {
			continue
		}
		if rate := promptMS / float64(total); rate > best {
			best = rate
		}
	}
	return best
}
