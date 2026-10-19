// pkg/server/monitor.go - what the web app's Monitor tab shows about requests: the one running now, the last
// finished ones and the totals since the server started (the "live", "requests", "requests_kept" and "totals" of
// GET /metrics). Fed from startGeneration, the one place every generation passes through.
package server

import (
	"math"
	"sort"
	"sync"
	"time"
)

const (
	monitorKeep   = 500 // finished requests kept (the page shows the last 12, or all of them on request)
	monitorWindow = 3 * time.Second
)

type requestMonitor struct {
	mu      sync.Mutex
	since   time.Time
	running map[*monitoredRequest]struct{}
	history []map[string]interface{} // newest last
	totals  map[string]interface{}
}

type monitoredRequest struct {
	m            *requestMonitor
	started      time.Time
	firstToken   time.Time
	promptTokens int
	promptRead   int
	maxTokens    int
	generated    int
	stamps       []time.Time // the last few seconds' tokens, for the current rate
	done         bool
	rec          map[string]interface{} // the history record finish wrote
}

func (m *requestMonitor) init() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.initLocked()
}

func (m *requestMonitor) initLocked() {
	if m.running != nil {
		return
	}
	m.since = time.Now()
	m.running = map[*monitoredRequest]struct{}{}
	m.totals = map[string]interface{}{"requests": 0, "prompt_tokens": 0, "reused": 0, "output_tokens": 0,
		"prompt_ms": 0.0, "decode_ms": 0.0}
}

// begin notes a request that has been tokenized and is about to run.
func (m *requestMonitor) begin(promptTokens, maxTokens int) *monitoredRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.initLocked()
	r := &monitoredRequest{m: m, started: time.Now(), promptTokens: promptTokens, maxTokens: maxTokens}
	m.running[r] = struct{}{}
	return r
}

// promptProgress updates the prompt ingestion progress during prefill.
func (r *monitoredRequest) promptProgress(read, total int) {
	r.m.mu.Lock()
	defer r.m.mu.Unlock()
	r.promptRead = read
	if total > 0 {
		r.promptTokens = total
	}
}

// token notes one generated token (the first one ends the prompt's reading).
func (r *monitoredRequest) token() {
	r.m.mu.Lock()
	defer r.m.mu.Unlock()
	now := time.Now()
	if r.firstToken.IsZero() {
		r.firstToken = now
	}
	r.generated++
	r.stamps = append(r.stamps, now)
	for len(r.stamps) > 0 && now.Sub(r.stamps[0]) > monitorWindow {
		r.stamps = r.stamps[1:]
	}
}

// finish records the request; only the first call counts. tokens is what the engine reported (0: the count seen);
// engineTokS is the engine's own decode speed (0: measured here).
func (r *monitoredRequest) finish(finish string, tokens int, engineTokS float64) {
	m := r.m
	m.mu.Lock()
	defer m.mu.Unlock()
	if r.done {
		return
	}
	r.done = true
	delete(m.running, r)
	now := time.Now()
	n := r.generated
	if tokens > 0 {
		n = tokens
	}
	if finish == "tool_calls" {
		finish = "stop"
	}
	rec := map[string]interface{}{
		"time": float64(r.started.UnixNano()) / 1e9, "duration_s": round1(now.Sub(r.started).Seconds()), "finish": finish,
		"prompt_tokens": r.promptTokens, "prompt_total": r.promptTokens, "output_tokens": n, "reused": nil,
		"prompt_ms": nil, "decode_ms": nil, "decode_tok_s": nil, "hit_rate": nil, "projection": nil,
	}
	promptMS, decodeMS := 0.0, 0.0
	if !r.firstToken.IsZero() {
		promptMS = float64(r.firstToken.Sub(r.started).Microseconds()) / 1000
		decodeMS = float64(now.Sub(r.firstToken).Microseconds()) / 1000
		rec["prompt_ms"], rec["decode_ms"] = promptMS, decodeMS
		if engineTokS > 0 {
			rec["decode_tok_s"] = round1(engineTokS)
		} else if n > 1 && decodeMS > 0 {
			rec["decode_tok_s"] = round1(float64(n-1) / (decodeMS / 1000)) // the first token starts the clock
		}
	}
	r.rec = rec
	m.history = append(m.history, rec)
	if len(m.history) > monitorKeep {
		m.history = m.history[len(m.history)-monitorKeep:]
	}
	t := m.totals
	t["requests"] = t["requests"].(int) + 1
	t["prompt_tokens"] = t["prompt_tokens"].(int) + r.promptTokens
	t["output_tokens"] = t["output_tokens"].(int) + n
	t["prompt_ms"] = t["prompt_ms"].(float64) + promptMS
	t["decode_ms"] = t["decode_ms"].(float64) + decodeMS
}

// relabel replaces the finish reason of a request that has ended: the runtime stops reading a reply that hit
// max_tokens, so the engine's own reason is a cancel, while the API answered "length".
func (r *monitoredRequest) relabel(finish string) {
	r.m.mu.Lock()
	defer r.m.mu.Unlock()
	if r.rec == nil {
		return
	}
	if finish == "tool_calls" {
		finish = "stop"
	}
	r.rec["finish"] = finish
}

func round1(v float64) float64 { return float64(int(v*10+0.5)) / 10 }

// live is the "live" block of /metrics: the newest running request, or idle (unloaded: no engine to ask).
func (m *requestMonitor) live(loaded bool) map[string]interface{} {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.initLocked()
	out := map[string]interface{}{"state": "idle", "queued": 0, "running": len(m.running), "phase": nil,
		"prompt_tokens": nil, "generated": nil, "max_tokens": nil, "elapsed_s": nil, "tok_s": nil, "tok_s_mean": nil,
		"prompt_read": nil, "prompt_total": nil, "prefill_tok_s_mean": nil}
	var cur *monitoredRequest
	for r := range m.running {
		if cur == nil || r.started.After(cur.started) {
			cur = r
		}
	}
	if cur == nil {
		if !loaded {
			out["state"] = "unloaded"
		}
		return out
	}
	now := time.Now()
	out["prompt_tokens"], out["max_tokens"], out["generated"] = cur.promptTokens, cur.maxTokens, cur.generated
	out["elapsed_s"] = round1(now.Sub(cur.started).Seconds())
	if cur.firstToken.IsZero() {
		out["state"] = "reading"
		out["prompt_read"] = cur.promptRead
		out["prompt_total"] = cur.promptTokens
		if v := prefillRate(cur.promptRead, now.Sub(cur.started)); v > 0 {
			out["prefill_tok_s_mean"] = v
		}
		return out
	}
	out["state"] = "generating"
	if v := prefillRate(cur.promptTokens, cur.firstToken.Sub(cur.started)); v > 0 {
		out["prefill_tok_s_mean"] = v
	}
	out["tok_s"], out["tok_s_mean"] = windowRate(cur.stamps, now), meanRate(cur, now)
	return out
}

// overlayEngineState refines an "unloaded" live state with what the engine is doing: loading while it starts,
// error after a failed start. A running request keeps its own state.
func overlayEngineState(live map[string]interface{}, engine string) {
	if live["state"] != "unloaded" {
		return
	}
	switch engine {
	case "starting":
		live["state"] = "loading"
	case "failed":
		live["state"] = "error"
	}
}

// prefillRate is the prompt tokens read per second over the time spent reading them (0: not known yet).
func prefillRate(tokens int, span time.Duration) float64 {
	if tokens <= 0 || span <= 0 {
		return 0
	}
	return round1(float64(tokens) / span.Seconds())
}

// windowRate is the tokens per second over the last few seconds.
func windowRate(stamps []time.Time, now time.Time) float64 {
	var in []time.Time
	for _, s := range stamps {
		if now.Sub(s) <= monitorWindow {
			in = append(in, s)
		}
	}
	if len(in) < 2 {
		return 0
	}
	span := in[len(in)-1].Sub(in[0]).Seconds()
	if span <= 0 {
		return 0
	}
	return round1(float64(len(in)-1) / span)
}

func meanRate(r *monitoredRequest, now time.Time) float64 {
	if r.generated < 2 || r.firstToken.IsZero() {
		return 0
	}
	span := now.Sub(r.firstToken).Seconds()
	if span <= 0 {
		return 0
	}
	return round1(float64(r.generated-1) / span)
}

// currentTokS is the decode speed of the running request for the sparkline (0 when nothing is generating).
func (m *requestMonitor) currentTokS() float64 {
	if v, ok := m.live(true)["tok_s"].(float64); ok {
		return v
	}
	return 0
}

// currentPrefillTokS is the running request's prompt tokens per second for the sparkline (0 when nothing is
// reading a prompt, and 0 while generating a request whose prefill speed is not known).
func (m *requestMonitor) currentPrefillTokS() float64 {
	if v, ok := m.live(true)["prefill_tok_s_mean"].(float64); ok {
		return v
	}
	return 0
}

// RequestStats is the workload view the runtime controller reads. It is computed from finished requests only:
// a single in-flight request must not be mistaken for a sustained shift.
type RequestStats struct {
	Samples        int
	PromptP95      int
	OutputP95      int
	TTFTmsP95      float64
	LatencySP95    float64
	DecodeTokSMean float64
}

// recentStats summarizes the last window of finished requests. Missing fields are ignored, never invented.
func (m *requestMonitor) recentStats(window time.Duration) RequestStats {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.initLocked()
	if window <= 0 {
		window = 10 * time.Second
	}
	cutoff := float64(time.Now().Add(-window).UnixNano()) / 1e9
	var prompts, outputs []int
	var ttfts, durations []float64
	decodeSum, decodeN := 0.0, 0
	for _, r := range m.history {
		t, _ := r["time"].(float64)
		if t < cutoff {
			continue
		}
		if v, ok := r["prompt_tokens"].(int); ok {
			prompts = append(prompts, v)
		}
		if v, ok := r["output_tokens"].(int); ok {
			outputs = append(outputs, v)
		}
		if v, ok := r["prompt_ms"].(float64); ok {
			ttfts = append(ttfts, v)
		}
		if v, ok := r["duration_s"].(float64); ok {
			durations = append(durations, v)
		}
		if v, ok := r["decode_tok_s"].(float64); ok && v > 0 {
			decodeSum += v
			decodeN++
		}
	}
	st := RequestStats{Samples: len(prompts)}
	if st.Samples == 0 {
		st.Samples = len(ttfts)
	}
	st.PromptP95 = percentileInt(prompts, 0.95)
	st.OutputP95 = percentileInt(outputs, 0.95)
	st.TTFTmsP95 = percentileFloat(ttfts, 0.95)
	st.LatencySP95 = percentileFloat(durations, 0.95)
	if decodeN > 0 {
		st.DecodeTokSMean = decodeSum / float64(decodeN)
	}
	return st
}

func percentileInt(v []int, p float64) int {
	if len(v) == 0 {
		return 0
	}
	s := append([]int(nil), v...)
	sort.Ints(s)
	i := int(math.Ceil(p*float64(len(s)))) - 1
	if i < 0 {
		i = 0
	}
	if i >= len(s) {
		i = len(s) - 1
	}
	return s[i]
}

func percentileFloat(v []float64, p float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	i := int(math.Ceil(p*float64(len(s)))) - 1
	if i < 0 {
		i = 0
	}
	if i >= len(s) {
		i = len(s) - 1
	}
	return s[i]
}

// view returns the finished requests newest first (the last 12, or all kept), how many are kept, and the totals.
func (m *requestMonitor) view(all bool) ([]map[string]interface{}, int, map[string]interface{}) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.initLocked()
	n := len(m.history)
	limit := n
	if !all && limit > 12 {
		limit = 12
	}
	out := make([]map[string]interface{}, 0, limit)
	for i := 0; i < limit; i++ {
		out = append(out, m.history[n-1-i])
	}
	totals := make(map[string]interface{}, len(m.totals)+1)
	for k, v := range m.totals {
		totals[k] = v
	}
	totals["since"] = float64(m.since.UnixNano()) / 1e9
	return out, n, totals
}
