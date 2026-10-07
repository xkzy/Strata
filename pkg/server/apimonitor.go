// pkg/server/apimonitor.go - the standalone Monitor page (/monitor, /api-monitor): the last requests' bodies and
// answers, kept in memory. It is off unless the run config's "api_monitor" is true (or --api-monitor, or
// STRATA_API_MONITOR=1): the prompts and answers of every client are kept, so nothing is kept (and the page and
// /api/requests do not exist) until the user asks for it.
package server

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"strata/pkg/engineipc"
	"strata/pkg/runconfig"
)

const (
	apiRecordsKept = 100
	apiCaptureMax  = 262144 // chars of a body or an answer kept per request
)

type apiRecord struct {
	ID         string                 `json:"id"`
	Path       string                 `json:"path"`
	Model      string                 `json:"model"`
	StartedAt  float64                `json:"started_at"`
	State      string                 `json:"state"` // generating | completed | error | disconnected
	Stream     bool                   `json:"stream"`
	Scope      map[string]interface{} `json:"scope,omitempty"`
	HTTPStatus int                    `json:"http_status,omitempty"`
	WallclockS *float64               `json:"wallclock_s,omitempty"`
	FinishedAt *float64               `json:"finished_at,omitempty"`
	QueueS     float64                `json:"queue_s"`
	LoadS      float64                `json:"load_s"`
	FirstTokS  *float64               `json:"first_token_s"`
	Error      interface{}            `json:"error,omitempty"`
	Usage      interface{}            `json:"usage,omitempty"`
	Timings    interface{}            `json:"timings,omitempty"`

	Input      string `json:"input,omitempty"`
	Output     string `json:"output"`
	Reasoning  string `json:"reasoning"`
	Response   string `json:"response,omitempty"`
	InputTrunc bool   `json:"input_truncated"`
	OutputTrunc,
	ReasoningTrunc,
	ResponseTrunc bool `json:"-"`

	clock time.Time
}

type apiMonitorState struct {
	once    sync.Once
	on      bool
	mu      sync.Mutex
	records []*apiRecord // newest last
	seq     int
}

// apiMonitorOn says whether the opt-in monitor is on, reading the run config once.
func (s *StrataServer) apiMonitorOn() bool {
	s.apiMon.once.Do(func() {
		on := s.Config.APIMonitor || os.Getenv("STRATA_API_MONITOR") == "1"
		if p := s.runConfigPath(); !on && p != "" {
			if cfg, err := runconfig.LoadRaw(p); err == nil && cfg["api_monitor"] == true {
				on = true
			}
		}
		s.apiMon.on = on
	})
	return s.apiMon.on
}

// captureWriter passes the response through and keeps a bounded copy for the record.
type captureWriter struct {
	http.ResponseWriter
	status  int
	body    bytes.Buffer
	trunc   bool
	first   time.Time
	started bool
}

func (c *captureWriter) WriteHeader(code int) {
	if !c.started {
		c.status = code
	}
	c.started = true
	c.ResponseWriter.WriteHeader(code)
}

func (c *captureWriter) Write(b []byte) (int, error) {
	if !c.started {
		c.status, c.started = http.StatusOK, true
	}
	if c.first.IsZero() && len(b) > 0 {
		c.first = time.Now()
	}
	if room := apiCaptureMax*4 - c.body.Len(); room > 0 { // bytes; the text is cut to apiCaptureMax chars later
		if len(b) > room {
			c.body.Write(b[:room])
			c.trunc = true
		} else {
			c.body.Write(b)
		}
	} else {
		c.trunc = true
	}
	return c.ResponseWriter.Write(b)
}

func (c *captureWriter) Flush() {
	if f, ok := c.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (c *captureWriter) Unwrap() http.ResponseWriter { return c.ResponseWriter }

func clip(s string) (string, bool) {
	r := []rune(s)
	if len(r) <= apiCaptureMax {
		return s, false
	}
	return string(r[:apiCaptureMax]), true
}

// recordAPI keeps what the monitor shows about a POST to a model API; with the monitor off it is the handler itself.
func (s *StrataServer) recordAPI(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || !s.apiMonitorOn() {
			next(w, r)
			return
		}
		raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 64<<20))
		if err != nil {
			next(w, r)
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(raw))

		rec := &apiRecord{Path: r.URL.Path, Model: s.Config.ModelName, State: "generating", clock: time.Now(),
			StartedAt: float64(time.Now().UnixNano()) / 1e9}
		var req map[string]interface{}
		if json.Unmarshal(raw, &req) == nil {
			if m, _ := req["model"].(string); m != "" {
				rec.Model = m
			}
			rec.Stream, _ = req["stream"].(bool)
		}
		input := string(raw)
		var pretty bytes.Buffer
		if json.Indent(&pretty, raw, "", "  ") == nil {
			input = pretty.String()
		}
		rec.Input, rec.InputTrunc = clip(input)
		sc := requestScope(r, "", "")
		rec.Scope = map[string]interface{}{"tenant_id": sc.TenantID, "user_id": sc.UserID, "workspace_id": sc.WorkspaceID,
			"agent_id": sc.AgentID, "session_id": sc.SessionID}

		s.apiMon.mu.Lock()
		s.apiMon.seq++
		rec.ID = newRecordID(s.apiMon.seq)
		s.apiMon.records = append(s.apiMon.records, rec)
		if len(s.apiMon.records) > apiRecordsKept {
			s.apiMon.records = s.apiMon.records[len(s.apiMon.records)-apiRecordsKept:]
		}
		s.apiMon.mu.Unlock()

		cw := &captureWriter{ResponseWriter: w}
		next(cw, r)
		s.finishRecord(rec, cw, r.Context())
	}
}

func newRecordID(seq int) string {
	const hex = "0123456789abcdef"
	b := make([]byte, 0, 12)
	n := uint64(time.Now().UnixNano())*2654435761 + uint64(seq)
	for i := 0; i < 12; i++ {
		b = append(b, hex[n&15])
		n >>= 4
	}
	return string(b)
}

func (s *StrataServer) finishRecord(rec *apiRecord, cw *captureWriter, ctx context.Context) {
	body := cw.body.String()
	var output, reasoning string
	var usage, timings, errObj interface{}
	response := ""
	if strings.Contains(cw.Header().Get("Content-Type"), "text/event-stream") {
		output, reasoning, usage, timings, errObj = parseSSE(body)
	} else {
		response = body
		var obj map[string]interface{}
		if json.Unmarshal([]byte(body), &obj) == nil {
			output, reasoning = answerOf(obj)
			usage, timings, errObj = obj["usage"], obj["timings"], obj["error"]
		}
	}
	s.apiMon.mu.Lock()
	defer s.apiMon.mu.Unlock()
	rec.HTTPStatus = cw.status
	now := time.Now()
	wall := round3(now.Sub(rec.clock).Seconds())
	fin := float64(now.UnixNano()) / 1e9
	rec.WallclockS, rec.FinishedAt = &wall, &fin
	if !cw.first.IsZero() && cw.status < 400 { // an error body is no token
		f := round3(cw.first.Sub(rec.clock).Seconds())
		rec.FirstTokS = &f
	}
	rec.Output, rec.OutputTrunc = clip(output)
	rec.Reasoning, rec.ReasoningTrunc = clip(reasoning)
	rec.Response, rec.ResponseTrunc = clip(response)
	rec.Usage, rec.Timings, rec.Error = usage, timings, errObj
	switch {
	case ctx.Err() != nil:
		rec.State = "disconnected"
	case errObj != nil || cw.status >= 400:
		rec.State = "error"
		if rec.Error == nil {
			rec.Error = map[string]interface{}{"message": "HTTP " + http.StatusText(cw.status)}
		}
	default:
		rec.State = "completed"
	}
}

func round3(v float64) float64 { return float64(int(v*1000+0.5)) / 1000 }

// answerOf is the text and the reasoning of a complete answer (OpenAI chat / completions, Anthropic messages).
func answerOf(obj map[string]interface{}) (string, string) {
	var out, reasoning strings.Builder
	if choices, _ := obj["choices"].([]interface{}); len(choices) > 0 {
		c, _ := choices[0].(map[string]interface{})
		if msg, _ := c["message"].(map[string]interface{}); msg != nil {
			if t, _ := msg["content"].(string); t != "" {
				out.WriteString(t)
			}
			if t, _ := msg["reasoning_content"].(string); t != "" {
				reasoning.WriteString(t)
			}
		} else if t, _ := c["text"].(string); t != "" {
			out.WriteString(t)
		}
	}
	if blocks, _ := obj["content"].([]interface{}); blocks != nil {
		for _, b := range blocks {
			blk, _ := b.(map[string]interface{})
			switch blk["type"] {
			case "text":
				t, _ := blk["text"].(string)
				out.WriteString(t)
			case "thinking":
				t, _ := blk["thinking"].(string)
				reasoning.WriteString(t)
			}
		}
	}
	return out.String(), reasoning.String()
}

// parseSSE collects the answer, the reasoning, the usage and any error from a streamed answer in either dialect.
func parseSSE(body string) (output, reasoning string, usage, timings, errObj interface{}) {
	var out, rs strings.Builder
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}
		var ev map[string]interface{}
		if json.Unmarshal([]byte(data), &ev) != nil {
			continue
		}
		if e := ev["error"]; e != nil {
			errObj = e
		}
		if u := ev["usage"]; u != nil {
			usage = u
		}
		if t := ev["timings"]; t != nil {
			timings = t
		}
		if choices, _ := ev["choices"].([]interface{}); len(choices) > 0 { // OpenAI
			c, _ := choices[0].(map[string]interface{})
			if d, _ := c["delta"].(map[string]interface{}); d != nil {
				if t, _ := d["content"].(string); t != "" {
					out.WriteString(t)
				}
				if t, _ := d["reasoning_content"].(string); t != "" {
					rs.WriteString(t)
				}
			} else if t, _ := c["text"].(string); t != "" {
				out.WriteString(t)
			}
		}
		if d, _ := ev["delta"].(map[string]interface{}); d != nil { // Anthropic / Responses
			switch d["type"] {
			case "text_delta":
				t, _ := d["text"].(string)
				out.WriteString(t)
			case "thinking_delta":
				t, _ := d["thinking"].(string)
				rs.WriteString(t)
			}
			if u, _ := ev["usage"].(map[string]interface{}); u != nil {
				usage = u
			}
		}
		switch ev["type"] { // OpenAI Responses stream
		case "response.output_text.delta":
			t, _ := ev["delta"].(string)
			out.WriteString(t)
		case "response.reasoning_text.delta":
			t, _ := ev["delta"].(string)
			rs.WriteString(t)
		}
	}
	return out.String(), rs.String(), usage, timings, errObj
}

// handleAPIRequests is GET /api/requests (the list, newest first, without the bodies) and ?id= (one request whole).
func (s *StrataServer) handleAPIRequests(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet || !s.apiMonitorOn() {
		http.NotFound(w, r)
		return
	}
	s.apiMon.mu.Lock()
	defer s.apiMon.mu.Unlock()
	if id := r.URL.Query().Get("id"); id != "" {
		for _, rec := range s.apiMon.records {
			if rec.ID == id {
				writeJSON(w, http.StatusOK, s.recordView(rec, true))
				return
			}
		}
		apiMessage(w, http.StatusNotFound, "not_found", "request no longer retained")
		return
	}
	list := make([]interface{}, 0, len(s.apiMon.records))
	for i := len(s.apiMon.records) - 1; i >= 0; i-- {
		list = append(list, s.recordView(s.apiMon.records[i], false))
	}
	st, _ := s.EngineIPC.Status()
	writeJSON(w, http.StatusOK, map[string]interface{}{"requests": list, "retention": apiRecordsKept, "persistent": false,
		"loaded": st == engineipc.StateReady, "auto_load": false})
}

// recordView is the record as JSON; the list leaves the bodies out and gives a running request its time so far.
func (s *StrataServer) recordView(rec *apiRecord, full bool) map[string]interface{} {
	b, _ := json.Marshal(rec)
	out := map[string]interface{}{}
	_ = json.Unmarshal(b, &out)
	if !full {
		for _, k := range []string{"input", "output", "reasoning", "response"} {
			delete(out, k)
		}
	}
	if rec.WallclockS == nil {
		out["wallclock_s"] = round3(time.Since(rec.clock).Seconds())
	}
	for k, v := range map[string]bool{"output_truncated": rec.OutputTrunc, "reasoning_truncated": rec.ReasoningTrunc,
		"response_truncated": rec.ResponseTrunc} {
		out[k] = v
	}
	return out
}
