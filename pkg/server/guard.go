package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"

	"strata/pkg/chattemplate"
	"strata/pkg/hallucination"
	"strata/pkg/reqtrace"
)

// The reply guard checks a finished reply before the client sees it and, when it fails, asks the model again:
//
//   - the syntax of every tool call: the arguments are a JSON object, the tool is one the request offered, the required
//     arguments are there and their types fit the tool's schema;
//   - the arithmetic and unit claims in the text (the CAS decides them, see hallucination.Checker). Claims about files,
//     functions and routes are checked against the server's own directory, not the caller's project: they are shown in
//     the Monitor, they never cause a retry.
//
// Policy "regenerate" holds the reply back until it passes (a streamed reply cannot be taken back once sent), "monitor"
// streams as usual and only records the findings, "off" does nothing. Every attempt is written to the request's debug
// trail (source "guard"), which the Monitor shows.
const (
	guardOff        = "off"
	guardMonitor    = "monitor"
	guardRegenerate = "regenerate"

	guardMaxRetries = 2
	guardTempStep   = 0.3
)

var guardStats struct {
	checked, failed, retries, gaveUp, passedAfterRetry atomic.Uint64
}

var (
	guardOnce    sync.Once
	guardChecker *hallucination.Checker
)

func theGuardChecker() *hallucination.Checker {
	guardOnce.Do(func() { guardChecker = hallucination.NewChecker(".") })
	return guardChecker
}

// guardPolicy is the policy for one request: the caller's header, else the server setting, else STRATA_GUARD, else
// the run config; regenerate when none says otherwise.
func (s *StrataServer) guardPolicy(spec genSpec) string {
	if spec.GuardOff || spec.Raw {
		return guardOff
	}
	p := strings.ToLower(strings.TrimSpace(s.Config.GuardPolicy))
	if p == "" {
		p = strings.ToLower(strings.TrimSpace(os.Getenv("STRATA_GUARD")))
	}
	if p == "" && s.ConfigManager != nil {
		p = strings.ToLower(strings.TrimSpace(s.ConfigManager.Get().GuardPolicy))
	}
	switch p {
	case guardOff, "false", "0":
		return guardOff
	case guardMonitor:
		return guardMonitor
	}
	return guardRegenerate
}

// guardOffHeader is true when the caller switched the guard off (X-Strata-Guard: off).
func guardOffHeader(r *http.Request) bool { return toolsOff(r.Header.Get("X-Strata-Guard")) }

type toolSchema struct {
	required []string
	types    map[string]string // parameter -> JSON schema type, when it names exactly one
}

func parseToolSchemas(tools []json.RawMessage) map[string]toolSchema {
	out := map[string]toolSchema{}
	for _, raw := range tools {
		var t struct {
			Function struct {
				Name       string `json:"name"`
				Parameters struct {
					Required   []string `json:"required"`
					Properties map[string]struct {
						Type any `json:"type"`
					} `json:"properties"`
				} `json:"parameters"`
			} `json:"function"`
		}
		if json.Unmarshal(raw, &t) != nil || t.Function.Name == "" {
			continue
		}
		ts := toolSchema{required: t.Function.Parameters.Required, types: map[string]string{}}
		for k, v := range t.Function.Parameters.Properties {
			if s, ok := v.Type.(string); ok {
				ts.types[k] = s
			}
		}
		out[t.Function.Name] = ts
	}
	return out
}

// jsonKind names the JSON type of a raw value the way a JSON schema does.
func jsonKind(raw json.RawMessage) string {
	s := strings.TrimSpace(string(raw))
	switch {
	case s == "":
		return ""
	case s == "null":
		return "null"
	case s == "true" || s == "false":
		return "boolean"
	case s[0] == '"':
		return "string"
	case s[0] == '[':
		return "array"
	case s[0] == '{':
		return "object"
	}
	if strings.ContainsAny(s, ".eE") {
		return "number"
	}
	return "integer"
}

func typeFits(want, got string) bool {
	return want == got || (want == "number" && got == "integer")
}

// checkToolCalls returns one message per syntax problem in the tool calls; none means they are well formed.
func checkToolCalls(tools []json.RawMessage, calls []*parsedTool) []string {
	schemas := parseToolSchemas(tools)
	var bad []string
	for _, c := range calls {
		if c == nil {
			continue
		}
		sc, known := schemas[c.Name]
		if !known {
			if len(schemas) == 0 {
				bad = append(bad, fmt.Sprintf("tool call %q, but this request offers no tools", c.Name))
			} else {
				names := make([]string, 0, len(schemas))
				for n := range schemas {
					names = append(names, n)
				}
				bad = append(bad, fmt.Sprintf("tool %q does not exist (the tools are: %s)", c.Name, strings.Join(names, ", ")))
			}
			continue
		}
		var args map[string]json.RawMessage
		if strings.TrimSpace(c.Arguments) == "" {
			args = map[string]json.RawMessage{}
		} else if err := json.Unmarshal([]byte(c.Arguments), &args); err != nil {
			bad = append(bad, fmt.Sprintf("tool %q: the arguments are not a valid JSON object (%v)", c.Name, err))
			continue
		}
		for _, r := range sc.required {
			if _, ok := args[r]; !ok {
				bad = append(bad, fmt.Sprintf("tool %q: required argument %q is missing", c.Name, r))
			}
		}
		for k, v := range args {
			if want, ok := sc.types[k]; ok {
				if got := jsonKind(v); got != "null" && !typeFits(want, got) {
					bad = append(bad, fmt.Sprintf("tool %q: argument %q must be %s, not %s", c.Name, k, want, got))
				}
			}
		}
	}
	return bad
}

// guardAttempt is one complete generation, held back.
type guardAttempt struct {
	events  []genEvent
	content strings.Builder
	calls   []*parsedTool
	end     genEvent
	hasEnd  bool
}

func (a *guardAttempt) add(ev genEvent) {
	a.events = append(a.events, ev)
	switch {
	case ev.End:
		a.end, a.hasEnd = ev, true
	case ev.Kind == "content":
		a.content.WriteString(ev.Text)
	case ev.Kind == "tool" && ev.Tool != nil:
		a.calls = append(a.calls, ev.Tool)
	}
}

// problems runs both checks. The first result makes a retry worth doing; the second is only shown in the Monitor.
func (s *StrataServer) guardProblems(spec genSpec, a *guardAttempt, requestID string) (fix, advisory []string) {
	if a.end.Err != nil || a.end.Finish == "cancel" || a.end.Finish == "length" {
		return nil, nil // an error, a cancel, or a reply cut off: asking again would not repair it
	}
	fix = checkToolCalls(spec.Tools, a.calls)
	for _, f := range theGuardChecker().Check(a.content.String(), spec.Scope.SessionID, requestID) {
		line := fmt.Sprintf("%s claim %q: %s", strings.ToLower(f.Type), f.Span, f.Reason)
		if f.Actionable {
			fix = append(fix, line)
		} else {
			advisory = append(advisory, line)
		}
	}
	return fix, advisory
}

// guardRetrySpec asks again: the draft and what was wrong with it are added to the conversation and the temperature
// is lowered.
func guardRetrySpec(spec genSpec, a *guardAttempt, fix []string) genSpec {
	draft := a.content.String()
	for _, c := range a.calls {
		draft += fmt.Sprintf("\n[tool call %s %s]", c.Name, c.Arguments)
	}
	msgs := append([]chattemplate.Message(nil), spec.Messages...)
	msgs = append(msgs,
		chattemplate.Message{Role: "assistant", Content: draft},
		chattemplate.Message{Role: "user", Content: "Your last answer failed an automatic check:\n- " + strings.Join(fix, "\n- ") +
			"\nAnswer again and fix exactly these points. Do not mention this check."})
	spec.Messages = msgs
	spec.Sampling.Temperature = max(0, spec.Sampling.Temperature-guardTempStep)
	spec.Sampling.Seed++
	return spec
}

// guardStartHook replaces the generation start in tests.
var guardStartHook func(ctx context.Context, spec genSpec) (<-chan genEvent, int, error)

func (s *StrataServer) guardStart(ctx context.Context, spec genSpec) (<-chan genEvent, int, error) {
	if guardStartHook != nil {
		return guardStartHook(ctx, spec)
	}
	return s.startChatUnguarded(ctx, spec)
}

// startChatOne picks the path for a request (see startChatUnguarded) and puts the reply guard around it.
func (s *StrataServer) startChatOne(ctx context.Context, spec genSpec) (<-chan genEvent, int, error) {
	policy := s.guardPolicy(spec)
	if policy == guardOff {
		return s.guardStart(ctx, spec)
	}
	// The interceptions run once, before the first attempt, so a retry does not read the correction as a user turn.
	s.interceptMathIntent(&spec)
	s.interceptToolIntent(&spec)

	first, promptTokens, err := s.guardStart(ctx, spec)
	if err != nil {
		return nil, 0, err
	}
	requestID := fmt.Sprintf("guard-%p", &spec)
	out := make(chan genEvent, 64)
	send := func(ev genEvent) bool {
		select {
		case out <- ev:
			return true
		case <-ctx.Done():
			return false
		}
	}
	go func() {
		defer close(out)
		cur, events := spec, first
		for attempt := 0; ; attempt++ {
			a := &guardAttempt{}
			for ev := range events {
				a.add(ev)
				if policy == guardMonitor && !ev.End { // monitor: stream as usual, check at the end
					if !send(ev) {
						for range events {
						}
						return
					}
				}
			}
			guardStats.checked.Add(1)
			fix, advisory := s.guardProblems(cur, a, requestID)
			for _, m := range advisory {
				reqtrace.Add(ctx, "guard", "advisory (no retry): %s", m)
			}
			flush := func() bool {
				from := 0
				if policy == guardMonitor { // everything but the end event went out already
					from = len(a.events) - 1
					if !a.hasEnd {
						from = len(a.events)
					}
				}
				for _, ev := range a.events[from:] {
					if !send(ev) {
						return false
					}
				}
				return true
			}
			if len(fix) == 0 {
				if attempt > 0 {
					guardStats.passedAfterRetry.Add(1)
					reqtrace.Add(ctx, "guard", "attempt %d passed the checks", attempt+1)
				} else {
					reqtrace.Add(ctx, "guard", "checked: tool calls and arithmetic claims are fine")
				}
				flush()
				return
			}
			guardStats.failed.Add(1)
			for _, m := range fix {
				reqtrace.Add(ctx, "guard", "attempt %d failed: %s", attempt+1, m)
			}
			if policy == guardMonitor || attempt >= guardMaxRetries || ctx.Err() != nil {
				if policy == guardRegenerate {
					guardStats.gaveUp.Add(1)
					reqtrace.Add(ctx, "guard", "still failing after %d attempts: the last answer is returned as it is", attempt+1)
				}
				flush()
				return
			}
			next := guardRetrySpec(cur, a, fix)
			guardStats.retries.Add(1)
			reqtrace.Add(ctx, "guard", "regenerating (attempt %d of %d, temperature %.2f -> %.2f), the rejected answer is held back",
				attempt+2, guardMaxRetries+1, cur.Sampling.Temperature, next.Sampling.Temperature)
			retry, _, rerr := s.guardStart(ctx, next)
			if rerr != nil { // the engine went away between attempts: give the first answer back rather than nothing
				reqtrace.Add(ctx, "guard", "cannot regenerate (%v): the answer is returned as it is", rerr)
				flush()
				return
			}
			cur, events = next, retry
		}
	}()
	return out, promptTokens, nil
}

// handleGuardMetrics is GET /v1/strata/guard/metrics: how often the guard checked, failed, retried and gave up.
func (s *StrataServer) handleGuardMetrics(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"checked": guardStats.checked.Load(), "failed": guardStats.failed.Load(), "retries": guardStats.retries.Load(),
		"gave_up": guardStats.gaveUp.Load(), "passed_after_retry": guardStats.passedAfterRetry.Load(),
	})
}
