package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"unicode/utf8"

	"strata/pkg/chattemplate"
	"strata/pkg/engineipc"
	"strata/pkg/mathruntime"
	"strata/pkg/multitenant"
	"strata/pkg/rtclient"
)

// genSpec is one generation request after the API layer has normalised it.
type genSpec struct {
	Messages            []chattemplate.Message
	Tools               []json.RawMessage
	Opt                 chattemplate.Options
	RawPrompt           string // /v1/completions: the prompt is used as given, no template
	Raw                 bool
	Sampling            engineipc.SamplingParams
	ExplicitAbliterate  *bool
	MaxTokens           int // 0 = not given
	Stops               []string
	Scope               multitenant.SecurityScope // tenant / user / session of the caller (headers + body)
	Debug               bool                      // collect the runtime's decision trace for the Monitor (on unless X-Strata-Debug: 0)
	DebugEcho           bool                      // also return it in the response body (X-Strata-Debug: 1)
	DisableMath         bool                      // disable math interception (X-Strata-Math-Engine: off)
	DisableTools        bool                      // disable tool interception (X-Strata-Tool-Intercept: off)
	GuardOff            bool                      // skip the reply guard (X-Strata-Guard: off)
	mathDone, toolsDone bool                      // the interceptions ran: they run once per request, however many paths it takes
}

// requestScope derives the caller's scope: X-Tenant-ID / X-User-ID / X-Session-ID headers win over the body's
// user / session_id / metadata.user_id. A client that sends no session id shares the default session of its user: the
// front end is expected to send one per conversation.
func requestScope(r *http.Request, user, session string) multitenant.SecurityScope {
	var body *multitenant.SecurityScope
	if user != "" {
		body = &multitenant.SecurityScope{UserID: user, SessionID: user}
	}
	if session != "" {
		if body == nil {
			body = &multitenant.SecurityScope{}
		}
		body.SessionID = session
	}
	return multitenant.FromHeaders(r.Header, body)
}

type genEvent struct {
	parsedEvent
	End          bool
	Finish       string // stop | length | tool_calls | cancel
	GenTokens    int
	PromptTokens int
	TokPerSec    float64
	Err          error
	Trace        []rtclient.TraceEvent
}

// apiError carries an HTTP status for the API layer.
type apiError struct {
	Status     int
	Type       string
	Msg        string
	RetryAfter int // seconds; set by admission control
}

func (e *apiError) Error() string { return e.Msg }

// startGeneration renders and tokenizes the request and starts the engine. The returned channel yields reasoning /
// content / tool-call events and ends with one End event; it is closed afterwards.
func (s *StrataServer) startGeneration(ctx context.Context, spec genSpec) (<-chan genEvent, int, error) {
	if status, msg := s.engineUnavailable(); status != 0 {
		return nil, 0, &apiError{Status: status, Type: "engine_unavailable", Msg: msg}
	}
	tok := s.tokenizerOrNil()

	var ids []int
	startInReasoning := false
	if spec.Raw {
		// a raw completion is plain text: control tokens (<|im_end|> ...) typed in it stay text, like any caller-supplied string
		ids = tok.Encode(spec.RawPrompt, false)
	} else {
		s.interceptMathIntent(&spec)
		s.interceptToolIntent(&spec)
		text, plain, err := chattemplate.RenderSpans(spec.Messages, spec.Opt)
		if err != nil {
			var re *chattemplate.RequestError
			if errors.As(err, &re) {
				return nil, 0, &apiError{Status: 400, Type: "invalid_request_error", Msg: re.Msg}
			}
			return nil, 0, &apiError{Status: 500, Type: "server_error", Msg: err.Error()}
		}
		ids = tok.EncodeWithPlain(text, true, plain)
		startInReasoning = spec.Opt.AddGenerationPrompt && strings.HasSuffix(text, "<think>\n")
		if os.Getenv("STRATA_DEBUG_PROMPT") != "" {
			fmt.Fprintf(os.Stderr, "strata: prompt for %s/%s/%s (%d tokens):\n%s\n", spec.Scope.TenantID, spec.Scope.UserID, spec.Scope.SessionID, len(ids), text)
		}
	}
	ctxLimit := s.Config.MaxContext
	if ctxLimit <= 0 {
		ctxLimit = 65536
	}
	const engineSafetyMargin = 8
	targetHeadroom := min(max(spec.MaxTokens, 512), 4096)
	if targetHeadroom >= ctxLimit-engineSafetyMargin-16 {
		targetHeadroom = 512
	}
	maxPromptTokens := ctxLimit - targetHeadroom - engineSafetyMargin
	if maxPromptTokens < 16 {
		maxPromptTokens = ctxLimit - engineSafetyMargin - 1
	}

	if len(ids) > maxPromptTokens && !spec.Raw && len(spec.Messages) > 1 {
		// Virtual context sliding window compaction for multi-turn chats
		msgs := append([]chattemplate.Message(nil), spec.Messages...)
		// Compress first (old reasoning, long old bodies), so whole turns are dropped only if that is not enough.
		if compressOld(msgs) {
			if text, plain, err := chattemplate.RenderSpans(msgs, spec.Opt); err == nil {
				ids = tok.EncodeWithPlain(text, true, plain)
				startInReasoning = spec.Opt.AddGenerationPrompt && strings.HasSuffix(text, "<think>\n")
			}
		}
		for len(msgs) > 1 && len(ids) > maxPromptTokens {
			if msgs[0].Role == "system" || msgs[0].Role == "developer" {
				if len(msgs) > 2 {
					msgs = append(msgs[:1], msgs[2:]...)
				} else {
					break
				}
			} else {
				msgs = msgs[1:]
			}
			text, plain, err := chattemplate.RenderSpans(msgs, spec.Opt)
			if err == nil {
				ids = tok.EncodeWithPlain(text, true, plain)
				startInReasoning = spec.Opt.AddGenerationPrompt && strings.HasSuffix(text, "<think>\n")
			}
		}
	}

	avail := ctxLimit - len(ids) - engineSafetyMargin
	if avail < 1 {
		return nil, 0, &apiError{Status: 400, Type: "context_length_exceeded",
			Msg: fmt.Sprintf("the prompt is %d tokens, which does not fit the model's context of %d tokens", len(ids), ctxLimit)}
	}
	maxNew := spec.MaxTokens
	if maxNew <= 0 {
		maxNew = min(avail, 16384)
	}
	if maxNew > avail { // the reply is cut to what the context can hold
		maxNew = avail
	}

	s.eng.mu.RLock()
	stopIDs := s.eng.stopIDs
	s.eng.mu.RUnlock()

	if spec.ExplicitAbliterate != nil {
		spec.Sampling.Abliterate = *spec.ExplicitAbliterate
	} else if !spec.Sampling.Abliterate && s.ConfigManager != nil {
		cfg := s.ConfigManager.Get()
		if cfg.Abliterate {
			spec.Sampling.Abliterate = true
			if spec.Sampling.AbliterationStrength == 0 && cfg.AbliterationStrength > 0 {
				spec.Sampling.AbliterationStrength = cfg.AbliterationStrength
			}
		}
	}
	if spec.Sampling.Abliterate && spec.Sampling.AbliterationStrength == 0 {
		spec.Sampling.AbliterationStrength = 1.0
	}

	ticket, aerr := s.admitRequest(ctx, len(ids))
	if aerr != nil {
		return nil, 0, aerr
	}

	engOut := make(chan engineipc.TokenEvent, 64)
	go s.EngineIPC.GenerateIDs(ctx, ids, spec.Sampling, maxNew, stopIDs, spec.Stops, engOut)

	rq := s.mon.begin(len(ids), maxNew)
	rq.link(recordIDOf(ctx))
	out := make(chan genEvent, 64)
	go func() {
		defer s.releaseRequest(ticket)
		defer close(out)
		defer func() { // a request that ended without an End event: the client left, or the engine stopped
			if ctx.Err() != nil {
				rq.finish("disconnect", 0, 0)
			} else {
				rq.finish("error", 0, 0)
			}
		}()
		var parser *outParser
		if !spec.Raw {
			parser = newOutParser(startInReasoning, spec.Tools)
		}
		send := func(ev genEvent) bool {
			select {
			case out <- ev:
				return true
			case <-ctx.Done():
				return false
			}
		}
		alive := true // false once the consumer is gone: keep draining the engine, stop forwarding
		for ev := range engOut {
			if ev.Error != nil {
				rq.finish("error", ev.GenTokens, 0)
				if alive {
					send(genEvent{End: true, Err: ev.Error, GenTokens: ev.GenTokens})
				}
				continue
			}
			if ev.IsProgress {
				rq.promptProgress(ev.PromptRead, ev.PromptTotal)
				continue
			}
			if !ev.IsEnd {
				rq.token()
				if !alive {
					continue
				}
				if parser == nil {
					alive = send(genEvent{parsedEvent: parsedEvent{Kind: "content", Text: ev.Text}})
					continue
				}
				for _, pe := range parser.Feed(ev.Text) {
					if alive {
						alive = send(genEvent{parsedEvent: pe})
					}
				}
				continue
			}
			rq.finish(endFinish(ev.FinishReason), ev.GenTokens, ev.TokPerSec)
			if !alive {
				continue
			}
			if parser != nil {
				for _, pe := range parser.Flush() {
					if alive {
						alive = send(genEvent{parsedEvent: pe})
					}
				}
			}
			finish := ev.FinishReason
			if finish == "" {
				finish = "stop"
			}
			if finish == "stop" && parser != nil && parser.sawToolCall {
				finish = "tool_calls"
			}
			if alive {
				send(genEvent{End: true, Finish: finish, GenTokens: ev.GenTokens, PromptTokens: len(ids), TokPerSec: ev.TokPerSec})
			}
		}
	}()
	return out, len(ids), nil
}

// endFinish is the finish reason the Monitor records for an engine end event.
func endFinish(reason string) string {
	if reason == "" {
		return "stop"
	}
	return reason
}

func (s *StrataServer) interceptMathIntent(spec *genSpec) {
	if spec.mathDone {
		return
	}
	spec.mathDone = true
	if s.MathRuntime == nil || spec.DisableMath || os.Getenv("STRATA_MATH_INTERCEPTION") == "off" {
		return
	}
	for i := len(spec.Messages) - 1; i >= 0; i-- {
		if spec.Messages[i].Role == "user" {
			var mathFacts []string
			seenFacts := make(map[string]bool)
			cas := mathruntime.GetUnifiedCASEngine()

			extractFact := func(text string) {
				text = strings.TrimSpace(text)
				if text == "" {
					return
				}
				if casRes := cas.InterceptScientificIntent(text); casRes != nil && casRes.Status == mathruntime.StatusSuccess && (casRes.ExactResult != "" || casRes.NumericResult != "") {
					val := casRes.ExactResult
					if val == "" {
						val = casRes.NumericResult
					}
					fact := fmt.Sprintf("Verified fact for this answer: %s = %s", casRes.CanonicalExpression, val)
					if !seenFacts[fact] {
						seenFacts[fact] = true
						mathFacts = append(mathFacts, fact)
					}
				} else if res := s.MathRuntime.InterceptAndVerifyIntent(text); res != nil && res.Status == mathruntime.StatusSuccess && (res.ExactResult != "" || res.NumericResult != "") {
					val := res.ExactResult
					if val == "" {
						val = res.NumericResult
					}
					fact := fmt.Sprintf("Verified fact for this answer: %s = %s", res.CanonicalExpression, val)
					if !seenFacts[fact] {
						seenFacts[fact] = true
						mathFacts = append(mathFacts, fact)
					}
				}
			}

			content := spec.Messages[i].Content
			extractFact(content)

			// Also inspect each line if multi-line
			if strings.Contains(content, "\n") {
				lines := strings.Split(content, "\n")
				for _, line := range lines {
					extractFact(line)
				}
			}

			if len(mathFacts) > 0 {
				combinedFact := strings.Join(mathFacts, "\n")
				injected := false
				for j := range spec.Messages {
					if spec.Messages[j].Role == "system" {
						spec.Messages[j].Content = strings.TrimSpace(spec.Messages[j].Content + "\n\n" + combinedFact)
						injected = true
						break
					}
				}
				if !injected {
					spec.Messages = append([]chattemplate.Message{{Role: "system", Content: combinedFact}}, spec.Messages...)
				}
				break
			}
		}
	}
}

// setDebug reads the debug headers: the trace is always collected for the Monitor (STRATA_DEBUG=0 or X-Strata-Debug: 0 turn
// it off), and only X-Strata-Debug: 1 puts it in the response body.
func (spec *genSpec) setDebug(r *http.Request) {
	h := r.Header.Get("X-Strata-Debug")
	spec.Debug = h != "0" && os.Getenv("STRATA_DEBUG") != "0"
	spec.DebugEcho = h == "1"
}

// setAbliterate parses X-Strata-Abliterate and X-Strata-Abliteration-Strength headers.
func (spec *genSpec) setAbliterate(r *http.Request) {
	if h := r.Header.Get("X-Strata-Abliterate"); h != "" {
		v := h == "1" || strings.EqualFold(h, "true") || strings.EqualFold(h, "on")
		spec.Sampling.Abliterate = v
		spec.ExplicitAbliterate = &v
	} else if h := r.Header.Get("X-Abliterate"); h != "" {
		v := h == "1" || strings.EqualFold(h, "true") || strings.EqualFold(h, "on")
		spec.Sampling.Abliterate = v
		spec.ExplicitAbliterate = &v
	}
	if h := r.Header.Get("X-Strata-Abliteration-Strength"); h != "" {
		if v, err := strconv.ParseFloat(h, 64); err == nil && v >= 0 {
			spec.Sampling.AbliterationStrength = v
		}
	} else if h := r.Header.Get("X-Abliteration-Strength"); h != "" {
		if v, err := strconv.ParseFloat(h, 64); err == nil && v >= 0 {
			spec.Sampling.AbliterationStrength = v
		}
	}
}

const compressKeep = 600 // chars kept at each end of a long old message

// compressOld shrinks everything but the last two messages in place: it drops old reasoning and cuts long bodies
// to head+tail. shortcut: char-based cut, not a model summary; upgrade if quality of long chats suffers.
func compressOld(msgs []chattemplate.Message) bool {
	changed := false
	for i := 0; i < len(msgs)-2; i++ {
		if msgs[i].ReasoningContent != "" {
			msgs[i].ReasoningContent = ""
			changed = true
		}
		if c := msgs[i].Content; len(c) > 4*compressKeep && msgs[i].Role != "system" && msgs[i].Role != "developer" {
			h, t := compressKeep, len(c)-compressKeep
			for h > 0 && !utf8.RuneStart(c[h]) {
				h--
			}
			for t < len(c) && !utf8.RuneStart(c[t]) {
				t++
			}
			msgs[i].Content = c[:h] + "\n[... " + strconv.Itoa(t-h) + " chars compressed ...]\n" + c[t:]
			changed = true
		}
	}
	return changed
}
