package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"

	"strata/pkg/chattemplate"
	"strata/pkg/engineipc"
	"strata/pkg/multitenant"
	"strata/pkg/rtclient"
)

// genSpec is one generation request after the API layer has normalised it.
type genSpec struct {
	Messages  []chattemplate.Message
	Tools     []json.RawMessage
	Opt       chattemplate.Options
	RawPrompt string // /v1/completions: the prompt is used as given, no template
	Raw       bool
	Sampling  engineipc.SamplingParams
	MaxTokens int // 0 = not given
	Stops     []string
	Scope     multitenant.SecurityScope // tenant / user / session of the caller (headers + body)
	Debug     bool                      // return the runtime's decision trace (X-Strata-Debug: 1)
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
	Status int
	Type   string
	Msg    string
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
	avail := ctxLimit - len(ids)
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
	engOut := make(chan engineipc.TokenEvent, 64)
	go s.EngineIPC.GenerateIDs(ctx, ids, spec.Sampling, maxNew, stopIDs, spec.Stops, engOut)

	out := make(chan genEvent, 64)
	go func() {
		defer close(out)
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
				if alive {
					send(genEvent{End: true, Err: ev.Error, GenTokens: ev.GenTokens})
				}
				continue
			}
			if !ev.IsEnd {
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
