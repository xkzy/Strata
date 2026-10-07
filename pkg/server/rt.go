package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"strata/pkg/chattemplate"
	"strata/pkg/engineipc"
	"strata/pkg/multitenant"
	"strata/pkg/rtclient"
)

// The C++ transparent runtime sits between the API and the engine. A chat request still looks like an ordinary model call
// to the client; internally the runtime keeps the session's virtual context, retrieves what the model needs, checks what
// the model says and regenerates when a claim is contradicted. See rtclient for the protocol.

// rtState is the runtime's place in the server.
type rtState struct {
	mu     sync.RWMutex
	client *rtclient.Client
	mode   string // on | off | auto
	note   string // why the runtime is not in use (shown in /health)
}

func (s *StrataServer) rtActive() bool {
	s.rt.mu.RLock()
	defer s.rt.mu.RUnlock()
	return s.rt.client != nil
}

// setupRuntime starts the sidecar according to the configured mode. Called once the engine and tokenizer exist.
func (s *StrataServer) setupRuntime(vocab int) {
	mode := strings.ToLower(s.Config.RuntimeMode)
	if mode == "" {
		mode = "auto"
	}
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	s.rt.mode = mode
	if mode == "off" {
		s.rt.note = "disabled (--rt=off): requests go straight to the model"
		return
	}
	bin := rtclient.FindBinary(s.Config.RuntimeBinary)
	if bin == "" {
		s.rt.note = "strata_rt_server not found (build it with: cmake --build build --target strata_rt_server): requests go straight to the model, with no verification or virtual context"
		fmt.Fprintln(os.Stderr, "strata: warning:", s.rt.note)
		return
	}
	window := s.Config.WindowTokens
	if window <= 0 {
		window = 32768
	}
	if window > s.Config.MaxContext {
		window = s.Config.MaxContext
	}
	s.eng.mu.RLock()
	tmpl := s.eng.templateSHA
	s.eng.mu.RUnlock()
	cl := rtclient.New(rtclient.Config{
		Binary: bin, PhysicalTokens: window, Persist: true, Verification: true,
		Identity: fmt.Sprintf("%s|vocab=%d|tmpl=%s", s.Config.ModelName, vocab, tmpl),
	})
	s.rt.client = cl
	go func() { // start now so a broken sidecar shows up in the log, not on the first request
		if _, err := cl.Metrics(context.Background()); err != nil {
			fmt.Fprintln(os.Stderr, "strata: warning: the runtime could not start:", err)
		}
	}()
}

func (s *StrataServer) runtimeStatus() map[string]interface{} {
	s.rt.mu.RLock()
	defer s.rt.mu.RUnlock()
	out := map[string]interface{}{"mode": s.rt.mode}
	if s.rt.client == nil {
		out["active"] = false
		if s.rt.note != "" {
			out["note"] = s.rt.note
		}
		return out
	}
	out["active"] = true
	out["healthy"] = s.rt.client.Healthy()
	if e := s.rt.client.LastError(); e != "" {
		out["error"] = e
	}
	return out
}

// requestSession decides the virtual-context identity of a request. A caller that names its session (X-Session-ID /
// session_id / user) gets a persistent context under that name. A caller that does not gets a random, private context for
// this one request: it is never stored and no other request can reach it. (An id derived from the conversation's content
// would collide between unrelated callers who open with the same words, and would let one read the other's context.)
func requestSession(sc multitenant.SecurityScope) (multitenant.SecurityScope, bool) {
	if sc.SessionID != multitenant.DefaultSecurityScope().SessionID {
		return sc, false
	}
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic("no randomness available: " + err.Error())
	}
	sc.SessionID = "ephemeral-" + hex.EncodeToString(b[:])
	return sc, true
}

// rtBackend is the host side of one runtime request.
type rtBackend struct {
	s       *StrataServer
	opt     chattemplate.Options
	started chan struct{} // closed when the runtime has asked for a generation, or its prompt was refused
	once    sync.Once
	mu      sync.Mutex
	err     error
	stops   []string
}

func (b *rtBackend) signal(err error) {
	b.once.Do(func() {
		b.mu.Lock()
		b.err = err
		b.mu.Unlock()
		close(b.started)
	})
}

func (b *rtBackend) BuildPrompt(ctx context.Context, msgs []rtclient.Message, prefix string) ([]int, error) {
	tok := b.s.tokenizerOrNil()
	if tok == nil {
		err := &apiError{Status: 503, Type: "engine_unavailable", Msg: "the model's tokenizer is not loaded"}
		b.signal(err)
		return nil, err
	}
	// the template accepts system text only at the start: any system message (the runtime adds notes as system messages)
	// is merged into one leading block
	var sys []string
	var rest []chattemplate.Message
	for _, m := range msgs {
		if m.Role == "system" || m.Role == "developer" {
			if strings.TrimSpace(m.Content) != "" {
				sys = append(sys, m.Content)
			}
			continue
		}
		rest = append(rest, chattemplate.Message{Role: m.Role, Content: m.Content})
	}
	var all []chattemplate.Message
	if len(sys) > 0 {
		all = append(all, chattemplate.Message{Role: "system", Content: strings.Join(sys, "\n\n")})
	}
	all = append(all, rest...)
	text, spans, err := chattemplate.RenderSpans(all, b.opt)
	if err != nil {
		var re *chattemplate.RequestError
		var ae *apiError
		if errors.As(err, &re) {
			ae = &apiError{Status: 400, Type: "invalid_request_error", Msg: re.Msg}
		} else {
			ae = &apiError{Status: 500, Type: "server_error", Msg: err.Error()}
		}
		b.signal(ae)
		return nil, ae
	}
	if prefix != "" {
		spans = append(spans, [2]int{len(text), len(text) + len(prefix)}) // text the model already produced is not control input
		text += prefix
	}
	ids := tok.EncodeWithPlain(text, true, spans)
	if len(ids) >= b.s.Config.MaxContext-1 {
		ae := &apiError{Status: 400, Type: "context_length_exceeded",
			Msg: fmt.Sprintf("the prompt is %d tokens, which does not fit the model's context of %d tokens", len(ids), b.s.Config.MaxContext)}
		b.signal(ae)
		return nil, ae
	}
	if os.Getenv("STRATA_DEBUG_PROMPT") != "" {
		fmt.Fprintf(os.Stderr, "strata: runtime prompt (%d tokens):\n%s\n", len(ids), text)
	}
	return ids, nil
}

func (b *rtBackend) Generate(ctx context.Context, ids []int, sp rtclient.Sampling, maxTokens int) <-chan rtclient.GenToken {
	b.signal(nil)
	out := make(chan rtclient.GenToken, 64)
	avail := b.s.Config.MaxContext - len(ids)
	if maxTokens <= 0 || maxTokens > avail {
		maxTokens = avail
	}
	b.s.eng.mu.RLock()
	stopIDs := b.s.eng.stopIDs
	b.s.eng.mu.RUnlock()
	eng := make(chan engineipc.TokenEvent, 64)
	go b.s.EngineIPC.GenerateIDs(ctx, ids, engineipc.SamplingParams{
		Temperature: sp.Temperature, TopP: sp.TopP, TopK: sp.TopK, MinP: sp.MinP, RepetitionPenalty: sp.RepetitionPenalty,
		FrequencyPenalty: sp.FrequencyPenalty, PresencePenalty: sp.PresencePenalty, Seed: sp.Seed,
	}, maxTokens, stopIDs, sp.Stop, eng)
	go func() {
		defer close(out)
		for ev := range eng {
			switch {
			case ev.Error != nil:
				out <- rtclient.GenToken{End: true, Err: ev.Error}
			case ev.IsEnd:
				out <- rtclient.GenToken{End: true, Finish: ev.FinishReason}
			default:
				out <- rtclient.GenToken{ID: ev.TokenID, Text: ev.Text}
			}
		}
	}()
	return out
}

// startRuntimeGeneration runs a chat request through the C++ runtime. The events have the same shape as the direct path's.
func (s *StrataServer) startRuntimeGeneration(ctx context.Context, spec genSpec) (<-chan genEvent, int, error) {
	if status, msg := s.engineUnavailable(); status != 0 {
		return nil, 0, &apiError{Status: status, Type: "engine_unavailable", Msg: msg}
	}
	s.rt.mu.RLock()
	client := s.rt.client
	s.rt.mu.RUnlock()

	msgs := make([]rtclient.Message, 0, len(spec.Messages))
	for _, m := range spec.Messages {
		content := m.Content
		if m.Role == "assistant" && len(m.ToolCalls) > 0 {
			content = chattemplate.FlattenToolCalls(m.Content, m.ToolCalls)
		}
		msgs = append(msgs, rtclient.Message{Role: m.Role, Content: content})
	}
	thinking := spec.Opt.EnableThinking == nil || *spec.Opt.EnableThinking
	sc, ephemeral := requestSession(spec.Scope)
	maxTokens := spec.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 16384
	}
	req := rtclient.Request{
		Scope:    rtclient.Scope{Tenant: sc.TenantID, User: sc.UserID, Workspace: sc.WorkspaceID, Agent: sc.AgentID, Session: sc.SessionID},
		Messages: msgs,
		Sampling: rtclient.Sampling{Temperature: spec.Sampling.Temperature, TopP: spec.Sampling.TopP, TopK: spec.Sampling.TopK, MinP: spec.Sampling.MinP,
			RepetitionPenalty: spec.Sampling.RepetitionPenalty, FrequencyPenalty: spec.Sampling.FrequencyPenalty, PresencePenalty: spec.Sampling.PresencePenalty,
			Seed: spec.Sampling.Seed, Stop: spec.Stops},
		MaxTokens: maxTokens, ReasoningPrefix: thinking && spec.Opt.AddGenerationPrompt, Debug: spec.Debug, Ephemeral: ephemeral,
	}
	be := &rtBackend{s: s, opt: spec.Opt, started: make(chan struct{})}
	out := make(chan genEvent, 64)
	parser := newOutParser(req.ReasoningPrefix, spec.Tools)
	failed := make(chan error, 1)

	go func() {
		defer close(out)
		send := func(ev genEvent) {
			select {
			case out <- ev:
			case <-ctx.Done():
			}
		}
		done, err := client.Run(ctx, req, be, func(text string) {
			for _, pe := range parser.Feed(text) {
				send(genEvent{parsedEvent: pe})
			}
		})
		if err != nil {
			be.signal(err) // an error before any generation reaches the caller as an HTTP status
			failed <- err
			var ae *rtclient.APIError
			if errors.As(err, &ae) {
				err = &apiError{Status: ae.Status, Type: ae.Type, Msg: ae.Msg}
			}
			send(genEvent{End: true, Err: err})
			return
		}
		for _, pe := range parser.Flush() {
			send(genEvent{parsedEvent: pe})
		}
		finish := done.Finish
		if finish == "stop" && parser.sawToolCall {
			finish = "tool_calls"
		}
		send(genEvent{End: true, Finish: finish, GenTokens: done.CompletionTokens, PromptTokens: done.PromptTokens, Trace: done.Trace})
	}()

	// wait until the request is past prompt building: a refused prompt is an HTTP error, not a half-sent stream
	select {
	case <-be.started:
		be.mu.Lock()
		err := be.err
		be.mu.Unlock()
		if err != nil {
			for range out { // drain
			}
			return nil, 0, err
		}
	case err := <-failed:
		for range out {
		}
		var ae *rtclient.APIError
		if errors.As(err, &ae) {
			return nil, 0, &apiError{Status: ae.Status, Type: ae.Type, Msg: ae.Msg}
		}
		return nil, 0, err
	case <-time.After(10 * time.Minute):
		return nil, 0, &apiError{Status: 504, Type: "runtime_error", Msg: "the runtime did not start the request"}
	}
	return out, 0, nil
}

// startChat picks the path for a request: through the runtime when it is active, else straight to the engine.
func (s *StrataServer) startChat(ctx context.Context, spec genSpec) (<-chan genEvent, int, error) {
	if !spec.Raw && s.rtActive() {
		return s.startRuntimeGeneration(ctx, spec)
	}
	return s.startGeneration(ctx, spec)
}
