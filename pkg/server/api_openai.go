package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"strata/pkg/chattemplate"
	"strata/pkg/engineipc"
	"strata/pkg/multitenant"
	"strata/pkg/rtclient"
)

// ---- shared request pieces ----

type oaMessage struct {
	Role             string          `json:"role"`
	Content          json.RawMessage `json:"content"`
	ReasoningContent string          `json:"reasoning_content"`
	Reasoning        string          `json:"reasoning"`
	ToolCalls        []oaToolCall    `json:"tool_calls"`
}

type oaToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      *string         `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

type oaRequest struct {
	Model         string          `json:"model"`
	Messages      []oaMessage     `json:"messages"`
	Prompt        json.RawMessage `json:"prompt"`
	Stream        bool            `json:"stream"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	Temperature          *float64               `json:"temperature"`
	TopP                 *float64               `json:"top_p"`
	TopK                 *int                   `json:"top_k"`
	MinP                 *float64               `json:"min_p"`
	MaxTokens            *int                   `json:"max_tokens"`
	MaxCompletionTokens  *int                   `json:"max_completion_tokens"`
	Stop                 json.RawMessage        `json:"stop"`
	Seed                 *int64                 `json:"seed"`
	FrequencyPenalty     *float64               `json:"frequency_penalty"`
	PresencePenalty      *float64               `json:"presence_penalty"`
	RepetitionPenalty    *float64               `json:"repetition_penalty"`
	Tools                []json.RawMessage      `json:"tools"`
	ToolChoice           json.RawMessage        `json:"tool_choice"`
	ReasoningEffort      string                 `json:"reasoning_effort"`
	EnableThinking       *bool                  `json:"enable_thinking"`
	MathEngine           *bool                  `json:"math_engine"`
	Abliterate           *bool                  `json:"abliterate"`
	AbliterationStrength *float64               `json:"abliteration_strength"`
	ChatTemplateKwargs   map[string]interface{} `json:"chat_template_kwargs"`
	User                 string                 `json:"user"`
	SessionID            string                 `json:"session_id"`
	Metadata             struct {
		UserID string `json:"user_id"`
	} `json:"metadata"`
}

func (r *oaRequest) scope(h *http.Request) multitenant.SecurityScope {
	user := r.User
	if user == "" {
		user = r.Metadata.UserID
	}
	return requestScope(h, user, r.SessionID)
}

func badRequest(msg string) *apiError {
	return &apiError{Status: 400, Type: "invalid_request_error", Msg: msg}
}

// contentText flattens a message's content (a string, or a list of parts) to text. Images are refused: this engine
// configuration has no vision path.
func contentText(raw json.RawMessage) (string, error) {
	raw = json.RawMessage(strings.TrimSpace(string(raw)))
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return "", badRequest("message content is not valid JSON text")
		}
		return s, nil
	}
	var parts []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", badRequest("message content must be a string or a list of content parts")
	}
	var sb strings.Builder
	for _, p := range parts {
		var typ string
		_ = json.Unmarshal(p["type"], &typ)
		switch typ {
		case "text", "input_text", "":
			var t string
			if json.Unmarshal(p["text"], &t) == nil {
				sb.WriteString(t)
			}
		case "image_url", "input_image", "image":
			return "", badRequest("image input is not supported: this model is running without a vision path")
		}
	}
	return sb.String(), nil
}

// toolArguments normalises a tool call's arguments to a JSON object (clients send a JSON string).
// toolArguments turns a call's arguments from a client's history into the JSON object the template renders. An object
// is used as it is and empty is nil; anything else (cut-off or malformed JSON, a list, a number) is kept as text under
// "arguments" instead of refusing the whole conversation: an agent resends its history, and one call that was cut off
// by max_tokens must not make every later request fail.
func toolArguments(raw json.RawMessage) (json.RawMessage, error) {
	raw = json.RawMessage(strings.TrimSpace(string(raw)))
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	text := string(raw)
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return keepAsText(text), nil
		}
		if strings.TrimSpace(s) == "" {
			return nil, nil
		}
		raw, text = json.RawMessage(s), s
	}
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(raw, &probe); err != nil {
		return keepAsText(text), nil
	}
	return raw, nil
}

func keepAsText(text string) json.RawMessage {
	b, _ := json.Marshal(map[string]string{"arguments": text})
	return b
}

func (r *oaRequest) toSpec(s *StrataServer) (genSpec, error) {
	var spec genSpec
	if err := s.useModel(context.Background(), r.Model); err != nil { // before the template format is read
		return spec, err
	}
	for _, m := range r.Messages {
		text, err := contentText(m.Content)
		if err != nil {
			return spec, err
		}
		cm := chattemplate.Message{Role: m.Role, Content: text, ReasoningContent: m.ReasoningContent}
		if cm.ReasoningContent == "" {
			cm.ReasoningContent = m.Reasoning
		}
		for _, tc := range m.ToolCalls {
			if tc.Function.Name == nil || *tc.Function.Name == "" {
				return spec, badRequest("Tool call is missing a function name.")
			}
			args, err := toolArguments(tc.Function.Arguments)
			if err != nil {
				return spec, err
			}
			cm.ToolCalls = append(cm.ToolCalls, chattemplate.ToolCall{Name: *tc.Function.Name, Arguments: args})
		}
		spec.Messages = append(spec.Messages, cm)
	}
	if len(spec.Messages) == 0 {
		return spec, badRequest("messages array is required")
	}
	if strings.Trim(string(r.ToolChoice), `" `) != "none" {
		spec.Tools = r.Tools
	}
	spec.Opt = chattemplate.Options{Tools: spec.Tools, AddGenerationPrompt: true, ReasoningEffort: r.ReasoningEffort, EnableThinking: r.EnableThinking}
	spec.Opt.Format = s.templateFormat()
	if v, ok := r.ChatTemplateKwargs["enable_thinking"].(bool); ok {
		spec.Opt.EnableThinking = &v
	}
	if v, ok := r.ChatTemplateKwargs["reasoning_effort"].(string); ok && spec.Opt.ReasoningEffort == "" {
		spec.Opt.ReasoningEffort = v
	}
	if v, ok := r.ChatTemplateKwargs["preserve_thinking"].(bool); ok {
		spec.Opt.PreserveThinking = &v
	}
	if err := normalizeEffort(&spec.Opt); err != nil {
		return spec, err
	}
	r.applyCommon(s, &spec)
	return spec, nil
}

// normalizeEffort maps the spellings clients use onto the template's levels (none = thinking off).
func normalizeEffort(o *chattemplate.Options) error {
	if o.ReasoningEffort == "" {
		return nil
	}
	switch strings.ToLower(strings.TrimSpace(o.ReasoningEffort)) {
	case "none", "off", "minimal", "disabled":
		off := false
		o.EnableThinking = &off
		o.ReasoningEffort = ""
	case "low":
		o.ReasoningEffort = "low"
	case "medium":
		o.ReasoningEffort = "medium"
	case "high", "xhigh", "max", "maximum":
		o.ReasoningEffort = "xhigh"
	default:
		return badRequest(fmt.Sprintf("unknown reasoning effort %q: use none, low, medium or high", o.ReasoningEffort))
	}
	return nil
}

func (r *oaRequest) applyCommon(s *StrataServer, spec *genSpec) {
	spec.Sampling = s.defaultSampling(thinkingOn(spec.Opt))
	if r.Temperature != nil {
		spec.Sampling.Temperature = *r.Temperature
	}
	if r.TopP != nil {
		spec.Sampling.TopP = *r.TopP
	}
	if r.TopK != nil {
		spec.Sampling.TopK = *r.TopK
	}
	if r.MinP != nil {
		spec.Sampling.MinP = *r.MinP
	}
	if r.Seed != nil {
		spec.Sampling.Seed = *r.Seed
	}
	if r.FrequencyPenalty != nil {
		spec.Sampling.FrequencyPenalty = *r.FrequencyPenalty
	}
	if r.PresencePenalty != nil {
		spec.Sampling.PresencePenalty = *r.PresencePenalty
	}
	if r.RepetitionPenalty != nil {
		spec.Sampling.RepetitionPenalty = *r.RepetitionPenalty
	}
	if r.Abliterate != nil {
		spec.Sampling.Abliterate = *r.Abliterate
		spec.ExplicitAbliterate = r.Abliterate
	}
	if r.AbliterationStrength != nil {
		spec.Sampling.AbliterationStrength = *r.AbliterationStrength
	}
	if r.MaxCompletionTokens != nil && *r.MaxCompletionTokens > 0 {
		spec.MaxTokens = *r.MaxCompletionTokens
	} else if r.MaxTokens != nil && *r.MaxTokens > 0 {
		spec.MaxTokens = *r.MaxTokens
	}
	if r.MathEngine != nil && !*r.MathEngine {
		spec.DisableMath = true
	}
	if v, ok := r.ChatTemplateKwargs["math_engine"].(bool); ok && !v {
		spec.DisableMath = true
	}
	if len(r.Stop) > 0 {
		var one string
		var many []string
		if json.Unmarshal(r.Stop, &one) == nil && one != "" {
			spec.Stops = append(spec.Stops, one)
		} else if json.Unmarshal(r.Stop, &many) == nil {
			for _, m := range many {
				if m != "" {
					spec.Stops = append(spec.Stops, m)
				}
			}
		}
	}
}

// defaultSampling is what a request that sends no sampling keys decodes with: the config's "sampling" block when it has
// one, otherwise the model card's set for the request's thinking state (cardSampling). A request's own fields win over
// it; greedy is for whoever asks (temperature 0 in the request, or in the config's block).
func (s *StrataServer) defaultSampling(thinking bool) engineipc.SamplingParams {
	s.eng.mu.RLock()
	defer s.eng.mu.RUnlock()
	if s.eng.samplingSet {
		return s.eng.sampling
	}
	return cardSampling(thinking)
}

// loadSamplingDefaults reads the config's "sampling" block; ok is false when the file has none (or is unreadable).
func loadSamplingDefaults(path string) (engineipc.SamplingParams, bool) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return engineipc.SamplingParams{}, false
	}
	var c struct {
		Sampling *struct {
			Temperature float64 `json:"temperature"`
			TopP        float64 `json:"top_p"`
			TopK        int     `json:"top_k"`
			MinP        float64 `json:"min_p"`
		} `json:"sampling"`
	}
	if json.Unmarshal(raw, &c) != nil || c.Sampling == nil {
		return engineipc.SamplingParams{}, false
	}
	return engineipc.SamplingParams{Temperature: c.Sampling.Temperature, TopP: c.Sampling.TopP, TopK: c.Sampling.TopK, MinP: c.Sampling.MinP}, true
}

// ---- HTTP helpers ----

func writeAPIError(w http.ResponseWriter, err error) {
	var ae *apiError
	if e, ok := err.(*apiError); ok {
		ae = e
	} else {
		ae = &apiError{Status: 500, Type: "server_error", Msg: err.Error()}
	}
	retry := ae.RetryAfter
	if retry <= 0 && ae.Status == 503 {
		retry = 5
	}
	if retry > 0 {
		w.Header().Set("Retry-After", fmt.Sprint(retry))
	}
	writeJSON(w, ae.Status, map[string]interface{}{"error": map[string]interface{}{"message": ae.Msg, "type": ae.Type, "code": ae.Type}})
}

func readJSONBody(w http.ResponseWriter, r *http.Request, into interface{}) bool {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 64<<20))
	if err != nil {
		writeAPIError(w, badRequest("could not read the request body"))
		return false
	}
	if err := json.Unmarshal(body, into); err != nil {
		writeAPIError(w, badRequest("Invalid JSON body: "+err.Error()))
		return false
	}
	return true
}

func (s *StrataServer) registerRequest(ctx context.Context, id string) (context.Context, func()) {
	ctx, cancel := context.WithCancel(ctx)
	s.cancelMu.Lock()
	s.activeCancels[id] = cancel
	s.cancelMu.Unlock()
	return ctx, func() {
		s.cancelMu.Lock()
		delete(s.activeCancels, id)
		s.cancelMu.Unlock()
		cancel()
	}
}

func sseWrite(w http.ResponseWriter, f http.Flusher, payload interface{}) {
	b, _ := json.Marshal(payload)
	fmt.Fprintf(w, "data: %s\n\n", b)
	if f != nil {
		f.Flush()
	}
}

// ---- /v1/chat/completions ----

func (s *StrataServer) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req oaRequest
	if !readJSONBody(w, r, &req) {
		return
	}
	s.applySharedOA(&req)
	spec, err := req.toSpec(s)
	if err != nil {
		writeAPIError(w, err)
		return
	}
	spec.Scope = req.scope(r)
	spec.setDebug(r)
	spec.setAbliterate(r)
	if h := r.Header.Get("X-Strata-Math-Engine"); h == "off" || h == "false" || h == "0" {
		spec.DisableMath = true
	}
	spec.DisableTools = toolsOff(r.Header.Get("X-Strata-Tool-Intercept"))
	spec.GuardOff = guardOffHeader(r)
	reqID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()
	ctx, done := s.registerRequest(r.Context(), reqID)
	defer done()
	events, _, err := s.startChat(ctx, spec)
	if err != nil {
		writeAPIError(w, err)
		return
	}
	model := s.Config.ModelName
	if req.Model != "" && s.Registry != nil {
		model = s.Registry.ResolveModel(req.Model, s.Config.ModelName)
	} else if req.Model != "" {
		model = req.Model
	}
	chunk := func(delta map[string]interface{}, finish interface{}) map[string]interface{} {
		return map[string]interface{}{"id": reqID, "object": "chat.completion.chunk", "created": created, "model": model,
			"choices": []map[string]interface{}{{"index": 0, "delta": delta, "finish_reason": finish}}}
	}
	usage := func(prompt, gen int) map[string]interface{} {
		return map[string]interface{}{"prompt_tokens": prompt, "completion_tokens": gen, "total_tokens": prompt + gen}
	}
	toolCallJSON := func(idx int, t *parsedTool) map[string]interface{} {
		return map[string]interface{}{"index": idx, "id": fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), idx), "type": "function",
			"function": map[string]interface{}{"name": t.Name, "arguments": t.Arguments}}
	}

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		flusher, _ := w.(http.Flusher)
		sseWrite(w, flusher, chunk(map[string]interface{}{"role": "assistant", "content": ""}, nil))
		nTools := 0
		for ev := range events {
			switch {
			case ev.Err != nil:
				sseWrite(w, flusher, map[string]interface{}{"error": map[string]interface{}{"message": ev.Err.Error(), "type": "engine_error"}})
				fmt.Fprint(w, "data: [DONE]\n\n")
				return
			case ev.End:
				finReason := ev.Finish
				if nTools > 0 && (finReason == "stop" || finReason == "") {
					finReason = "tool_calls"
				}
				fin := chunk(map[string]interface{}{}, finReason)
				if ev.Finish == "cancel" {
					fin = chunk(map[string]interface{}{}, "stop")
				}
				if !req.StreamOptions.IncludeUsage {
					fin["usage"] = usage(ev.PromptTokens, ev.GenTokens)
				}
				sseWrite(w, flusher, fin)
				if req.StreamOptions.IncludeUsage {
					sseWrite(w, flusher, map[string]interface{}{"id": reqID, "object": "chat.completion.chunk", "created": created, "model": model,
						"choices": []interface{}{}, "usage": usage(ev.PromptTokens, ev.GenTokens)})
				}
				fmt.Fprint(w, "data: [DONE]\n\n")
				if flusher != nil {
					flusher.Flush()
				}
				return
			case ev.Kind == "reasoning":
				sseWrite(w, flusher, chunk(map[string]interface{}{"reasoning_content": ev.Text}, nil))
			case ev.Kind == "content":
				sseWrite(w, flusher, chunk(map[string]interface{}{"content": ev.Text}, nil))
			case ev.Kind == "tool":
				sseWrite(w, flusher, chunk(map[string]interface{}{"tool_calls": []interface{}{toolCallJSON(nTools, ev.Tool)}}, nil))
				nTools++
			}
		}
		return
	}

	var content, reasoning strings.Builder
	var tools []interface{}
	finish, prompt, gen := "stop", 0, 0
	var trace []rtclient.TraceEvent
	for ev := range events {
		switch {
		case ev.Err != nil:
			var ae *apiError
			if errors.As(ev.Err, &ae) {
				writeAPIError(w, ae)
			} else {
				writeAPIError(w, &apiError{Status: 502, Type: "engine_error", Msg: ev.Err.Error()})
			}
			return
		case ev.End:
			finish, prompt, gen, trace = ev.Finish, ev.PromptTokens, ev.GenTokens, ev.Trace
		case ev.Kind == "reasoning":
			reasoning.WriteString(ev.Text)
		case ev.Kind == "content":
			content.WriteString(ev.Text)
		case ev.Kind == "tool":
			tools = append(tools, toolCallJSON(len(tools), ev.Tool))
		}
	}
	if finish == "cancel" {
		finish = "stop"
	}
	if len(tools) > 0 && (finish == "stop" || finish == "") {
		finish = "tool_calls"
	}
	msg := map[string]interface{}{"role": "assistant", "content": content.String()}
	if reasoning.Len() > 0 {
		msg["reasoning_content"] = reasoning.String()
	}
	if len(tools) > 0 {
		msg["tool_calls"] = tools
		if content.Len() == 0 {
			msg["content"] = nil
		}
	}
	resp := map[string]interface{}{
		"id": reqID, "object": "chat.completion", "created": created, "model": model,
		"choices": []map[string]interface{}{{"index": 0, "message": msg, "finish_reason": finish}},
		"usage":   usage(prompt, gen),
	}
	if spec.DebugEcho && len(trace) > 0 {
		resp["strata_trace"] = trace // what the runtime did internally; only on request
	}
	writeJSON(w, http.StatusOK, resp)
}

// ---- /v1/completions (raw prompt, no chat template) ----

func (s *StrataServer) handleCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req oaRequest
	if !readJSONBody(w, r, &req) {
		return
	}
	var prompt string
	var many []string
	if len(req.Prompt) > 0 && json.Unmarshal(req.Prompt, &prompt) != nil {
		if json.Unmarshal(req.Prompt, &many) != nil || len(many) != 1 {
			writeAPIError(w, badRequest("prompt must be a string"))
			return
		}
		prompt = many[0]
	}
	if prompt == "" {
		writeAPIError(w, badRequest("prompt is required"))
		return
	}
	spec := genSpec{Raw: true, RawPrompt: prompt, Scope: req.scope(r)}
	req.applyCommon(s, &spec)
	spec.setAbliterate(r)
	reqID := fmt.Sprintf("cmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()
	ctx, done := s.registerRequest(r.Context(), reqID)
	defer done()
	events, _, err := s.startChat(ctx, spec)
	if err != nil {
		writeAPIError(w, err)
		return
	}
	model := s.Config.ModelName
	if req.Model != "" && s.Registry != nil {
		model = s.Registry.ResolveModel(req.Model, s.Config.ModelName)
	} else if req.Model != "" {
		model = req.Model
	}
	mk := func(text string, finish interface{}) map[string]interface{} {
		return map[string]interface{}{"id": reqID, "object": "text_completion", "created": created, "model": model,
			"choices": []map[string]interface{}{{"index": 0, "text": text, "finish_reason": finish}}}
	}
	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher, _ := w.(http.Flusher)
		for ev := range events {
			switch {
			case ev.Err != nil:
				sseWrite(w, flusher, map[string]interface{}{"error": map[string]interface{}{"message": ev.Err.Error(), "type": "engine_error"}})
				fmt.Fprint(w, "data: [DONE]\n\n")
				return
			case ev.End:
				sseWrite(w, flusher, mk("", ev.Finish))
				fmt.Fprint(w, "data: [DONE]\n\n")
				return
			default:
				sseWrite(w, flusher, mk(ev.Text, nil))
			}
		}
		return
	}
	var sb strings.Builder
	finish, pt, gt := "stop", 0, 0
	for ev := range events {
		switch {
		case ev.Err != nil:
			writeAPIError(w, &apiError{Status: 502, Type: "engine_error", Msg: ev.Err.Error()})
			return
		case ev.End:
			finish, pt, gt = ev.Finish, ev.PromptTokens, ev.GenTokens
		default:
			sb.WriteString(ev.Text)
		}
	}
	out := mk(sb.String(), finish)
	out["usage"] = map[string]interface{}{"prompt_tokens": pt, "completion_tokens": gt, "total_tokens": pt + gt}
	writeJSON(w, http.StatusOK, out)
}
