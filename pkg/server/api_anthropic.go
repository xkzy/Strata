package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"strata/pkg/chattemplate"
)

type anBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	Thinking  string          `json:"thinking"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"`
}

type anMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type anRequest struct {
	Model         string          `json:"model"`
	System        json.RawMessage `json:"system"`
	Messages      []anMessage     `json:"messages"`
	MaxTokens     int             `json:"max_tokens"`
	Stream        bool            `json:"stream"`
	Temperature   *float64        `json:"temperature"`
	TopP          *float64        `json:"top_p"`
	TopK          *int            `json:"top_k"`
	StopSequences []string        `json:"stop_sequences"`
	Tools         []struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		InputSchema json.RawMessage `json:"input_schema"`
	} `json:"tools"`
	ToolChoice json.RawMessage `json:"tool_choice"`
	Metadata   struct {
		UserID string `json:"user_id"`
	} `json:"metadata"`
	Thinking *struct {
		Type         string `json:"type"`
		BudgetTokens int    `json:"budget_tokens"`
	} `json:"thinking"`
}

// anBlocks reads Anthropic content: a string or a list of blocks.
func anBlocks(raw json.RawMessage) ([]anBlock, error) {
	raw = json.RawMessage(strings.TrimSpace(string(raw)))
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, badRequest("invalid message content")
		}
		return []anBlock{{Type: "text", Text: s}}, nil
	}
	var bs []anBlock
	if err := json.Unmarshal(raw, &bs); err != nil {
		return nil, badRequest("message content must be a string or a list of content blocks")
	}
	return bs, nil
}

func blocksText(bs []anBlock) (string, error) {
	var sb strings.Builder
	for _, b := range bs {
		switch b.Type {
		case "text", "":
			sb.WriteString(b.Text)
		case "image":
			return "", badRequest("image input is not supported: this model is running without a vision path")
		}
	}
	return sb.String(), nil
}

func (r *anRequest) toSpec(s *StrataServer) (genSpec, error) {
	var spec genSpec
	if len(r.System) > 0 && string(r.System) != "null" {
		bs, err := anBlocks(r.System)
		if err != nil {
			return spec, err
		}
		if text, _ := blocksText(bs); strings.TrimSpace(text) != "" {
			spec.Messages = append(spec.Messages, chattemplate.Message{Role: "system", Content: text})
		}
	}
	for _, m := range r.Messages {
		bs, err := anBlocks(m.Content)
		if err != nil {
			return spec, err
		}
		switch m.Role {
		case "assistant":
			cm := chattemplate.Message{Role: "assistant"}
			var text strings.Builder
			for _, b := range bs {
				switch b.Type {
				case "text":
					text.WriteString(b.Text)
				case "thinking":
					cm.ReasoningContent += b.Thinking
				case "tool_use":
					args, err := toolArguments(b.Input)
					if err != nil {
						return spec, err
					}
					cm.ToolCalls = append(cm.ToolCalls, chattemplate.ToolCall{Name: b.Name, Arguments: args})
				}
			}
			cm.Content = text.String()
			spec.Messages = append(spec.Messages, cm)
		default: // user: tool results become tool messages, the rest is the user's text
			var rest []anBlock
			for _, b := range bs {
				if b.Type == "tool_result" {
					inner, err := anBlocks(b.Content)
					if err != nil {
						return spec, err
					}
					text, err := blocksText(inner)
					if err != nil {
						return spec, err
					}
					spec.Messages = append(spec.Messages, chattemplate.Message{Role: "tool", Content: text})
				} else {
					rest = append(rest, b)
				}
			}
			if len(rest) > 0 || len(bs) == 0 {
				text, err := blocksText(rest)
				if err != nil {
					return spec, err
				}
				spec.Messages = append(spec.Messages, chattemplate.Message{Role: "user", Content: text})
			}
		}
	}
	if len(r.Messages) == 0 {
		return spec, badRequest("messages: at least one message is required")
	}
	if r.MaxTokens <= 0 {
		return spec, badRequest("max_tokens: a positive integer is required")
	}
	if strings.Contains(string(r.ToolChoice), `"none"`) {
		r.Tools = nil
	}
	for _, t := range r.Tools {
		def := map[string]interface{}{"type": "function", "function": map[string]interface{}{
			"name": t.Name, "description": t.Description, "parameters": json.RawMessage(orObject(t.InputSchema))}}
		b, _ := json.Marshal(def)
		spec.Tools = append(spec.Tools, b)
	}
	spec.Opt = chattemplate.Options{Tools: spec.Tools, AddGenerationPrompt: true}
	if r.Thinking != nil {
		switch r.Thinking.Type {
		case "disabled":
			off := false
			spec.Opt.EnableThinking = &off
		case "enabled":
			switch {
			case r.Thinking.BudgetTokens > 0 && r.Thinking.BudgetTokens < 2048:
				spec.Opt.ReasoningEffort = "low"
			case r.Thinking.BudgetTokens > 0 && r.Thinking.BudgetTokens < 8192:
				spec.Opt.ReasoningEffort = "medium"
			}
		}
	}
	spec.Sampling = s.defaultSampling()
	if r.Temperature != nil {
		spec.Sampling.Temperature = *r.Temperature
	}
	if r.TopP != nil {
		spec.Sampling.TopP = *r.TopP
	}
	if r.TopK != nil {
		spec.Sampling.TopK = *r.TopK
	}
	spec.MaxTokens = r.MaxTokens
	spec.Stops = r.StopSequences
	return spec, nil
}

func orObject(raw json.RawMessage) []byte {
	if len(raw) == 0 || string(raw) == "null" {
		return []byte(`{"type":"object","properties":{}}`)
	}
	return raw
}

func anStopReason(finish string) string {
	switch finish {
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	}
	return "end_turn"
}

func anError(w http.ResponseWriter, err error) {
	ae, ok := err.(*apiError)
	if !ok {
		ae = &apiError{Status: 500, Type: "server_error", Msg: err.Error()}
	}
	typ := "invalid_request_error"
	switch {
	case ae.Status == 503:
		typ = "overloaded_error"
		w.Header().Set("Retry-After", "5")
	case ae.Status >= 500:
		typ = "api_error"
	}
	writeJSON(w, ae.Status, map[string]interface{}{"type": "error", "error": map[string]interface{}{"type": typ, "message": ae.Msg}})
}

func (s *StrataServer) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req anRequest
	if !readJSONBody(w, r, &req) {
		return
	}
	spec, err := req.toSpec(s)
	if err != nil {
		anError(w, err)
		return
	}
	spec.Scope = requestScope(r, req.Metadata.UserID, "")
	if h := r.Header.Get("X-Strata-Math-Engine"); h == "off" || h == "false" || h == "0" {
		spec.DisableMath = true
	}
	msgID := fmt.Sprintf("msg_%d", time.Now().UnixNano())
	ctx, done := s.registerRequest(r.Context(), msgID)
	defer done()
	events, promptTokens, err := s.startChat(ctx, spec)
	if err != nil {
		anError(w, err)
		return
	}
	model := req.Model
	if model == "" {
		model = s.Config.ModelName
	}

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		flusher, _ := w.(http.Flusher)
		send := func(event string, payload map[string]interface{}) {
			b, _ := json.Marshal(payload)
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
			if flusher != nil {
				flusher.Flush()
			}
		}
		send("message_start", map[string]interface{}{"type": "message_start", "message": map[string]interface{}{
			"id": msgID, "type": "message", "role": "assistant", "model": model, "content": []interface{}{},
			"stop_reason": nil, "stop_sequence": nil, "usage": map[string]interface{}{"input_tokens": promptTokens, "output_tokens": 0}}})
		index, open := -1, ""
		closeBlock := func() {
			if open == "" {
				return
			}
			if open == "thinking" {
				send("content_block_delta", map[string]interface{}{"type": "content_block_delta", "index": index, "delta": map[string]interface{}{"type": "signature_delta", "signature": ""}})
			}
			send("content_block_stop", map[string]interface{}{"type": "content_block_stop", "index": index})
			open = ""
		}
		openBlock := func(kind string, block map[string]interface{}) {
			closeBlock()
			index++
			open = kind
			send("content_block_start", map[string]interface{}{"type": "content_block_start", "index": index, "content_block": block})
		}
		for ev := range events {
			switch {
			case ev.Err != nil:
				closeBlock()
				send("error", map[string]interface{}{"type": "error", "error": map[string]interface{}{"type": "api_error", "message": ev.Err.Error()}})
				return
			case ev.End:
				closeBlock()
				send("message_delta", map[string]interface{}{"type": "message_delta",
					"delta": map[string]interface{}{"stop_reason": anStopReason(ev.Finish), "stop_sequence": nil},
					"usage": map[string]interface{}{"output_tokens": ev.GenTokens}})
				send("message_stop", map[string]interface{}{"type": "message_stop"})
				return
			case ev.Kind == "reasoning":
				if open != "thinking" {
					openBlock("thinking", map[string]interface{}{"type": "thinking", "thinking": ""})
				}
				send("content_block_delta", map[string]interface{}{"type": "content_block_delta", "index": index, "delta": map[string]interface{}{"type": "thinking_delta", "thinking": ev.Text}})
			case ev.Kind == "content":
				if open != "text" {
					openBlock("text", map[string]interface{}{"type": "text", "text": ""})
				}
				send("content_block_delta", map[string]interface{}{"type": "content_block_delta", "index": index, "delta": map[string]interface{}{"type": "text_delta", "text": ev.Text}})
			case ev.Kind == "tool":
				openBlock("tool_use", map[string]interface{}{"type": "tool_use", "id": fmt.Sprintf("toolu_%d", time.Now().UnixNano()), "name": ev.Tool.Name, "input": map[string]interface{}{}})
				send("content_block_delta", map[string]interface{}{"type": "content_block_delta", "index": index, "delta": map[string]interface{}{"type": "input_json_delta", "partial_json": ev.Tool.Arguments}})
			}
		}
		return
	}

	var blocks []map[string]interface{}
	appendText := func(kind, key, text string) {
		if n := len(blocks); n > 0 && blocks[n-1]["type"] == kind {
			blocks[n-1][key] = blocks[n-1][key].(string) + text
			return
		}
		blocks = append(blocks, map[string]interface{}{"type": kind, key: text})
	}
	finish, gen := "stop", 0
	for ev := range events {
		switch {
		case ev.Err != nil:
			anError(w, &apiError{Status: 502, Type: "api_error", Msg: ev.Err.Error()})
			return
		case ev.End:
			finish, gen = ev.Finish, ev.GenTokens
		case ev.Kind == "reasoning":
			appendText("thinking", "thinking", ev.Text)
		case ev.Kind == "content":
			appendText("text", "text", ev.Text)
		case ev.Kind == "tool":
			var input interface{}
			_ = json.Unmarshal([]byte(ev.Tool.Arguments), &input)
			blocks = append(blocks, map[string]interface{}{"type": "tool_use", "id": fmt.Sprintf("toolu_%d_%d", time.Now().UnixNano(), len(blocks)), "name": ev.Tool.Name, "input": input})
		}
	}
	for _, b := range blocks {
		if b["type"] == "thinking" {
			b["signature"] = ""
		}
	}
	if blocks == nil {
		blocks = []map[string]interface{}{}
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"id": msgID, "type": "message", "role": "assistant", "model": model, "content": blocks,
		"stop_reason": anStopReason(finish), "stop_sequence": nil,
		"usage": map[string]interface{}{"input_tokens": promptTokens, "output_tokens": gen},
	})
}
