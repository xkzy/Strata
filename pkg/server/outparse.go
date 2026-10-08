package server

import (
	"encoding/json"
	"strings"
	"unicode"
)

// The model answers in one stream: reasoning, then `</think>`, then the reply, which may contain tool calls in the
// template's XML form
//
//	<tool_call>
//	<function=NAME>
//	<parameter=P>
//	VALUE
//	</parameter>
//	</function>
//	</tool_call>
//
// outParser splits that stream into reasoning / content / tool-call events incrementally. It never emits half of a tag.

type parsedEvent struct {
	Kind string // "reasoning", "content", "tool"
	Text string
	Tool *parsedTool
}

type parsedTool struct {
	Name      string
	Arguments string // a JSON object
}

type outParser struct {
	mode         int // 0 reasoning, 1 content, 2 inside a tool call
	buf          string
	trimLead     bool
	paramTypes   map[string]map[string]string // tool -> parameter -> JSON schema type
	sawToolCall  bool
	emittedAny   bool
	reasoningLen int
}

func newOutParser(startInReasoning bool, tools []json.RawMessage) *outParser {
	p := &outParser{paramTypes: map[string]map[string]string{}, trimLead: !startInReasoning}
	if !startInReasoning {
		p.mode = 1
	}
	for _, raw := range tools {
		var t struct {
			Function struct {
				Name       string `json:"name"`
				Parameters struct {
					Properties map[string]struct {
						Type any `json:"type"`
					} `json:"properties"`
				} `json:"parameters"`
			} `json:"function"`
		}
		if json.Unmarshal(raw, &t) != nil || t.Function.Name == "" {
			continue
		}
		m := map[string]string{}
		for k, v := range t.Function.Parameters.Properties {
			if s, ok := v.Type.(string); ok {
				m[k] = s
			}
		}
		p.paramTypes[t.Function.Name] = m
	}
	return p
}

// holdSuffix returns how many trailing bytes of s could be the start of tag (must be held back).
func holdSuffix(s, tag string) int {
	for k := min(len(tag)-1, len(s)); k > 0; k-- {
		if strings.HasSuffix(s, tag[:k]) {
			return k
		}
	}
	return 0
}

func trailingSpace(s string) int {
	n := len(s)
	t := strings.TrimRightFunc(s, unicode.IsSpace)
	return n - len(t)
}

func (p *outParser) Feed(text string) []parsedEvent {
	p.buf += text
	return p.drain(false)
}

// Flush ends the stream: whatever is held is released (an unfinished tool call comes out as plain text).
func (p *outParser) Flush() []parsedEvent { return p.drain(true) }

func (p *outParser) emit(out *[]parsedEvent, kind, text string) {
	if text == "" {
		return
	}
	*out = append(*out, parsedEvent{Kind: kind, Text: text})
	p.emittedAny = true
}

func (p *outParser) drain(final bool) []parsedEvent {
	var out []parsedEvent
	for {
		switch p.mode {
		case 0: // reasoning until </think> or </thought>
			endTag := ""
			idx := -1
			if i := strings.Index(p.buf, "</think>"); i >= 0 {
				idx = i
				endTag = "</think>"
			} else if i := strings.Index(p.buf, "</thought>"); i >= 0 {
				idx = i
				endTag = "</thought>"
			}
			if idx >= 0 {
				seg := strings.TrimRightFunc(p.buf[:idx], unicode.IsSpace)
				p.emit(&out, "reasoning", seg)
				p.buf = p.buf[idx+len(endTag):]
				p.mode, p.trimLead = 1, true
				continue
			}
			hold := max(holdSuffix(p.buf, "</think>"), holdSuffix(p.buf, "</thought>"))
			rest := p.buf[:len(p.buf)-hold]
			if !final { // trailing whitespace may be dropped when </think> follows
				hold += trailingSpace(rest)
			}
			if final {
				hold = 0
			}
			p.emit(&out, "reasoning", p.buf[:len(p.buf)-hold])
			p.buf = p.buf[len(p.buf)-hold:]
			return out
		case 1: // reply
			if p.trimLead {
				p.buf = strings.TrimLeftFunc(p.buf, unicode.IsSpace)
				if p.buf == "" && !final {
					return out
				}
				if p.buf != "" {
					p.trimLead = false
				}
			}
			// Catch any <think> or <thought> emitted mid-stream or when starting in content mode
			if i := strings.Index(p.buf, "<think>"); i >= 0 {
				p.emit(&out, "content", strings.TrimRightFunc(p.buf[:i], unicode.IsSpace))
				p.buf = p.buf[i+len("<think>"):]
				p.mode = 0
				continue
			}
			if i := strings.Index(p.buf, "<thought>"); i >= 0 {
				p.emit(&out, "content", strings.TrimRightFunc(p.buf[:i], unicode.IsSpace))
				p.buf = p.buf[i+len("<thought>"):]
				p.mode = 0
				continue
			}
			if i := strings.Index(p.buf, "<tool_call>"); i >= 0 {
				p.emit(&out, "content", strings.TrimRightFunc(p.buf[:i], unicode.IsSpace))
				p.buf = p.buf[i+len("<tool_call>"):]
				p.mode = 2
				continue
			}
			hold := max(holdSuffix(p.buf, "<tool_call>"), holdSuffix(p.buf, "<think>"), holdSuffix(p.buf, "<thought>"))
			if !final {
				hold += trailingSpace(p.buf[:len(p.buf)-hold])
			} else {
				hold = 0
			}
			p.emit(&out, "content", p.buf[:len(p.buf)-hold])
			p.buf = p.buf[len(p.buf)-hold:]
			return out
		case 2: // inside <tool_call> ... </tool_call>
			if i := strings.Index(p.buf, "</tool_call>"); i >= 0 {
				body := p.buf[:i]
				p.buf = p.buf[i+len("</tool_call>"):]
				if tools := p.parseTools(body); len(tools) > 0 {
					for _, t := range tools {
						out = append(out, parsedEvent{Kind: "tool", Tool: t})
					}
					p.sawToolCall, p.emittedAny = true, true
				} else {
					p.emit(&out, "content", "<tool_call>"+body+"</tool_call>")
				}
				p.mode, p.trimLead = 1, true
				continue
			}
			if final {
				p.emit(&out, "content", "<tool_call>"+p.buf)
				p.buf = ""
			}
			return out
		}
	}
}

// parseTool reads the body of a <tool_call>. Returns the first parsed tool or nil.
func (p *outParser) parseTool(body string) *parsedTool {
	tools := p.parseTools(body)
	if len(tools) > 0 {
		return tools[0]
	}
	return nil
}

// parseTools parses tool calls from XML <function=...> or JSON {"name": ...} bodies.
func (p *outParser) parseTools(body string) []*parsedTool {
	body = strings.TrimSpace(body)
	if body == "" {
		return nil
	}

	// Strip optional markdown code fences
	if strings.HasPrefix(body, "```") {
		lines := strings.Split(body, "\n")
		if len(lines) >= 2 && strings.HasPrefix(lines[0], "```") {
			body = strings.Join(lines[1:], "\n")
			if idx := strings.LastIndex(body, "```"); idx >= 0 {
				body = body[:idx]
			}
			body = strings.TrimSpace(body)
		}
	}

	// 1. JSON Array of tool calls: [{"name": ...}, ...]
	if strings.HasPrefix(body, "[") {
		var list []json.RawMessage
		if err := json.Unmarshal([]byte(body), &list); err == nil {
			var res []*parsedTool
			for _, item := range list {
				if t := p.parseSingleJSONTool(string(item)); t != nil {
					res = append(res, t)
				}
			}
			if len(res) > 0 {
				return res
			}
		}
	}

	// 2. Single JSON Object: {"name": ..., "arguments": ...}
	if strings.HasPrefix(body, "{") {
		if t := p.parseSingleJSONTool(body); t != nil {
			return []*parsedTool{t}
		}
	}

	// 3. XML format: <function=NAME>... (supports one or multiple function tags)
	var res []*parsedTool
	rest := body
	for {
		i := strings.Index(rest, "<function=")
		if i < 0 {
			break
		}
		rest = rest[i+len("<function="):]
		j := strings.Index(rest, ">")
		if j < 0 {
			break
		}
		name := strings.TrimSpace(rest[:j])
		rest = rest[j+1:]
		var funcBody string
		if k := strings.Index(rest, "</function>"); k >= 0 {
			funcBody = rest[:k]
			rest = rest[k+len("</function>"):]
		} else {
			funcBody = rest
			rest = ""
		}

		var sb strings.Builder
		sb.WriteString("{")
		n := 0
		paramsRest := funcBody
		for {
			a := strings.Index(paramsRest, "<parameter=")
			if a < 0 {
				break
			}
			paramsRest = paramsRest[a+len("<parameter="):]
			b := strings.Index(paramsRest, ">")
			if b < 0 {
				break
			}
			pname := strings.TrimSpace(paramsRest[:b])
			paramsRest = paramsRest[b+1:]
			var val string
			if e := strings.Index(paramsRest, "</parameter>"); e >= 0 {
				val, paramsRest = paramsRest[:e], paramsRest[e+len("</parameter>"):]
			} else {
				val, paramsRest = paramsRest, ""
			}
			val = strings.TrimPrefix(val, "\n")
			val = strings.TrimSuffix(val, "\n")
			if n > 0 {
				sb.WriteString(",")
			}
			kb, _ := json.Marshal(pname)
			sb.Write(kb)
			sb.WriteString(":")
			sb.Write(p.paramJSON(name, pname, val))
			n++
		}
		sb.WriteString("}")
		res = append(res, &parsedTool{Name: name, Arguments: sb.String()})
	}

	return res
}

// parseSingleJSONTool extracts tool name and arguments from JSON format.
func (p *outParser) parseSingleJSONTool(body string) *parsedTool {
	var raw struct {
		Name        string          `json:"name"`
		Function    json.RawMessage `json:"function"`
		Action      string          `json:"action"`
		Arguments   json.RawMessage `json:"arguments"`
		Parameters  json.RawMessage `json:"parameters"`
		ActionInput json.RawMessage `json:"action_input"`
		Input       json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal([]byte(body), &raw); err != nil {
		return nil
	}

	name := raw.Name
	if name == "" && len(raw.Function) > 0 {
		var fn struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if json.Unmarshal(raw.Function, &fn) == nil && fn.Name != "" {
			name = fn.Name
			if len(fn.Arguments) > 0 {
				raw.Arguments = fn.Arguments
			}
		} else {
			var s string
			if json.Unmarshal(raw.Function, &s) == nil {
				name = s
			}
		}
	}
	if name == "" {
		name = raw.Action
	}
	if name == "" {
		return nil
	}

	argsRaw := raw.Arguments
	if len(argsRaw) == 0 || string(argsRaw) == "null" {
		argsRaw = raw.Parameters
	}
	if len(argsRaw) == 0 || string(argsRaw) == "null" {
		argsRaw = raw.ActionInput
	}
	if len(argsRaw) == 0 || string(argsRaw) == "null" {
		argsRaw = raw.Input
	}

	argsStr := "{}"
	if len(argsRaw) > 0 && string(argsRaw) != "null" {
		trimmed := strings.TrimSpace(string(argsRaw))
		if strings.HasPrefix(trimmed, "\"") {
			var s string
			if json.Unmarshal(argsRaw, &s) == nil {
				argsStr = s
			} else {
				argsStr = trimmed
			}
		} else {
			argsStr = trimmed
		}
	}

	var testObj map[string]interface{}
	if err := json.Unmarshal([]byte(argsStr), &testObj); err != nil {
		var testAny interface{}
		if json.Unmarshal([]byte(argsStr), &testAny) == nil {
			wrapped, _ := json.Marshal(map[string]interface{}{"input": testAny})
			argsStr = string(wrapped)
		} else {
			argsStr = "{}"
		}
	}

	return &parsedTool{Name: name, Arguments: argsStr}
}

// paramJSON turns a parameter's text into JSON: strings stay strings, other values are decoded when they parse.
func (p *outParser) paramJSON(tool, param, val string) []byte {
	if typ := p.paramTypes[tool][param]; typ == "string" {
		b, _ := json.Marshal(val)
		return b
	}
	var v any
	dec := json.NewDecoder(strings.NewReader(strings.TrimSpace(val)))
	dec.UseNumber()
	if err := dec.Decode(&v); err == nil && !dec.More() {
		if _, isStr := v.(string); !isStr || p.paramTypes[tool][param] != "" {
			return []byte(strings.TrimSpace(val))
		}
	}
	b, _ := json.Marshal(val)
	return b
}
