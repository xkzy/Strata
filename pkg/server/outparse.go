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
		case 0: // reasoning until </think>
			if i := strings.Index(p.buf, "</think>"); i >= 0 {
				seg := strings.TrimRightFunc(p.buf[:i], unicode.IsSpace)
				p.emit(&out, "reasoning", seg)
				p.buf = p.buf[i+len("</think>"):]
				p.mode, p.trimLead = 1, true
				continue
			}
			hold := holdSuffix(p.buf, "</think>")
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
			if i := strings.Index(p.buf, "<tool_call>"); i >= 0 {
				p.emit(&out, "content", strings.TrimRightFunc(p.buf[:i], unicode.IsSpace))
				p.buf = p.buf[i+len("<tool_call>"):]
				p.mode = 2
				continue
			}
			hold := holdSuffix(p.buf, "<tool_call>")
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
				if t := p.parseTool(body); t != nil {
					out = append(out, parsedEvent{Kind: "tool", Tool: t})
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

// parseTool reads the body of a <tool_call>. nil if it is not in the template's form.
func (p *outParser) parseTool(body string) *parsedTool {
	i := strings.Index(body, "<function=")
	if i < 0 {
		return nil
	}
	rest := body[i+len("<function="):]
	j := strings.Index(rest, ">")
	if j < 0 {
		return nil
	}
	name := strings.TrimSpace(rest[:j])
	rest = rest[j+1:]
	if k := strings.LastIndex(rest, "</function>"); k >= 0 {
		rest = rest[:k]
	}
	var sb strings.Builder
	sb.WriteString("{")
	n := 0
	for {
		a := strings.Index(rest, "<parameter=")
		if a < 0 {
			break
		}
		rest = rest[a+len("<parameter="):]
		b := strings.Index(rest, ">")
		if b < 0 {
			return nil
		}
		pname := strings.TrimSpace(rest[:b])
		rest = rest[b+1:]
		var val string
		if e := strings.Index(rest, "</parameter>"); e >= 0 {
			val, rest = rest[:e], rest[e+len("</parameter>"):]
		} else {
			val, rest = rest, ""
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
	return &parsedTool{Name: name, Arguments: sb.String()}
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
