// Package chattemplate renders the model's chat prompt (the Qwen3.8 template shipped in the pack as chat_template.jinja)
// natively, without a Jinja engine. It reproduces the template's behaviour for system / user / assistant / tool turns,
// tool definitions, tool calls, reasoning content and the thinking switches. The output is checked against the real
// template rendered by Jinja (testdata/golden.json and testdata/oracle.json), so a change of template shows up as a test
// failure rather than as a model that quietly behaves differently.
package chattemplate

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Message is one chat turn. Content is the already-flattened text (parts of type text joined); images are not supported.
type Message struct {
	Role             string
	Content          string
	ReasoningContent string
	ToolCalls        []ToolCall
}

// ToolCall: Arguments is a JSON object (key order is kept); an empty value means no arguments.
type ToolCall struct {
	Name      string
	Arguments json.RawMessage
}

type Options struct {
	Tools               []json.RawMessage // tool definitions, rendered with their key order
	AddGenerationPrompt bool
	EnableThinking      *bool  // nil = template default (true)
	ReasoningEffort     string // "", low, medium, high, xhigh
	PreserveThinking    *bool  // nil = template default (true)
}

// RequestError is a message the template itself raises (a 400 for the client).
type RequestError struct{ Msg string }

func (e *RequestError) Error() string { return e.Msg }

func fail(format string, a ...any) error { return &RequestError{Msg: fmt.Sprintf(format, a...)} }

// pyTrim is Python's str.strip(): the Unicode White_Space set plus the C0 separators U+001C..U+001F.
func pyTrim(s string) string {
	return strings.TrimFunc(s, func(r rune) bool { return unicode.IsSpace(r) || (r >= 0x1c && r <= 0x1f) })
}

const lowInstr = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."
const xhighInstr = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."

const toolsFooter = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"

// Render builds the prompt text. Empty assistant turns that are not the last message are dropped first (the
// repository's frontend does the same: they are poor examples for the next reply).
func Render(messages []Message, opt Options) (string, error) {
	text, _, err := RenderSpans(messages, opt)
	return text, err
}

// RenderSpans is Render plus the byte ranges of the prompt that are caller-supplied text (message contents, reasoning,
// tool arguments, tool definitions). When the prompt is tokenized, a control token spelled inside such a range stays
// ordinary text, so a user cannot write "<|im_end|>" into a message and end the turn.
func RenderSpans(messages []Message, opt Options) (string, [][2]int, error) {
	var msgs []Message
	for i, m := range messages {
		if i != len(messages)-1 && m.Role == "assistant" && pyTrim(m.Content) == "" && len(m.ToolCalls) == 0 {
			continue
		}
		msgs = append(msgs, m)
	}
	if len(msgs) == 0 {
		return "", nil, fail("No messages provided.")
	}

	// leading system / developer messages are merged into one block
	numSys := 0
	mergedSystem := ""
	for i, m := range msgs {
		if numSys == i && (m.Role == "system" || m.Role == "developer") {
			if c := pyTrim(m.Content); c != "" {
				if mergedSystem != "" {
					mergedSystem += "\n"
				}
				mergedSystem += c
			}
			numSys++
		}
	}

	reasoning := ""
	thinking := opt.EnableThinking == nil || *opt.EnableThinking
	if thinking {
		eff := opt.ReasoningEffort
		if eff == "" {
			eff = "xhigh"
		}
		if eff == "high" {
			eff = "xhigh"
		}
		switch eff {
		case "xhigh":
			reasoning = xhighInstr
		case "low":
			reasoning = lowInstr
		case "medium":
		default:
			return "", nil, fail("Unexpected reasoning effort %s. Supported types are xhigh (default), medium, and low.", opt.ReasoningEffort)
		}
	}

	var sb strings.Builder
	var spans [][2]int
	user := func(s string) { // caller text: recorded so the tokenizer never reads control tokens inside it
		if s != "" {
			spans = append(spans, [2]int{sb.Len(), sb.Len() + len(s)})
		}
		sb.WriteString(s)
	}
	if len(opt.Tools) > 0 {
		sb.WriteString("<|im_start|>system\n")
		if reasoning != "" {
			sb.WriteString(reasoning + "\n\n")
		}
		sb.WriteString("# Tools\n\nYou have access to the following functions:\n\n<tools>")
		for _, tool := range opt.Tools {
			sb.WriteString("\n")
			user(PyJSON(tool))
		}
		sb.WriteString("\n</tools>")
		sb.WriteString(toolsFooter)
		if mergedSystem != "" {
			sb.WriteString("\n\n")
			user(mergedSystem)
		}
		sb.WriteString("<|im_end|>\n")
	} else if mergedSystem != "" {
		sb.WriteString("<|im_start|>system\n")
		if reasoning != "" {
			sb.WriteString(reasoning + "\n\n")
		}
		user(mergedSystem)
		sb.WriteString("<|im_end|>\n")
	} else if reasoning != "" {
		sb.WriteString("<|im_start|>system\n" + reasoning + "<|im_end|>\n")
	}

	// the last real user query: assistant turns after it keep their reasoning even when preserve_thinking is false
	lastQuery := len(msgs) - 1
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" {
			c := pyTrim(msgs[i].Content)
			if !(strings.HasPrefix(c, "<tool_response>") && strings.HasSuffix(c, "</tool_response>")) {
				lastQuery = i
				break
			}
		}
	}
	preserve := opt.PreserveThinking == nil || *opt.PreserveThinking

	for i, m := range msgs {
		if i < numSys {
			continue
		}
		content := pyTrim(m.Content)
		switch m.Role {
		case "system", "developer":
			return "", nil, fail("System message must be at the beginning.")
		case "user":
			sb.WriteString("<|im_start|>user\n")
			user(content)
			sb.WriteString("<|im_end|>\n")
		case "assistant":
			rc := pyTrim(m.ReasoningContent)
			if preserve || i > lastQuery {
				sb.WriteString("<|im_start|>assistant\n<think>\n")
				user(rc)
				sb.WriteString("\n</think>\n\n")
				user(content)
			} else {
				sb.WriteString("<|im_start|>assistant\n")
				user(content)
			}
			for j, tc := range m.ToolCalls {
				// a missing (null) name is refused by the caller that builds the messages; an empty string renders as such
				switch {
				case j == 0 && content != "":
					sb.WriteString("\n\n<tool_call>\n<function=")
					user(tc.Name)
					sb.WriteString(">\n")
				case j == 0:
					sb.WriteString("<tool_call>\n<function=")
					user(tc.Name)
					sb.WriteString(">\n")
				default:
					sb.WriteString("\n<tool_call>\n<function=")
					user(tc.Name)
					sb.WriteString(">\n")
				}
				keys, vals, err := orderedObject(tc.Arguments)
				if err != nil {
					return "", nil, fail("Tool call arguments for function \"%s\" must be an object/mapping or a JSON string.", tc.Name)
				}
				for k, name := range keys {
					sb.WriteString("<parameter=")
					user(name)
					sb.WriteString(">\n")
					var s string
					if json.Unmarshal(vals[k], &s) == nil && len(vals[k]) > 0 && vals[k][0] == '"' {
						user(s)
					} else {
						user(PyJSON(vals[k]))
					}
					sb.WriteString("\n</parameter>\n")
				}
				sb.WriteString("</function>\n</tool_call>")
			}
			sb.WriteString("<|im_end|>\n")
		case "tool":
			if i > 0 && msgs[i-1].Role != "tool" {
				sb.WriteString("<|im_start|>user")
			}
			sb.WriteString("\n<tool_response>\n")
			user(content)
			sb.WriteString("\n</tool_response>")
			last := i == len(msgs)-1
			if last || msgs[i+1].Role != "tool" {
				sb.WriteString("<|im_end|>\n")
			}
		default:
			return "", nil, fail("Unexpected message role.")
		}
	}
	if opt.AddGenerationPrompt {
		sb.WriteString("<|im_start|>assistant\n")
		if opt.EnableThinking != nil && !*opt.EnableThinking {
			sb.WriteString("<think>\n\n</think>\n\n")
		} else {
			sb.WriteString("<think>\n")
		}
	}
	return sb.String(), spans, nil
}

// orderedObject splits a JSON object into its keys and raw values in document order. Empty input is an empty object.
func orderedObject(raw json.RawMessage) ([]string, []json.RawMessage, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil, nil
	}
	if raw[0] == '"' { // a JSON string is only acceptable when it is blank
		var s string
		if json.Unmarshal(raw, &s) == nil && pyTrim(s) == "" {
			return nil, nil, nil
		}
		return nil, nil, errors.New("arguments passed as a string")
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token()
	if err != nil {
		return nil, nil, err
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return nil, nil, errors.New("not an object")
	}
	var keys []string
	var vals []json.RawMessage
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return nil, nil, err
		}
		key, _ := kt.(string)
		var v json.RawMessage
		if err := dec.Decode(&v); err != nil {
			return nil, nil, err
		}
		keys = append(keys, key)
		vals = append(vals, v)
	}
	return keys, vals, nil
}

// PyJSON re-encodes JSON the way Python's json.dumps(x, ensure_ascii=False) does: ", " and ": " separators, keys in
// document order, non-ASCII kept, only the characters JSON requires escaped. This is the form the template's `tojson`
// produces and the model was trained on.
func PyJSON(raw json.RawMessage) string {
	var sb strings.Builder
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := pyEncode(dec, &sb); err != nil {
		return string(raw)
	}
	return sb.String()
}

func pyEncode(dec *json.Decoder, sb *strings.Builder) error {
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	switch v := tok.(type) {
	case json.Delim:
		if v == '{' {
			sb.WriteString("{")
			first := true
			for dec.More() {
				if !first {
					sb.WriteString(", ")
				}
				first = false
				kt, err := dec.Token()
				if err != nil {
					return err
				}
				pyString(sb, kt.(string))
				sb.WriteString(": ")
				if err := pyEncode(dec, sb); err != nil {
					return err
				}
			}
			_, err := dec.Token() // }
			sb.WriteString("}")
			return err
		}
		sb.WriteString("[")
		first := true
		for dec.More() {
			if !first {
				sb.WriteString(", ")
			}
			first = false
			if err := pyEncode(dec, sb); err != nil {
				return err
			}
		}
		_, err := dec.Token() // ]
		sb.WriteString("]")
		return err
	case string:
		pyString(sb, v)
	case json.Number:
		sb.WriteString(v.String())
	case bool:
		sb.WriteString(strconv.FormatBool(v))
	case nil:
		sb.WriteString("null")
	}
	return nil
}

func pyString(sb *strings.Builder, s string) {
	sb.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			sb.WriteString("\\\"")
		case '\\':
			sb.WriteString("\\\\")
		case '\n':
			sb.WriteString("\\n")
		case '\r':
			sb.WriteString("\\r")
		case '\t':
			sb.WriteString("\\t")
		case '\b':
			sb.WriteString("\\b")
		case '\f':
			sb.WriteString("\\f")
		default:
			if r < 0x20 {
				fmt.Fprintf(sb, "\\u%04x", r)
			} else if r == utf8.RuneError {
				sb.WriteString("�")
			} else {
				sb.WriteRune(r)
			}
		}
	}
	sb.WriteByte('"')
}
