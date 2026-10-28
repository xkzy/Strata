package chattemplate

import (
	"encoding/json"
	"strings"
)

// flintEscape is the template's flint_text macro: NULs and think tags go, the role control tokens are made harmless.
var flintEscape = strings.NewReplacer("\x00", "", "<think>", "", "</think>", "",
	"<|begin_of_text|>", "&lt;|begin_of_text|&gt;", "<|end_of_text|>", "&lt;|end_of_text|&gt;", "<|system|>", "&lt;|system|&gt;",
	"<|developer|>", "&lt;|developer|&gt;", "<|user|>", "&lt;|user|&gt;", "<|assistant|>", "&lt;|assistant|&gt;",
	"<|observation|>", "&lt;|observation|&gt;")

// renderFlint renders the flint chat template (GLM style: <|user|>text<|assistant|>reply, no end-of-turn marker, no
// content trimming, tools announced before the first non-system message). Checked against Jinja in the tests.
func renderFlint(msgs []Message, opt Options) (string, [][2]int, error) {
	var sb strings.Builder
	var spans [][2]int
	user := func(s string) {
		if s != "" {
			spans = append(spans, [2]int{sb.Len(), sb.Len() + len(s)})
		}
		sb.WriteString(s)
	}
	pending := len(opt.Tools) > 0
	preamble := func() {
		sb.WriteString("<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>")
		for _, tool := range opt.Tools {
			if keys, vals, err := orderedObject(tool); err == nil {
				for i, k := range keys {
					if k == "function" {
						tool = vals[i]
					}
				}
			}
			sb.WriteString("\n")
			user(PyJSON(tool))
		}
		sb.WriteString("\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>\n")
		pending = false
	}
	for _, m := range msgs {
		if pending && m.Role != "system" {
			preamble()
		}
		switch m.Role {
		case "system", "developer", "user":
			sb.WriteString("<|" + m.Role + "|>")
			user(flintEscape.Replace(m.Content))
		case "assistant":
			sb.WriteString("<|assistant|>")
			if m.ReasoningContent != "" {
				sb.WriteString("<think>")
				user(m.ReasoningContent)
				sb.WriteString("</think>")
			}
			user(m.Content)
			for _, tc := range m.ToolCalls {
				sb.WriteString("<tool_call>")
				user(tc.Name)
				keys, vals, err := orderedObject(tc.Arguments)
				if err != nil {
					return "", nil, fail("Tool call arguments for function \"%s\" must be an object/mapping.", tc.Name)
				}
				for i, k := range keys {
					sb.WriteString("<arg_key>")
					user(k)
					sb.WriteString("</arg_key><arg_value>")
					var s string
					if len(vals[i]) > 0 && vals[i][0] == '"' && json.Unmarshal(vals[i], &s) == nil {
						user(s)
					} else {
						user(PyJSON(vals[i]))
					}
					sb.WriteString("</arg_value>")
				}
				sb.WriteString("</tool_call>")
			}
		case "tool":
			sb.WriteString("<|observation|><tool_response>")
			user(flintEscape.Replace(m.Content))
			sb.WriteString("</tool_response>")
		}
	}
	if pending {
		preamble()
	}
	if opt.AddGenerationPrompt {
		sb.WriteString("<|assistant|>")
	}
	return sb.String(), spans, nil
}
