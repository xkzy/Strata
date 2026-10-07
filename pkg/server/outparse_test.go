package server

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func feedAll(p *outParser, chunks []string) (reasoning, content string, tools []parsedTool) {
	handle := func(evs []parsedEvent) {
		for _, e := range evs {
			switch e.Kind {
			case "reasoning":
				reasoning += e.Text
			case "content":
				content += e.Text
			case "tool":
				tools = append(tools, *e.Tool)
			}
		}
	}
	for _, c := range chunks {
		handle(p.Feed(c))
	}
	handle(p.Flush())
	return
}

func split(s string, n int) []string {
	var out []string
	for i := 0; i < len(s); i += n {
		out = append(out, s[i:min(i+n, len(s))])
	}
	return out
}

func TestReasoningThenContent(t *testing.T) {
	stream := "The user says hi.\nI should greet.\n</think>\n\nHello! How can I help?"
	for _, n := range []int{1, 2, 3, 7, 1000} {
		r, c, tl := feedAll(newOutParser(true, nil), split(stream, n))
		if r != "The user says hi.\nI should greet." || c != "Hello! How can I help?" || len(tl) != 0 {
			t.Fatalf("chunk %d: reasoning=%q content=%q tools=%v", n, r, c, tl)
		}
	}
}

func TestThinkingDisabledStartsInContent(t *testing.T) {
	r, c, _ := feedAll(newOutParser(false, nil), split("Hello there.", 2))
	if r != "" || c != "Hello there." {
		t.Fatalf("%q %q", r, c)
	}
}

func TestAHalfTagIsNeverEmitted(t *testing.T) {
	p := newOutParser(false, nil)
	var seen strings.Builder
	for _, ch := range split("See <tool_call>\n<function=f>\n</function>\n</tool_call>", 1) {
		for _, e := range p.Feed(ch) {
			if e.Kind == "content" {
				if strings.Contains(e.Text, "<") && !strings.Contains(e.Text, "<tool_call>") {
					// a lone "<" must not leak before we know it is not a tag
					t.Fatalf("leaked a partial tag: %q", e.Text)
				}
				seen.WriteString(e.Text)
			}
		}
	}
	if seen.String() != "See" {
		t.Fatalf("content before the tool call = %q", seen.String())
	}
}

func TestToolCallParsing(t *testing.T) {
	tools := []json.RawMessage{json.RawMessage(`{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"zip":{"type":"string"}}}}}`)}
	stream := "</think>\n\nLet me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=zip>\n75001\n</parameter>\n</function>\n</tool_call>"
	for _, n := range []int{1, 5, 1000} {
		_, c, tl := feedAll(newOutParser(true, tools), split(stream, n))
		if c != "Let me check." || len(tl) != 1 {
			t.Fatalf("chunk %d: content=%q tools=%v", n, c, tl)
		}
		if tl[0].Name != "get_weather" || tl[0].Arguments != `{"city":"Paris","days":3,"zip":"75001"}` {
			t.Fatalf("tool = %+v", tl[0])
		}
	}
}

func TestTwoToolCallsAndMultilineValue(t *testing.T) {
	stream := "<tool_call>\n<function=a>\n<parameter=code>\nline1\nline2\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n</function>\n</tool_call>"
	_, c, tl := feedAll(newOutParser(false, nil), split(stream, 3))
	if c != "" || len(tl) != 2 {
		t.Fatalf("content=%q tools=%v", c, tl)
	}
	if !reflect.DeepEqual(tl[0].Arguments, `{"code":"line1\nline2"}`) || tl[1].Arguments != `{}` {
		t.Fatalf("%+v", tl)
	}
}

func TestUnfinishedToolCallIsNotLost(t *testing.T) {
	_, c, tl := feedAll(newOutParser(false, nil), split("ok <tool_call>\n<function=a>\n<param", 4))
	if len(tl) != 0 || !strings.Contains(c, "<tool_call>") || !strings.HasPrefix(c, "ok") {
		t.Fatalf("content=%q tools=%v", c, tl)
	}
}

func TestNonStringValuesKeepTheirJSONType(t *testing.T) {
	stream := "<tool_call>\n<function=f>\n<parameter=flag>\ntrue\n</parameter>\n<parameter=list>\n[1, 2]\n</parameter>\n<parameter=text>\nhello world\n</parameter>\n</function>\n</tool_call>"
	_, _, tl := feedAll(newOutParser(false, nil), split(stream, 6))
	if tl[0].Arguments != `{"flag":true,"list":[1, 2],"text":"hello world"}` {
		t.Fatalf("%s", tl[0].Arguments)
	}
}
