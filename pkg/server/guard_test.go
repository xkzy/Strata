package server

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"strata/pkg/chattemplate"
	"strata/pkg/reqtrace"
)

type traceSink struct {
	mu   sync.Mutex
	logs []string
}

func (t *traceSink) Add(source, msg string) {
	t.mu.Lock()
	t.logs = append(t.logs, source+": "+msg)
	t.mu.Unlock()
}
func (t *traceSink) text() string { t.mu.Lock(); defer t.mu.Unlock(); return strings.Join(t.logs, "\n") }

var weatherTool = json.RawMessage(`{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","required":["city"],"properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}`)

func TestCheckToolCalls(t *testing.T) {
	tools := []json.RawMessage{weatherTool}
	cases := []struct {
		name string
		call *parsedTool
		want string // substring of the problem, "" = well formed
	}{
		{"ok", &parsedTool{Name: "get_weather", Arguments: `{"city":"Oslo","days":3}`}, ""},
		{"no args when none required is fine only if optional", &parsedTool{Name: "get_weather", Arguments: `{"city":"Oslo"}`}, ""},
		{"unknown tool", &parsedTool{Name: "get_wether", Arguments: `{}`}, "does not exist"},
		{"bad json", &parsedTool{Name: "get_weather", Arguments: `{"city": "Oslo"`}, "not a valid JSON object"},
		{"not an object", &parsedTool{Name: "get_weather", Arguments: `["Oslo"]`}, "not a valid JSON object"},
		{"missing required", &parsedTool{Name: "get_weather", Arguments: `{"days":2}`}, `"city" is missing`},
		{"wrong type", &parsedTool{Name: "get_weather", Arguments: `{"city":"Oslo","days":"three"}`}, "must be integer, not string"},
		{"float for integer is accepted as number", &parsedTool{Name: "get_weather", Arguments: `{"city":7}`}, "must be string, not integer"},
	}
	for _, c := range cases {
		got := strings.Join(checkToolCalls(tools, []*parsedTool{c.call}), "; ")
		if (c.want == "") != (got == "") || !strings.Contains(got, c.want) {
			t.Errorf("%s: got %q, want %q", c.name, got, c.want)
		}
	}
	if got := checkToolCalls(nil, []*parsedTool{{Name: "x", Arguments: `{}`}}); len(got) != 1 || !strings.Contains(got[0], "offers no tools") {
		t.Errorf("a tool call without any tools offered: %v", got)
	}
}

// scriptedEngine answers each start with the next scripted reply and records the specs it was started with.
type scriptedEngine struct {
	mu      sync.Mutex
	replies [][]genEvent
	specs   []genSpec
}

func (e *scriptedEngine) start(ctx context.Context, spec genSpec) (<-chan genEvent, int, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	i := len(e.specs)
	e.specs = append(e.specs, spec)
	if i >= len(e.replies) {
		i = len(e.replies) - 1
	}
	ch := make(chan genEvent, len(e.replies[i]))
	for _, ev := range e.replies[i] {
		ch <- ev
	}
	close(ch)
	return ch, 7, nil
}

func content(s string) genEvent { return genEvent{parsedEvent: parsedEvent{Kind: "content", Text: s}} }
func toolCall(name, args string) genEvent {
	return genEvent{parsedEvent: parsedEvent{Kind: "tool", Tool: &parsedTool{Name: name, Arguments: args}}}
}
func end(finish string) genEvent { return genEvent{End: true, Finish: finish} }

func runGuard(t *testing.T, policy string, spec genSpec, replies ...[]genEvent) (text string, calls []*parsedTool, eng *scriptedEngine, trace string) {
	t.Helper()
	eng = &scriptedEngine{replies: replies}
	guardStartHook = eng.start
	defer func() { guardStartHook = nil }()
	srv := NewStrataServer(ServerConfig{Port: 0, MaxContext: 2048, GuardPolicy: policy})
	sink := &traceSink{}
	ctx := reqtrace.With(context.Background(), sink)
	spec.Messages = append(spec.Messages, chattemplate.Message{Role: "user", Content: "hi"})
	spec.mathDone, spec.toolsDone = true, true
	out, n, err := srv.startChat(ctx, spec)
	if err != nil || n != 7 {
		t.Fatalf("startChat: %v, prompt tokens %d", err, n)
	}
	var sb strings.Builder
	for ev := range out {
		if ev.Kind == "content" {
			sb.WriteString(ev.Text)
		}
		if ev.Kind == "tool" {
			calls = append(calls, ev.Tool)
		}
	}
	return sb.String(), calls, eng, sink.text()
}

func TestGuardRegeneratesWrongArithmetic(t *testing.T) {
	spec := genSpec{}
	spec.Sampling.Temperature = 0.7
	text, _, eng, trace := runGuard(t, "regenerate", spec,
		[]genEvent{content("So 12 * 12 = 154."), end("stop")},
		[]genEvent{content("So 12 * 12 = 144."), end("stop")})
	if strings.Contains(text, "154") || !strings.Contains(text, "144") {
		t.Fatalf("the client got %q, want only the corrected answer", text)
	}
	if len(eng.specs) != 2 {
		t.Fatalf("%d generations, want 2", len(eng.specs))
	}
	retry := eng.specs[1]
	if n := len(retry.Messages); n != 3 || retry.Messages[1].Role != "assistant" || !strings.Contains(retry.Messages[2].Content, "12 * 12") {
		t.Errorf("the retry does not carry the draft and the reason: %+v", retry.Messages)
	}
	if retry.Sampling.Temperature >= 0.7 {
		t.Errorf("temperature not lowered: %v", retry.Sampling.Temperature)
	}
	for _, want := range []string{"guard: attempt 1 failed", "regenerating", "attempt 2 passed"} {
		if !strings.Contains(trace, want) {
			t.Errorf("the Monitor trail lacks %q:\n%s", want, trace)
		}
	}
}

func TestGuardRegeneratesMalformedToolCall(t *testing.T) {
	spec := genSpec{Tools: []json.RawMessage{weatherTool}}
	_, calls, eng, trace := runGuard(t, "regenerate", spec,
		[]genEvent{toolCall("get_weather", `{"days":2}`), end("tool_calls")},
		[]genEvent{toolCall("get_weather", `{"city":"Oslo"}`), end("tool_calls")})
	if len(calls) != 1 || calls[0].Arguments != `{"city":"Oslo"}` || len(eng.specs) != 2 {
		t.Fatalf("calls %+v after %d generations", calls, len(eng.specs))
	}
	if !strings.Contains(trace, `required argument "city" is missing`) {
		t.Errorf("trail:\n%s", trace)
	}
}

func TestGuardGivesTheLastAnswerWhenRetriesRunOut(t *testing.T) {
	text, _, eng, trace := runGuard(t, "regenerate", genSpec{}, []genEvent{content("1 + 1 = 3"), end("stop")})
	if len(eng.specs) != guardMaxRetries+1 || text != "1 + 1 = 3" {
		t.Fatalf("%d generations, text %q", len(eng.specs), text)
	}
	if !strings.Contains(trace, "still failing after 3 attempts") {
		t.Errorf("trail:\n%s", trace)
	}
}

func TestGuardLeavesCleanRepliesAndOtherPoliciesAlone(t *testing.T) {
	good := []genEvent{content("2 + 2 = 4"), end("stop")}
	if text, _, eng, _ := runGuard(t, "regenerate", genSpec{}, good); text != "2 + 2 = 4" || len(eng.specs) != 1 {
		t.Errorf("clean reply: %q after %d generations", text, len(eng.specs))
	}
	bad := []genEvent{content("2 + 2 = 5"), end("stop")}
	if text, _, eng, trace := runGuard(t, "monitor", genSpec{}, bad); text != "2 + 2 = 5" || len(eng.specs) != 1 || !strings.Contains(trace, "attempt 1 failed") {
		t.Errorf("monitor: %q after %d generations, trail %q", text, len(eng.specs), trace)
	}
	if text, _, eng, trace := runGuard(t, "off", genSpec{}, bad); text != "2 + 2 = 5" || len(eng.specs) != 1 || trace != "" {
		t.Errorf("off: %q after %d generations, trail %q", text, len(eng.specs), trace)
	}
	if text, _, eng, _ := runGuard(t, "regenerate", genSpec{GuardOff: true}, bad); text != "2 + 2 = 5" || len(eng.specs) != 1 {
		t.Errorf("header off: %q after %d generations", text, len(eng.specs))
	}
}

// A reply that was cut off, or ended in an error, is passed on: asking again would not repair it.
func TestGuardDoesNotRetryTruncatedOrFailedReplies(t *testing.T) {
	_, _, eng, _ := runGuard(t, "regenerate", genSpec{}, []genEvent{content("9 * 9 = 80"), end("length")})
	if len(eng.specs) != 1 {
		t.Errorf("a truncated reply was regenerated (%d generations)", len(eng.specs))
	}
}
