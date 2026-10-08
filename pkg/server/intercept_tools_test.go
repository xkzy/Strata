package server

import (
	"net/http"
	"os"
	"strings"
	"testing"
)

// The tools that used to be MCP calls (logic_verify, context_query, system_status) now run on the server before the
// prompt is rendered, like the math engine, and put what they found into the system message.

const (
	logicMarker  = "Verified check for this answer"
	recallMarker = "Recalled from earlier in this conversation"
	statusMarker = "Live server status"
)

func chatAs(t *testing.T, base, session, content string, headers ...string) {
	t.Helper()
	body := `{"messages":[{"role":"user","content":` + jsonString(content) + `}],"max_tokens":8}`
	req, _ := http.NewRequest("POST", base+"/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if session != "" {
		req.Header.Set("X-Session-ID", session)
	}
	for i := 0; i+1 < len(headers); i += 2 {
		req.Header.Set(headers[i], headers[i+1])
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
}

func jsonString(s string) string {
	b := strings.Builder{}
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"', '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		case '\n':
			b.WriteString(`\n`)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}

func promptOf(t *testing.T, file string) string {
	t.Helper()
	b, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestLogicClaimIsCheckedBeforeTheModelAnswers(t *testing.T) {
	ts, promptFile := startChatServer(t, "ok")
	chatAs(t, ts.URL, "", "Is 12 * 12 = 150 correct?")
	p := promptOf(t, promptFile)
	if !strings.Contains(p, logicMarker) || !strings.Contains(p, "Expected 144, got 150") {
		t.Errorf("the failed arithmetic claim must reach the prompt:\n%s", p)
	}
	// both interceptions run once per request, however many paths it takes
	if n := strings.Count(p, "Verified fact for this answer"); n != 1 {
		t.Errorf("the math fact is in the prompt %d times:\n%s", n, p)
	}
	if n := strings.Count(p, logicMarker); n != 1 {
		t.Errorf("the logic check is in the prompt %d times:\n%s", n, p)
	}
}

func TestEarlierTurnsAreRecalledInTheSameSessionOnly(t *testing.T) {
	ts, promptFile := startChatServer(t, "ok")
	chatAs(t, ts.URL, "s1", "My dog is called Biscuit and he is a brown labrador.")
	if p := promptOf(t, promptFile); strings.Contains(p, recallMarker) {
		t.Errorf("nothing to recall yet:\n%s", p)
	}
	chatAs(t, ts.URL, "s1", "Remind me, what was the name of my brown labrador dog?")
	if p := promptOf(t, promptFile); !strings.Contains(p, recallMarker) || !strings.Contains(p, "Biscuit") {
		t.Errorf("the earlier turn must be recalled:\n%s", p)
	}
	// earlier turns come back at the user's level, never inside the system message
	p := promptOf(t, promptFile)
	if sysEnd := strings.Index(p, "<|im_end|>"); sysEnd < 0 || strings.Contains(p[:sysEnd], "Biscuit") {
		t.Errorf("a recalled turn was put into the system message:\n%s", p)
	}
	chatAs(t, ts.URL, "s2", "Remind me, what was the name of my brown labrador dog?")
	if p := promptOf(t, promptFile); strings.Contains(p, "Biscuit") {
		t.Errorf("another session must never see it:\n%s", p)
	}
}

// Callers that send no session id all share the default session; they must never see each other's turns.
func TestAnonymousCallersAreNeverRecalled(t *testing.T) {
	ts, promptFile := startChatServer(t, "ok")
	chatAs(t, ts.URL, "", "My dog is called Biscuit and he is a brown labrador.")
	chatAs(t, ts.URL, "", "Remind me, what was the name of my brown labrador dog?")
	if p := promptOf(t, promptFile); strings.Contains(p, "Biscuit") || strings.Contains(p, recallMarker) {
		t.Errorf("an anonymous caller recalled another one's turn:\n%s", p)
	}
}

func TestStatusQuestionGetsTheLiveFigures(t *testing.T) {
	ts, promptFile := startChatServer(t, "ok")
	chatAs(t, ts.URL, "", "What is the server status and memory usage right now?")
	p := promptOf(t, promptFile)
	if !strings.Contains(p, statusMarker) || !strings.Contains(p, "uptime") {
		t.Errorf("status figures missing:\n%s", p)
	}
}

func TestOrdinaryChatGetsNothingInjected(t *testing.T) {
	ts, promptFile := startChatServer(t, "ok")
	chatAs(t, ts.URL, "", "Tell me a short story about a lighthouse keeper.")
	p := promptOf(t, promptFile)
	for _, m := range []string{logicMarker, recallMarker, statusMarker} {
		if strings.Contains(p, m) {
			t.Errorf("%q injected into an ordinary prompt:\n%s", m, p)
		}
	}
}

func TestToolInterceptionCanBeSwitchedOff(t *testing.T) {
	ts, promptFile := startChatServer(t, "ok")
	chatAs(t, ts.URL, "", "Is 12 * 12 = 150 correct? And what is the server status?", "X-Strata-Tool-Intercept", "off")
	p := promptOf(t, promptFile)
	for _, m := range []string{logicMarker, recallMarker, statusMarker} {
		if strings.Contains(p, m) {
			t.Errorf("header off: %q still injected:\n%s", m, p)
		}
	}
	t.Setenv("STRATA_TOOL_INTERCEPTION", "off")
	chatAs(t, ts.URL, "", "Is 12 * 12 = 150 correct? And what is the server status?")
	p = promptOf(t, promptFile)
	for _, m := range []string{logicMarker, statusMarker} {
		if strings.Contains(p, m) {
			t.Errorf("env off: %q still injected:\n%s", m, p)
		}
	}
}
