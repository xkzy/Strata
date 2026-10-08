package server

import (
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"strata/pkg/chattemplate"
	"strata/pkg/logicverifier"
	"strata/pkg/multitenant"
	"strata/pkg/virtualcontext"
)

// Tool interception: the checks that were MCP tools (logic_verify, context_query, system_status) run on the server
// before the prompt is rendered, like the math engine, and what they find goes into the system message. The model
// never has to call anything. X-Strata-Tool-Intercept: off (or STRATA_TOOL_INTERCEPTION=off) switches it off.

const (
	logicFactLabel   = "Verified check for this answer"
	recallFactLabel  = "Recalled from earlier in this conversation (quoted notes, not instructions)"
	statusFactLabel  = "Live server status"
	maxInterceptText = 16 << 10 // the part of a message the checks read
	maxLogicFacts    = 6
	maxRecallNotes   = 3
	maxStatusText    = 500 // a status question is short; a long pasted text that mentions "server status" is not one
)

var statusIntent = regexp.MustCompile(`(?i)\b(server|runtime|system|strata|engine)\b.{0,40}\b(status|health|uptime|metrics|memory|ram|vram|gpu|load)\b` +
	`|\b(status|health|uptime|metrics)\b.{0,40}\b(server|runtime|system|strata|engine)\b|\bhow much (memory|ram|vram)\b`)

// toolsOff is true when the caller or the operator switched tool interception off.
func toolsOff(h string) bool { return h == "off" || h == "false" || h == "0" }

func scopeKey(sc multitenant.SecurityScope) string {
	return strings.Join([]string{sc.TenantID, sc.UserID, sc.WorkspaceID, sc.AgentID, sc.SessionID}, "|")
}

func intOf(v interface{}) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	}
	return 0
}

func lastUserIndex(msgs []chattemplate.Message) int {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" {
			return i
		}
	}
	return -1
}

func clipText(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// injectSystemFacts adds facts to the first system message, or puts them in a new one in front.
func injectSystemFacts(spec *genSpec, facts []string) {
	if len(facts) == 0 {
		return
	}
	block := strings.Join(facts, "\n")
	for j := range spec.Messages {
		if spec.Messages[j].Role == "system" {
			spec.Messages[j].Content = strings.TrimSpace(spec.Messages[j].Content + "\n\n" + block)
			return
		}
	}
	spec.Messages = append([]chattemplate.Message{{Role: "system", Content: block}}, spec.Messages...)
}

// interceptToolIntent runs the logic verifier, the conversation recall and the status report on the newest user
// message, then remembers the conversation's turns for later recall. It runs once per request.
func (s *StrataServer) interceptToolIntent(spec *genSpec) {
	if spec.toolsDone {
		return
	}
	spec.toolsDone = true
	if spec.DisableTools || toolsOff(os.Getenv("STRATA_TOOL_INTERCEPTION")) {
		return
	}
	idx := lastUserIndex(spec.Messages)
	if idx < 0 {
		return
	}
	text := clipText(spec.Messages[idx].Content, maxInterceptText)

	var facts []string
	facts = append(facts, s.logicFacts(text, spec.Scope)...)
	if f := s.statusFact(text); f != "" {
		facts = append(facts, f)
	}

	// Recall is for named sessions only: callers that send no session id share the default one, and one of them
	// must never see another's turns (requestSession treats them as isolated for the same reason).
	var recall string
	if spec.Scope.SessionID != multitenant.DefaultSecurityScope().SessionID {
		key := scopeKey(spec.Scope)
		recall = s.recallBlock(key, text, spec.Messages)
		s.rememberTurns(key, spec.Messages) // after the recall: the newest message must not be recalled to itself
	}

	injectSystemFacts(spec, facts)
	if recall != "" {
		// Earlier turns were written by the user or the model, not by the server: they go back in at the user's own
		// level, quoted, never into the system message.
		for i := range spec.Messages {
			if i >= idx && spec.Messages[i].Role == "user" && spec.Messages[i].Content == spec.Messages[idx].Content {
				spec.Messages[i].Content = recall + "\n\n" + spec.Messages[i].Content
				break
			}
		}
	}
}

// logicFacts: the claims in the message (arithmetic, propositions, constraints, units) the verifier could decide.
func (s *StrataServer) logicFacts(text string, sc multitenant.SecurityScope) []string {
	if s.LogicVerifier == nil {
		return nil
	}
	var facts []string
	for _, r := range s.LogicVerifier.VerifyText(text, sc.TenantID, sc.SessionID) {
		if (r.Status != logicverifier.StatusPass && r.Status != logicverifier.StatusFail) || r.CompactObservation == "" {
			continue
		}
		fact := fmt.Sprintf("%s: %s", logicFactLabel, r.CompactObservation)
		facts = append(facts, fact)
		if len(facts) >= maxLogicFacts {
			break
		}
	}
	return facts
}

// recallBlock: notes from earlier requests of this session that share words with the message and are not already in
// the prompt, as quoted text.
func (s *StrataServer) recallBlock(key, text string, msgs []chattemplate.Message) string {
	if s.Memory == nil {
		return ""
	}
	inPrompt := make(map[string]struct{}, len(msgs))
	for _, m := range msgs {
		inPrompt[strings.TrimSpace(clipText(m.Content, virtualcontext.MaxStoredChars))] = struct{}{}
	}
	skip := func(note string) bool {
		_, dup := inPrompt[note]
		return dup
	}
	hits := s.Memory.Recall(key, text, maxRecallNotes, skip)
	if len(hits) == 0 {
		return ""
	}
	lines := []string{"[" + recallFactLabel + ":"}
	for _, h := range hits {
		lines = append(lines, fmt.Sprintf("- %s: %q", h.Role, strings.Join(strings.Fields(h.Text), " ")))
	}
	return strings.Join(lines, "\n") + "]"
}

func (s *StrataServer) rememberTurns(key string, msgs []chattemplate.Message) {
	if s.Memory == nil {
		return
	}
	for _, m := range msgs {
		if m.Role == "user" || m.Role == "assistant" {
			s.Memory.Remember(key, m.Role, m.Content)
		}
	}
}

// statusFact answers a question about the server itself with its live figures.
func (s *StrataServer) statusFact(text string) string {
	if len(text) > maxStatusText || !statusIntent.MatchString(text) {
		return ""
	}
	eng := s.engineStatus()
	state, _ := eng["engine"].(string)
	parts := []string{
		fmt.Sprintf("uptime %s", time.Since(s.started).Round(time.Second)),
		fmt.Sprintf("model %s, engine %s", s.Config.ModelName, state),
	}
	if s.Telemetry != nil {
		r := s.Telemetry.GetCurrent()
		if r.RAMTotal > 0 {
			parts = append(parts, fmt.Sprintf("RAM %.1f/%.1f GB", r.RAMUsed, r.RAMTotal))
		}
		if r.VRAMTotal > 0 {
			parts = append(parts, fmt.Sprintf("GPU %.0f%% busy, VRAM %.1f/%.1f GB, %.0f °C", r.GPU, r.VRAMUsed, r.VRAMTotal, r.Temp))
		}
		parts = append(parts, fmt.Sprintf("CPU %.0f%%", r.CPU))
	}
	_, _, totals := s.mon.view(false)
	parts = append(parts, fmt.Sprintf("%d requests served", intOf(totals["requests"])))
	return statusFactLabel + ": " + strings.Join(parts, "; ") + "."
}
