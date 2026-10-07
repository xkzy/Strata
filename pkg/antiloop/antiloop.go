package antiloop

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"sync"
	"time"
)

type EscalationState string

const (
	StateNormal    EscalationState = "NORMAL"
	StateSuspected EscalationState = "SUSPECTED"
	StateThrottled EscalationState = "THROTTLED"
	StateBlocked   EscalationState = "BLOCKED"
	StateCancelled EscalationState = "CANCELLED"
)

type ActionKind string

const (
	ActionInference       ActionKind = "inference"
	ActionToolCall        ActionKind = "tool_call"
	ActionToolResult      ActionKind = "tool_result"
	ActionRetrieval       ActionKind = "retrieval"
	ActionCompaction      ActionKind = "context_compaction"
	ActionAgentDelegation ActionKind = "agent_delegation"
	ActionMathEvaluation  ActionKind = "math_evaluation"
)

type ActionRecord struct {
	ActionID          string     `json:"action_id"`
	ParentID          string     `json:"parent_id"`
	TenantID          string     `json:"tenant_id"`
	UserID            string     `json:"user_id"`
	AgentID           string     `json:"agent_id"`
	SessionID         string     `json:"session_id"`
	Kind              ActionKind `json:"kind"`
	TargetName        string     `json:"target_name"`
	NormalizedPayload string     `json:"normalized_payload"`
	CanonicalHash     string     `json:"canonical_hash"`
	TimestampSec      float64    `json:"timestamp_sec"`
	IsError           bool       `json:"is_error"`
	ErrorMessage      string     `json:"error_message,omitempty"`
	ResultSummary     string     `json:"result_summary,omitempty"`
}

type GuardVerdict struct {
	State                 EscalationState `json:"state"`
	Allowed               bool            `json:"allowed"`
	IsThrottled           bool            `json:"is_throttled"`
	ThrottleDelayMs       float64         `json:"throttle_delay_ms"`
	Reason                string          `json:"reason,omitempty"`
	SuggestedRemediation  string          `json:"suggested_remediation,omitempty"`
	StructuredObservation string          `json:"structured_observation,omitempty"`
	ProgressScore         float64         `json:"progress_score"`
	CurrentStep           int             `json:"current_step"`
	NoProgressStreak      int             `json:"no_progress_streak"`
}

type AntiLoopManager struct {
	mu           sync.Mutex
	actionCounts map[string]int
	sessionSteps map[string]int
	blockedSess  map[string]time.Time
}

var (
	instance *AntiLoopManager
	once     sync.Once
)

func Instance() *AntiLoopManager {
	once.Do(func() {
		instance = &AntiLoopManager{
			actionCounts: make(map[string]int),
			sessionSteps: make(map[string]int),
			blockedSess:  make(map[string]time.Time),
		}
	})
	return instance
}

func CanonicalizePayload(target, payload string) (string, string) {
	norm := strings.TrimSpace(payload)
	h := sha256.Sum256([]byte(target + ":" + norm))
	return norm, hex.EncodeToString(h[:])
}

func (m *AntiLoopManager) EvaluateAction(act ActionRecord) GuardVerdict {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.sessionSteps[act.SessionID]++
	step := m.sessionSteps[act.SessionID]

	key := act.SessionID + ":" + act.CanonicalHash
	m.actionCounts[key]++
	count := m.actionCounts[key]

	if count > 4 {
		return GuardVerdict{
			State:                 StateBlocked,
			Allowed:               false,
			Reason:                "Action repeated more than allowed threshold (loop detected).",
			SuggestedRemediation:  "Break execution loop or delegate to deterministic tool/math runtime.",
			StructuredObservation: "[GuardBlock: Runaway execution loop stopped]",
			CurrentStep:           step,
			ProgressScore:         0.0,
		}
	} else if count > 2 {
		return GuardVerdict{
			State:           StateSuspected,
			Allowed:         true,
			IsThrottled:     true,
			ThrottleDelayMs: 50.0,
			Reason:          "Action repetition suspected.",
			CurrentStep:     step,
			ProgressScore:   0.5,
		}
	}

	return GuardVerdict{
		State:         StateNormal,
		Allowed:       true,
		CurrentStep:   step,
		ProgressScore: 1.0,
	}
}

func (m *AntiLoopManager) RecordActionOutcome(act ActionRecord, stateChanged bool, progressDelta float64, tokens int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	// Track state progress
}

func (m *AntiLoopManager) Stats() map[string]interface{} {
	m.mu.Lock()
	defer m.mu.Unlock()
	return map[string]interface{}{
		"active_tracked_actions": len(m.actionCounts),
		"active_sessions":        len(m.sessionSteps),
		"blocked_sessions":       len(m.blockedSess),
	}
}
