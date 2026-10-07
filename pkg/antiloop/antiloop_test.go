package antiloop

import (
	"testing"
)

func TestAntiLoopManager(t *testing.T) {
	mgr := Instance()

	norm, hash := CanonicalizePayload("tool_1", `{"query":"ls"}`)
	if norm == "" || hash == "" {
		t.Fatalf("canonicalization failed")
	}

	act := ActionRecord{
		ActionID:          "act_1",
		SessionID:         "sess_1",
		Kind:              ActionToolCall,
		TargetName:        "tool_1",
		NormalizedPayload: norm,
		CanonicalHash:     hash,
	}

	// First execution: normal
	v1 := mgr.EvaluateAction(act)
	if v1.State != StateNormal || !v1.Allowed {
		t.Errorf("expected normal state, got %v", v1.State)
	}

	// Repeated identical executions
	mgr.EvaluateAction(act)
	v3 := mgr.EvaluateAction(act)
	if v3.State != StateSuspected || !v3.IsThrottled {
		t.Errorf("expected suspected throttled state, got %v", v3.State)
	}

	mgr.EvaluateAction(act)
	v5 := mgr.EvaluateAction(act)
	if v5.State != StateBlocked || v5.Allowed {
		t.Errorf("expected blocked state on repeated loop, got %v", v5.State)
	}
}
