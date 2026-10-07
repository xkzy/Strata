package resourcemanager

import (
	"testing"
)

func TestResourceManager(t *testing.T) {
	rm := NewResourceManager()

	ok, retry := rm.AdmitRequest(MathEvaluation)
	if !ok || retry != 0 {
		t.Fatalf("expected admission for initial work")
	}

	met := rm.Metrics()
	if met["active_work"].(int) != 1 {
		t.Errorf("expected active work 1")
	}

	rm.FinishRequest()
	met2 := rm.Metrics()
	if met2["active_work"].(int) != 0 {
		t.Errorf("expected active work 0")
	}
}
