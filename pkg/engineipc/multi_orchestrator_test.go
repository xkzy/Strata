// pkg/engineipc/multi_orchestrator_test.go - Unit Tests for MultiEngineManager
package engineipc

import (
	"testing"
	"time"
)

func TestMultiEngineManager_LifecycleAndRouting(t *testing.T) {
	mgr := NewMultiEngineManager(true, 100*time.Millisecond)

	cfgQwen := EngineConfig{
		BinaryPath:  "strata",
		ModelPath:   "/models/qwen.gguf",
		ContextSize: 32768,
	}

	cfgMiMo := EngineConfig{
		BinaryPath:  "strata",
		ModelPath:   "/models/mimo.gguf",
		ContextSize: 131072,
	}

	// 1. Register Models
	mgr.RegisterModel("qwen3.8-flash-next", cfgQwen, nil)
	mgr.RegisterModel("mimo-v2.6-flash-rl", cfgMiMo, nil)

	if mgr.ActiveModel() != "qwen3.8-flash-next" {
		t.Fatalf("expected initial active model qwen3.8-flash-next, got %s", mgr.ActiveModel())
	}

	models := mgr.ListModels()
	if len(models) != 2 {
		t.Fatalf("expected 2 registered models, got %d", len(models))
	}
	if models["qwen3.8-flash-next"] != StateStopped.String() {
		t.Fatalf("expected initial state stopped, got %s", models["qwen3.8-flash-next"])
	}

	// 2. GetOrchestrator for Qwen
	orchQwen, err := mgr.GetOrchestrator("qwen3.8-flash-next")
	if err != nil || orchQwen == nil {
		t.Fatalf("failed to get qwen orchestrator: %v", err)
	}

	// 3. GetOrchestrator for MiMo
	orchMiMo, err := mgr.GetOrchestrator("mimo-v2.6-flash-rl")
	if err != nil || orchMiMo == nil {
		t.Fatalf("failed to get mimo orchestrator: %v", err)
	}

	if mgr.ActiveModel() != "mimo-v2.6-flash-rl" {
		t.Fatalf("expected active model switched to mimo-v2.6-flash-rl, got %s", mgr.ActiveModel())
	}

	// 4. Fallback to active model on empty or unknown requested model
	orchFallback, err := mgr.GetOrchestrator("")
	if err != nil || orchFallback == nil {
		t.Fatalf("expected fallback to active model on empty ID, got error: %v", err)
	}

	// 5. SetActiveModel
	if err := mgr.SetActiveModel("qwen3.8-flash-next"); err != nil {
		t.Fatalf("failed to set active model: %v", err)
	}
	if mgr.ActiveModel() != "qwen3.8-flash-next" {
		t.Fatalf("expected active model qwen3.8-flash-next, got %s", mgr.ActiveModel())
	}

	// 6. PruneIdle after sleeping past idle timeout
	time.Sleep(150 * time.Millisecond)
	pruned := mgr.PruneIdle()
	// MiMo was idle and not active, so it should be pruned, while Qwen is active
	if pruned < 1 {
		t.Fatalf("expected at least 1 idle engine pruned, got %d", pruned)
	}

	// StopAll cleanup
	mgr.StopAll()
}
