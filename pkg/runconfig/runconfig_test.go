// pkg/runconfig/runconfig_test.go - Unit Tests for Configuration Manager
package runconfig

import (
	"os"
	"path/filepath"
	"testing"
)

func TestConfigManager_DefaultsAndFilePersistence(t *testing.T) {
	cm := NewConfigManager("")
	cfg := cm.Get()

	if cfg.Port != 8080 {
		t.Fatalf("expected default port 8080, got %d", cfg.Port)
	}
	if cfg.Model != "Qwen3.8-Flash-Next" {
		t.Fatalf("expected default model Qwen3.8-Flash-Next, got %s", cfg.Model)
	}

	// Update configuration
	cfg.Port = 9090
	cfg.Sampling.Temperature = 0.7
	cm.Update(cfg)

	// Save to temporary file
	tmpDir := t.TempDir()
	tmpFile := filepath.Join(tmpDir, "test-config.json")

	if err := cm.SaveToFile(tmpFile); err != nil {
		t.Fatalf("failed to save config: %v", err)
	}

	// Load in a new manager
	cm2 := NewConfigManager(tmpFile)
	cfg2 := cm2.Get()

	if cfg2.Port != 9090 {
		t.Fatalf("expected loaded port 9090, got %d", cfg2.Port)
	}
	if cfg2.Sampling.Temperature != 0.7 {
		t.Fatalf("expected loaded temperature 0.7, got %v", cfg2.Sampling.Temperature)
	}

	os.Remove(tmpFile)
}

func TestConfig_MultiModelSupport(t *testing.T) {
	// Single model legacy
	legacy := DefaultConfig()
	legacy.Model = "qwen3.8-flash-next"
	legacy.ModelPath = "/models/qwen.gguf"

	actID, actPath := legacy.GetActiveModel()
	if actID != "qwen3.8-flash-next" || actPath != "/models/qwen.gguf" {
		t.Fatalf("unexpected legacy active model: %s, %s", actID, actPath)
	}

	entry, ok := legacy.GetModelEntry("qwen3.8-flash-next")
	if !ok || entry.ModelPath != "/models/qwen.gguf" {
		t.Fatalf("failed to get legacy model entry: %+v", entry)
	}

	all := legacy.AllModels()
	if len(all) != 1 || all["qwen3.8-flash-next"].ModelPath != "/models/qwen.gguf" {
		t.Fatalf("unexpected all models from legacy config: %+v", all)
	}

	// Multi-model config
	multi := DefaultConfig()
	multi.DefaultModel = "mimo"
	multi.Models = map[string]ModelEntry{
		"qwen": {
			ID:        "qwen3.8-flash-next",
			ModelPath: "/models/qwen.gguf",
		},
		"mimo": {
			ID:        "mimo-v2.6-flash-rl",
			ModelPath: "/models/mimo.gguf",
		},
	}

	mActID, mActPath := multi.GetActiveModel()
	if mActID != "mimo-v2.6-flash-rl" || mActPath != "/models/mimo.gguf" {
		t.Fatalf("unexpected multi-model active model: %s, %s", mActID, mActPath)
	}

	mEntry, ok := multi.GetModelEntry("qwen")
	if !ok || mEntry.ModelPath != "/models/qwen.gguf" {
		t.Fatalf("failed to get multi-model entry: %+v", mEntry)
	}

	mAll := multi.AllModels()
	if len(mAll) != 2 {
		t.Fatalf("expected 2 models in multi config, got %d", len(mAll))
	}
}

// The config file may hold api_key: it is written owner-only, an older world-readable file included.
func TestSaveToFileIsOwnerOnly(t *testing.T) {
	path := filepath.Join(t.TempDir(), "strata-config.json")
	if err := os.WriteFile(path, []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	cm := NewConfigManager("")
	if err := cm.SaveToFile(path); err != nil {
		t.Fatal(err)
	}
	if st, _ := os.Stat(path); st.Mode().Perm() != 0o600 {
		t.Fatalf("mode = %v, want 0600", st.Mode().Perm())
	}
}

// Only an explicit false turns the runtime controller off; a missing key keeps the default (on).
func TestResourceAdaptAbsentKeepsTheDefault(t *testing.T) {
	dir := t.TempDir()
	write := func(body string) StrataConfig {
		path := filepath.Join(dir, "cfg.json")
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		cm := NewConfigManager(path)
		return cm.Get()
	}
	if !write(`{"port": 8080}`).ResourceAdapt {
		t.Error("a config without resource_adapt must keep the default (on)")
	}
	if write(`{"resource_adapt": false}`).ResourceAdapt {
		t.Error("an explicit false must disable adaptation")
	}
	if !write(`{"resource_adapt": true}`).ResourceAdapt {
		t.Error("an explicit true must enable adaptation")
	}
}
