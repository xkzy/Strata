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
