package installer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteRunConfig(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "strata-test.json")

	opts := SetupConfigOptions{
		ModelTag:           "coder-iq1_m",
		PackDir:            "/path/to/pack",
		NativeGGUF:         "/path/to/model.gguf",
		MaxContext:         32768,
		ResidentBudgetGiB:  8,
		UseMmapExperts:     true,
		UseResidentExperts: false,
		SpecDraft:          4,
	}

	cfg, err := GenerateRunConfig(opts)
	if err != nil {
		t.Fatalf("GenerateRunConfig failed: %v", err)
	}

	if err := SaveRunConfig(cfg, configPath); err != nil {
		t.Fatalf("SaveRunConfig failed: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to read config file: %v", err)
	}

	var parsed map[string]any
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}

	args, ok := parsed["args"].([]any)
	if !ok {
		t.Fatal("expected 'args' array in config")
	}

	hasBudget := false
	for i, arg := range args {
		if arg == "--resident-budget-gib" && i+1 < len(args) {
			if args[i+1] == "8" {
				hasBudget = true
			}
		}
	}
	if !hasBudget {
		t.Errorf("expected --resident-budget-gib 8 in args, got %v", args)
	}

	// Test launcher script generation
	scriptPath := filepath.Join(tmpDir, "run.sh")
	if err := WriteLauncherScript(configPath, scriptPath); err != nil {
		t.Fatalf("WriteLauncherScript failed: %v", err)
	}

	scriptContent, err := os.ReadFile(scriptPath)
	if err != nil {
		t.Fatalf("failed to read script: %v", err)
	}
	if len(scriptContent) == 0 {
		t.Errorf("launcher script is empty")
	}
}
