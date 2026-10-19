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

// The generated config is where resource_adapt is chosen; both values must survive, and false must be written
// explicitly so it is not read back as an absent key.
func TestRunConfigHonoursResourceAdaptSwitch(t *testing.T) {
	for _, want := range []bool{true, false} {
		cfg, err := GenerateRunConfig(SetupConfigOptions{PackDir: "/pack", NativeGGUF: "/model.gguf", ResourceAdapt: want})
		if err != nil {
			t.Fatal(err)
		}
		if cfg.ResourceAdapt != want {
			t.Errorf("GenerateRunConfig(ResourceAdapt=%v) = %v", want, cfg.ResourceAdapt)
		}
		data, err := json.Marshal(cfg)
		if err != nil {
			t.Fatal(err)
		}
		var parsed map[string]any
		if err := json.Unmarshal(data, &parsed); err != nil {
			t.Fatal(err)
		}
		got, ok := parsed["resource_adapt"]
		if !ok {
			t.Fatalf("resource_adapt must be written even when %v: %s", want, data)
		}
		if got != want {
			t.Errorf("written resource_adapt = %v, want %v", got, want)
		}
	}
}
