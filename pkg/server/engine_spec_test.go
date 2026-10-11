// pkg/server/engine_spec_test.go - Tests for engine spec and pack directory config resolution.
package server

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFindEngineSpecWithPackDirectory(t *testing.T) {
	tmpDir := t.TempDir()
	origWd, _ := os.Getwd()
	defer os.Chdir(origWd)
	_ = os.Chdir(tmpDir)

	packDir := filepath.Join(tmpDir, "packs", "coder-iq1_m")
	if err := os.MkdirAll(packDir, 0o755); err != nil {
		t.Fatal(err)
	}

	// Create strata-coder-iq1_m.json in working dir
	configContent := `{
		"exe": "engine/strata",
		"args": ["--pack", "` + packDir + `", "--max-context", "32768"],
		"model_name": "qwen3.8-flash-next-coder-iq1_m"
	}`
	configFile := filepath.Join(tmpDir, "strata-coder-iq1_m.json")
	if err := os.WriteFile(configFile, []byte(configContent), 0o644); err != nil {
		t.Fatal(err)
	}

	// Case 1: configFile argument is the pack directory
	spec, source, err := findEngineSpec(packDir, "")
	if err != nil {
		t.Fatalf("expected findEngineSpec to resolve pack dir, got error: %v", err)
	}
	if spec == nil || spec.ModelName != "qwen3.8-flash-next-coder-iq1_m" {
		t.Fatalf("unexpected spec: %+v", spec)
	}
	if filepath.Base(source) != "strata-coder-iq1_m.json" {
		t.Fatalf("expected source strata-coder-iq1_m.json, got %q", source)
	}

	// Case 2: lastConfigFile argument is the pack directory
	spec2, source2, err := findEngineSpec("", packDir)
	if err != nil {
		t.Fatalf("expected findEngineSpec to resolve lastConfigFile pack dir, got error: %v", err)
	}
	if spec2 == nil || spec2.ModelName != "qwen3.8-flash-next-coder-iq1_m" {
		t.Fatalf("unexpected spec: %+v", spec2)
	}
	if filepath.Base(source2) != "strata-coder-iq1_m.json" {
		t.Fatalf("expected source strata-coder-iq1_m.json, got %q", source2)
	}

	// Case 3: readEngineSpec on a directory returns error gracefully without "is a directory" panic
	_, err = readEngineSpec(packDir)
	if err == nil {
		t.Fatal("expected error reading directory with readEngineSpec, got nil")
	}
}
