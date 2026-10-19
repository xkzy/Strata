package autotune

import (
	"path/filepath"
	"reflect"
	"testing"
)

func TestRelativeEnginePathsUseTheEngineCwd(t *testing.T) {
	cwd := t.TempDir()
	args := []string{"--pack", "pack", "--native", "/abs/model.gguf", "--max-context", "4096"}
	got := absolutePaths(args, cwd)
	want := []string{"--pack", filepath.Join(cwd, "pack"), "--native", "/abs/model.gguf", "--max-context", "4096"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
	if !reflect.DeepEqual(args, []string{"--pack", "pack", "--native", "/abs/model.gguf", "--max-context", "4096"}) {
		t.Errorf("the caller's arguments were modified: %v", args)
	}
}
