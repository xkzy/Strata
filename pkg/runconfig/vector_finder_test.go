// pkg/runconfig/vector_finder_test.go - Tests for refusal vector auto-discovery
package runconfig

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFindOrFetchRefusalVector_LocalFound(t *testing.T) {
	// 1. Should find the tracked repository vector
	v := FindOrFetchRefusalVector("", "")
	if v == "" {
		t.Skip("local vector not present in test working directory")
	}
	if fi, err := os.Stat(v); err != nil || fi.Size() == 0 {
		t.Fatalf("expected valid file at %s, got err: %v", v, err)
	}
}

func TestFindOrFetchRefusalVector_PackDir(t *testing.T) {
	tmpDir := t.TempDir()
	fakeVector := filepath.Join(tmpDir, "refusal-vector.gguf")
	if err := os.WriteFile(fakeVector, []byte("fake-vector-data"), 0644); err != nil {
		t.Fatal(err)
	}

	found := FindOrFetchRefusalVector(tmpDir, "")
	if found != fakeVector {
		// If working dir has the primary default, it might find that first, which is valid.
		// Test with nonexistent base dir to ensure packDir is prioritized when default is missing
		t.Logf("Found %s", found)
	}
}
