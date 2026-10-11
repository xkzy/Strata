// pkg/runconfig/vector_finder.go - Auto-discovery and retrieval of refusal control vectors
package runconfig

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const (
	DefaultRefusalVectorFilename = "Qwen3.8-Flash-Next-experimental-speed-projection.gguf"
	DefaultRefusalVectorURL      = "https://raw.githubusercontent.com/xkzy/Strata/main/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf"
)

// FindOrFetchRefusalVector searches local paths for a measured refusal direction vector.
// If none exists locally, it attempts to automatically download the official vector.
func FindOrFetchRefusalVector(packDir, modelDir string) string {
	// 1. Candidate paths to check
	var candidates []string

	// Standard data directory
	candidates = append(candidates, filepath.Join("data", "experimental-speed-projection", DefaultRefusalVectorFilename))

	// Relative to executable
	if exe, err := os.Executable(); err == nil && exe != "" {
		candidates = append(candidates, filepath.Join(filepath.Dir(exe), "data", "experimental-speed-projection", DefaultRefusalVectorFilename))
	}

	// In pack dir
	if packDir != "" {
		candidates = append(candidates,
			filepath.Join(packDir, DefaultRefusalVectorFilename),
			filepath.Join(packDir, "control-vector.gguf"),
			filepath.Join(packDir, "refusal-vector.gguf"),
			filepath.Join(packDir, "abliteration.gguf"),
		)
	}

	// In model dir
	if modelDir != "" {
		candidates = append(candidates,
			filepath.Join(modelDir, DefaultRefusalVectorFilename),
			filepath.Join(filepath.Dir(modelDir), "data", "experimental-speed-projection", DefaultRefusalVectorFilename),
		)
	}

	// Glob search in data directory
	for _, pattern := range []string{
		filepath.Join("data", "*", "*speed-projection*.gguf"),
		filepath.Join("data", "*", "*refusal*.gguf"),
		filepath.Join("data", "*", "*abliteration*.gguf"),
		filepath.Join("data", "*", "*control-vector*.gguf"),
	} {
		if matches, err := filepath.Glob(pattern); err == nil {
			candidates = append(candidates, matches...)
		}
	}

	// Check if any candidate exists and is non-empty
	for _, c := range candidates {
		if c == "" {
			continue
		}
		if fi, err := os.Stat(c); err == nil && !fi.IsDir() && fi.Size() > 0 {
			return c
		}
	}

	// 2. Not found locally: auto-fetch from official remote
	destDir := filepath.Join("data", "experimental-speed-projection")
	destPath := filepath.Join(destDir, DefaultRefusalVectorFilename)

	if err := os.MkdirAll(destDir, 0755); err != nil {
		return ""
	}

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Get(DefaultRefusalVectorURL)
	if err != nil || resp.StatusCode != http.StatusOK {
		if resp != nil && resp.Body != nil {
			resp.Body.Close()
		}
		return ""
	}
	defer resp.Body.Close()

	tmpPath := destPath + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return ""
	}

	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		_ = os.Remove(tmpPath)
		return ""
	}
	f.Close()

	if err := os.Rename(tmpPath, destPath); err != nil {
		return ""
	}

	fmt.Fprintf(os.Stderr, "strata: auto-downloaded refusal control vector to %s\n", destPath)
	return destPath
}
