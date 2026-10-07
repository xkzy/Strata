// pkg/installer/installer_test.go - Unit Tests for Installer Package
package installer

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestDetectHardware(t *testing.T) {
	specs := DetectHardware()
	if specs.CPUCores <= 0 {
		t.Fatalf("expected positive CPU cores count, got %d", specs.CPUCores)
	}
	if specs.RAMTotalGB <= 0 {
		t.Fatalf("expected positive RAM total, got %f", specs.RAMTotalGB)
	}
}

func TestRecommendModel(t *testing.T) {
	specs1 := HardwareSpecs{RAMTotalGB: 32.0}
	rec1 := RecommendModel(specs1)
	if rec1.Name != "Qwen2.5-Coder-7B-Instruct-Q8_0" {
		t.Errorf("expected Q8_0 for 32GB RAM, got %s", rec1.Name)
	}

	specs2 := HardwareSpecs{RAMTotalGB: 16.0}
	rec2 := RecommendModel(specs2)
	if rec2.Name != "Qwen2.5-Coder-7B-Instruct-Q4_K_M" {
		t.Errorf("expected Q4_K_M for 16GB RAM, got %s", rec2.Name)
	}

	specs3 := HardwareSpecs{RAMTotalGB: 8.0}
	rec3 := RecommendModel(specs3)
	if rec3.Name != "Qwen2.5-Coder-7B-Instruct-Q2_K" {
		t.Errorf("expected Q2_K for 8GB RAM, got %s", rec3.Name)
	}
}

func TestDownloadWithResumeAndVerify(t *testing.T) {
	content := "Dummy model weights content payload for testing."
	h := sha256.New()
	h.Write([]byte(content))
	expectedHash := hex.EncodeToString(h.Sum(nil))

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "48")
		w.Write([]byte(content))
	}))
	defer ts.Close()

	tmpDir := t.TempDir()
	dest := filepath.Join(tmpDir, "model.gguf")

	err := DownloadWithResume(ts.URL, dest, nil)
	if err != nil {
		t.Fatalf("download failed: %v", err)
	}

	valid, err := VerifyFileSHA256(dest, expectedHash)
	if err != nil || !valid {
		t.Fatalf("sha256 verification failed: valid=%v, err=%v", valid, err)
	}

	os.Remove(dest)
}

func TestSetupManager(t *testing.T) {
	content := "Dummy test payload content for SetupManager."
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "44")
		w.Write([]byte(content))
	}))
	defer ts.Close()

	sm := NewSetupManager()
	status := sm.GetStatus()
	if status.Status != StatusIdle {
		t.Fatalf("expected idle status initially, got %s", status.Status)
	}

	tmpDir := t.TempDir()
	dest := filepath.Join(tmpDir, "setup_model.gguf")

	err := sm.StartDownload("TestModel", ts.URL, dest)
	if err != nil {
		t.Fatalf("failed to start download: %v", err)
	}

	// Double start should fail
	err2 := sm.StartDownload("TestModel2", ts.URL, dest)
	if err2 == nil {
		t.Fatalf("expected error when starting duplicate download, got nil")
	}

	// Wait for completion or check status
	time.Sleep(100 * time.Millisecond)
	st := sm.GetStatus()
	if st.Status != StatusCompleted && st.Status != StatusDownloading {
		t.Fatalf("unexpected status: %s", st.Status)
	}
}

