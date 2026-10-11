// pkg/installer/installer_test.go - Unit Tests for Installer Package
package installer

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
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
	if !strings.Contains(rec1.Name, "Q8_0") {
		t.Errorf("expected Q8_0 for 32GB RAM, got %s", rec1.Name)
	}

	specs2 := HardwareSpecs{RAMTotalGB: 16.0}
	rec2 := RecommendModel(specs2)
	if !strings.Contains(rec2.Name, "Q4_K_M") {
		t.Errorf("expected Q4_K_M for 16GB RAM, got %s", rec2.Name)
	}

	specs3 := HardwareSpecs{RAMTotalGB: 8.0}
	rec3 := RecommendModel(specs3)
	if !strings.Contains(rec3.Name, "Q2_K") {
		t.Errorf("expected Q2_K for 8GB RAM, got %s", rec3.Name)
	}
}

func TestResolveHuggingFaceInput(t *testing.T) {
	// 1. Direct GGUF URL
	u1 := "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
	m1, err := ResolveHuggingFaceInput(u1)
	if err != nil {
		t.Fatalf("unexpected error resolving URL: %v", err)
	}
	if m1.RepoID != "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF" {
		t.Errorf("expected repo ID bartowski/Qwen2.5-Coder-7B-Instruct-GGUF, got %s", m1.RepoID)
	}
	if m1.Filename != "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf" {
		t.Errorf("expected filename Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf, got %s", m1.Filename)
	}
	if m1.Category != "coder" {
		t.Errorf("expected category coder, got %s", m1.Category)
	}

	// 2. Repo ID without filename
	u2 := "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF"
	m2, err := ResolveHuggingFaceInput(u2)
	if err != nil {
		t.Fatalf("unexpected error resolving repo: %v", err)
	}
	if m2.Category != "reasoning" {
		t.Errorf("expected category reasoning, got %s", m2.Category)
	}

	// 3. Foreign hostname should be rejected
	u3 := "https://evil.example.com/malicious.gguf"
	_, err = ResolveHuggingFaceInput(u3)
	if err == nil {
		t.Fatalf("expected error for non-huggingface URL, got nil")
	}

	// 4. Directory traversal attempt should be rejected
	u4 := "https://huggingface.co/foo/bar/resolve/main/../../etc/passwd.gguf"
	m4, err := ResolveHuggingFaceInput(u4)
	if err == nil && strings.Contains(m4.Filename, "..") {
		t.Fatalf("expected sanitized filename without path traversal, got %s", m4.Filename)
	}
}

func TestGetAvailableModels(t *testing.T) {
	models := GetAvailableModels()
	if len(models) == 0 {
		t.Fatalf("expected non-empty models list")
	}

	var foundHF, foundCoder bool
	for _, m := range models {
		if m.Source == SourceHuggingFace {
			foundHF = true
		}
		if m.Category == "coder" {
			foundCoder = true
		}
	}
	if !foundHF {
		t.Errorf("expected Hugging Face models in catalog")
	}
	if !foundCoder {
		t.Errorf("expected Coder category in catalog")
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

func TestMultiShardResolutionAndDisk(t *testing.T) {
	// 1. Shard paths expansion
	p := "/models/MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf"
	shards := FindAllShardPaths(p)
	if len(shards) != 2 {
		t.Fatalf("expected 2 shards, got %d", len(shards))
	}
	if shards[0] != p || shards[1] != "/models/MiMo-V2.6-Flash-RL-MXFP4-00002-of-00002.gguf" {
		t.Errorf("unexpected shards: %v", shards)
	}

	// 2. Non-shard path
	single := "/models/single_model.gguf"
	if s := FindAllShardPaths(single); len(s) != 1 || s[0] != single {
		t.Errorf("expected single path, got %v", s)
	}

	// 3. Shard URLs resolution
	u := "https://huggingface.co/repo/resolve/main/Model-00001-of-00003.gguf"
	urls := ResolveShardURLs(u, nil)
	if len(urls) != 3 {
		t.Fatalf("expected 3 shard URLs, got %d", len(urls))
	}
	if urls[2] != "https://huggingface.co/repo/resolve/main/Model-00003-of-00003.gguf" {
		t.Errorf("unexpected 3rd shard URL: %s", urls[2])
	}

	// 4. Free disk space query
	freeGB := GetFreeDiskSpaceGB(".")
	if freeGB <= 0 {
		t.Errorf("expected positive free disk space, got %f", freeGB)
	}

	// 5. AllShardsExist check with temp files
	tmpDir := t.TempDir()
	s1 := filepath.Join(tmpDir, "Test-00001-of-00002.gguf")
	s2 := filepath.Join(tmpDir, "Test-00002-of-00002.gguf")

	// Neither exists
	if AllShardsExist(s1) {
		t.Errorf("expected false when no shards exist")
	}

	// Only shard 1 exists
	os.WriteFile(s1, []byte("shard1"), 0644)
	if AllShardsExist(s1) {
		t.Errorf("expected false when shard 2 missing")
	}

	// Shard 2 exists but is smaller than required 100MB for real weights
	os.WriteFile(s2, []byte("shard2"), 0644)
	if AllShardsExist(s1) {
		t.Errorf("expected false when multi-shard weight shard is tiny/stub")
	}

	// Shard 2 with fake 101MB payload
	f, _ := os.Create(s2)
	f.Truncate(101 * 1024 * 1024)
	f.Close()
	if !AllShardsExist(s1) {
		t.Errorf("expected true when all shards exist and have weight data")
	}
}

func TestDeleteModelFilesAndSecurity(t *testing.T) {
	// 1. Security rejection of paths outside models/packs
	unsafePaths := []string{
		"/etc/passwd",
		"../../../../etc/shadow",
		"/tmp/random_file.gguf",
		"",
	}
	for _, p := range unsafePaths {
		if IsSafeModelPath(p) {
			t.Errorf("expected IsSafeModelPath to reject %q", p)
		}
		if _, _, err := DeleteModelFiles(p); err == nil {
			t.Errorf("expected DeleteModelFiles to reject unsafe path %q", p)
		}
	}

	// 2. Allowed model directory safety
	if !IsSafeModelPath("models/test.gguf") {
		t.Errorf("expected IsSafeModelPath to allow models/test.gguf")
	}
	if !IsSafeModelPath("packs/test_pack") {
		t.Errorf("expected IsSafeModelPath to allow packs/test_pack")
	}
	// Container roots themselves must not be deletable
	if IsSafeModelPath("models") || IsSafeModelPath("models/") {
		t.Errorf("root models directory should not be considered deletable target")
	}

	// 3. Test deleting files inside models/
	os.MkdirAll("models", 0755)
	testShard1 := "models/TestDelete-00001-of-00002.gguf"
	testShard2 := "models/TestDelete-00002-of-00002.gguf"
	testPart := "models/TestDelete-00002-of-00002.gguf.part"

	os.WriteFile(testShard1, []byte("12345"), 0644)
	os.WriteFile(testShard2, []byte("6789012345"), 0644)
	os.WriteFile(testPart, []byte("part_content"), 0644)

	if !HasAnyShardOrPart(testShard1) {
		t.Errorf("expected HasAnyShardOrPart to be true for testShard1")
	}

	freedBytes, deleted, err := DeleteModelFiles("TestDelete-00001-of-00002.gguf")
	if err != nil {
		t.Fatalf("DeleteModelFiles failed: %v", err)
	}

	expectedBytes := int64(5 + 10 + 12)
	if freedBytes != expectedBytes {
		t.Errorf("expected %d freed bytes, got %d", expectedBytes, freedBytes)
	}
	if len(deleted) != 3 {
		t.Errorf("expected 3 deleted files, got %d: %v", len(deleted), deleted)
	}

	// Verify files actually removed
	if _, err := os.Stat(testShard1); !os.IsNotExist(err) {
		t.Errorf("testShard1 was not deleted")
	}
	if _, err := os.Stat(testShard2); !os.IsNotExist(err) {
		t.Errorf("testShard2 was not deleted")
	}
	if _, err := os.Stat(testPart); !os.IsNotExist(err) {
		t.Errorf("testPart was not deleted")
	}
}

