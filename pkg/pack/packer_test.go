package pack

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"strata/pkg/gguf"
)

func createMoEGGUF(t *testing.T, path string) {
	w := gguf.NewWriter()
	w.SetMetadata("general.architecture", "qwen3.8-moe")
	w.SetMetadata("expert_count", uint32(4))

	// Layer 0 experts: gate, up, down
	// Dims: [hidden, experts]
	dummyGate := make([]byte, 64*4*4) // F32
	dummyUp := make([]byte, 64*4*4)
	dummyDown := make([]byte, 64*4*4)

	w.AddTensor("blk.0.ffn_gate_exps.weight", []uint64{64, 4}, gguf.GGMLTypeF32, dummyGate)
	w.AddTensor("blk.0.ffn_up_exps.weight", []uint64{64, 4}, gguf.GGMLTypeF32, dummyUp)
	w.AddTensor("blk.0.ffn_down_exps.weight", []uint64{64, 4}, gguf.GGMLTypeF32, dummyDown)

	// Router weight
	dummyRouter := make([]byte, 64*4*4)
	w.AddTensor("blk.0.ffn_gate_inp.weight", []uint64{64, 4}, gguf.GGMLTypeF32, dummyRouter)

	// Attention norm
	dummyNorm := make([]byte, 64*4)
	w.AddTensor("blk.0.attn_norm.weight", []uint64{64}, gguf.GGMLTypeF32, dummyNorm)

	if err := w.WriteFile(path); err != nil {
		t.Fatalf("failed to create fixture GGUF: %v", err)
	}
}

func TestPackModel(t *testing.T) {
	tmpDir := t.TempDir()
	ggufPath := filepath.Join(tmpDir, "test-model.gguf")
	outDir := filepath.Join(tmpDir, "pack-out")

	createMoEGGUF(t, ggufPath)

	opts := PackOptions{
		GGUFPath:   ggufPath,
		OutDir:     outDir,
		ExpertsBin: true,
		CompatBF16: true,
	}

	err := PackModel(opts)
	if err != nil {
		t.Fatalf("PackModel failed: %v", err)
	}

	// Verify generated files
	indexFile := filepath.Join(outDir, "index.txt")
	if _, err := os.Stat(indexFile); err != nil {
		t.Errorf("index.txt missing: %v", err)
	}

	nativeFile := filepath.Join(outDir, "native_experts.txt")
	content, err := os.ReadFile(nativeFile)
	if err != nil {
		t.Fatalf("native_experts.txt missing: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(content)), "\n")
	if len(lines) == 0 || !strings.HasPrefix(lines[0], "0 ") {
		t.Errorf("expected layer 0 line in native_experts.txt, got %q", string(content))
	}

	denseFile := filepath.Join(outDir, "dense.bin")
	if _, err := os.Stat(denseFile); err != nil {
		t.Errorf("dense.bin missing: %v", err)
	}

	expertsFile := filepath.Join(outDir, "experts.bin")
	if _, err := os.Stat(expertsFile); err != nil {
		t.Errorf("experts.bin missing: %v", err)
	}
}

func TestGenerateDraftVocab(t *testing.T) {
	tokens := []string{"hello", "world", "你好", "世界", "bonjour", "café", "français"}
	// Chinese/Japanese/Korean
	cjkIndices, err := FilterDraftVocab(tokens, "cjk")
	if err != nil {
		t.Fatalf("FilterDraftVocab failed: %v", err)
	}
	if len(cjkIndices) != 2 {
		t.Errorf("expected 2 CJK tokens, got %d", len(cjkIndices))
	}

	// French
	frIndices, err := FilterDraftVocab(tokens, "fr")
	if err != nil {
		t.Fatalf("FilterDraftVocab fr failed: %v", err)
	}
	if len(frIndices) == 0 {
		t.Errorf("expected French tokens to be selected")
	}
}
