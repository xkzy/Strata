package densecpu

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func argmax(x []float32) int {
	b := 0
	for i, v := range x {
		if v > x[b] {
			b = i
		}
	}
	return b
}

// greedy continues ids by n tokens, re-running the whole sequence each step (no KV cache: this is a reference).
func greedy(fwd func([]int) []float32, ids []int, n int) []int {
	ids = append([]int(nil), ids...)
	for i := 0; i < n; i++ {
		ids = append(ids, argmax(fwd(ids)))
	}
	return ids
}

func modelPath(t *testing.T, name string) string {
	if testing.Short() {
		t.Skip("slow")
	}
	p := filepath.Join(os.Getenv("HOME"), "Downloads/Strata-data/models", name)
	if _, err := os.Stat(p); err != nil {
		t.Skip("model not present")
	}
	return p
}

func text(vocab []string, ids []int, space string) string {
	var sb strings.Builder
	for _, id := range ids {
		sb.WriteString(strings.ReplaceAll(vocab[id], space, " "))
	}
	return sb.String()
}

func TestGemmaContinuation(t *testing.T) {
	m, err := OpenGemma(modelPath(t, "gemma-4-31B-it-UD-Q4_K_XL.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := append([]int{2}, m.GreedyEncode("The capital of France is")...)
	out := greedy(m.Forward, ids, 12)
	s := text(m.Vocab, out[len(ids):], "▁")
	t.Logf("gemma continuation: %q", s)
	if !strings.Contains(s, "Paris") {
		t.Fatalf("continuation %q does not mention Paris", s)
	}
}

func TestQwenContinuation(t *testing.T) {
	m, err := OpenQwen(modelPath(t, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.GreedyEncode("The capital of France is")
	out := greedy(m.Forward, ids, 12)
	s := text(m.Vocab, out[len(ids):], "Ġ")
	t.Logf("qwen continuation: %q", s)
	if !strings.Contains(s, "Paris") {
		t.Fatalf("continuation %q does not mention Paris", s)
	}
}
