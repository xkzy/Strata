package densecpu

import (
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// Slow (a CPU pass over 17 GB of weights) and needs the real file: skipped when it is absent or with -short.
func TestGemmaParis(t *testing.T) {
	if testing.Short() {
		t.Skip("slow")
	}
	p := filepath.Join(os.Getenv("HOME"), "Downloads/Strata-data/models/gemma-4-31B-it-UD-Q4_K_XL.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skip("model not present")
	}
	m, err := OpenGemma(p)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := append([]int{2}, m.GreedyEncode("The capital of France is")...) // 2 = <bos>
	logits := m.Forward(ids)
	idx := make([]int, len(logits))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
	for _, i := range idx[:5] {
		t.Logf("%q %.3f", m.Vocab[i], logits[i])
	}
	for _, i := range idx[:3] {
		if m.Vocab[i] == "▁Paris" {
			return
		}
	}
	t.Fatalf("▁Paris is not in the top 3 for %v", ids)
}
