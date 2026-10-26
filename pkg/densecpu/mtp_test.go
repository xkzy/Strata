package densecpu

import "testing"

// The MTP block drafts token i+2 from the main model's hidden state at i and token i+1. Its drafts should mostly
// match what the main model itself continues with.
func TestQwenMTPAgreesWithMainModel(t *testing.T) {
	m, err := OpenQwen(modelPath(t, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompt := m.GreedyEncode("The capital of France is")
	ids := greedy(m.Forward, prompt, 12)
	hs := m.hidden(ids)
	best := -1.0
	for off := 0; off <= 1; off++ {
		logits := m.MTP(ids, hs, off)
		hit := 0
		for i := range logits {
			if i+2 < len(ids) && argmax(logits[i]) == ids[i+2] {
				hit++
			}
		}
		rate := float64(hit) / float64(len(ids)-2)
		t.Logf("rope position offset %d: %d/%d drafts match the main model (%.0f%%)", off, hit, len(ids)-2, 100*rate)
		best = max(best, rate)
	}
	if best < 0.5 {
		t.Fatalf("best agreement %.0f%% < 50%%: the MTP block is not running correctly", 100*best)
	}
}
