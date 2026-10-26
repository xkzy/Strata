package densecpu

import (
	"math"
	"testing"
)

func sameLogits(t *testing.T, a, b []float32) {
	t.Helper()
	var worst float64
	for i := range a {
		worst = math.Max(worst, math.Abs(float64(a[i]-b[i])))
	}
	if worst > 1e-2 || argmax(a) != argmax(b) {
		t.Fatalf("incremental and full logits differ: max |diff| %.4g, top %d vs %d", worst, argmax(a), argmax(b))
	}
}

func TestQwenStepMatchesForward(t *testing.T) {
	if testing.Short() {
		t.Skip("slow")
	}
	m, err := OpenQwen(modelPath(t, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.GreedyEncode("The capital of France is Paris. The capital of Germany is")
	st := m.NewState()
	m.Step(st, ids[:3]) // a chunk, then one token at a time
	var got []float32
	for _, id := range ids[3:] {
		got = m.Step(st, []int{id})
	}
	sameLogits(t, m.Forward(ids), got)
}

func TestGemmaStepMatchesForward(t *testing.T) {
	if testing.Short() {
		t.Skip("slow")
	}
	m, err := OpenGemma(modelPath(t, "gemma-4-31B-it-UD-Q4_K_XL.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	m.window = 4 // make the sliding layers trim their cache within a short test
	ids := append([]int{2}, m.GreedyEncode("The capital of France is Paris. The capital of Germany is")...)
	st := m.NewState()
	m.Step(st, ids[:5])
	var got []float32
	for _, id := range ids[5:] {
		got = m.Step(st, []int{id})
	}
	sameLogits(t, m.Forward(ids), got)
}
