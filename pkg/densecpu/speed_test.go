package densecpu

import (
	"testing"
	"time"
)

// One-token pass = the weight traffic and arithmetic of one decode step (no KV cache here, so a real decode step
// would add only attention over the cache). Logged, not asserted: it is a measurement.
func TestPassSpeed(t *testing.T) {
	g, err := OpenGemma(modelPath(t, "gemma-4-31B-it-UD-Q4_K_XL.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	g.Forward([]int{2}) // warm the page cache
	t0 := time.Now()
	g.Forward([]int{2})
	t.Logf("gemma 1-token pass: %.2fs", time.Since(t0).Seconds())

	q, err := OpenQwen(modelPath(t, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	q.Forward([]int{760})
	t0 = time.Now()
	q.Forward([]int{760})
	t.Logf("qwen 1-token pass: %.2fs", time.Since(t0).Seconds())
}
