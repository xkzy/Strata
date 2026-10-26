//go:build cuda

package densecpu

import (
	"testing"
	"time"
)

// decodeAt fills a session with n filler tokens (in chunks of 512), then times one more decode step.
func decodeAt(t *testing.T, name string, n int, feed func(ids []int), step func()) {
	GPUCallNanos.Store(0)
	t0 := time.Now()
	for done := 0; done < n; done += 512 {
		c := min(512, n-done)
		ids := make([]int, c)
		for i := range ids {
			ids[i] = 760 + (done+i)%50
		}
		feed(ids)
	}
	pre := time.Since(t0)
	gpu := float64(GPUCallNanos.Load()) / 1e9
	t0 = time.Now()
	step()
	t.Logf("%s: context %d, prefill %.1fs (%.1fs in GPU calls), one decode step %.3fs", name, n, pre.Seconds(), gpu, time.Since(t0).Seconds())
}

// TestPrefill512 is the short run to profile: go test -tags cuda -run Prefill512 -cpuprofile cpu.out
func TestPrefill512(t *testing.T) {
	q, err := OpenQwen(modelPath(t, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	if _, err := q.EnableGPU(); err != nil {
		t.Fatal(err)
	}
	st := q.NewState()
	decodeAt(t, "qwen", 512, func(ids []int) { q.Step(st, ids) }, func() { q.Step(st, []int{760}) })
}

func TestDecodeVsContext(t *testing.T) {
	if testing.Short() {
		t.Skip("slow")
	}
	q, err := OpenQwen(modelPath(t, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	if _, err := q.EnableGPU(); err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{512, 2048, 4096} {
		st := q.NewState()
		decodeAt(t, "qwen", n, func(ids []int) { q.Step(st, ids) }, func() { q.Step(st, []int{760}) })
	}
	q.Close()
	g, err := OpenGemma(modelPath(t, "gemma-4-31B-it-UD-Q4_K_XL.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	if _, err := g.EnableGPU(); err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{512, 2048, 4096} {
		st := g.NewState()
		decodeAt(t, "gemma", n, func(ids []int) { g.Step(st, ids) }, func() { g.Step(st, []int{760}) })
	}
}
