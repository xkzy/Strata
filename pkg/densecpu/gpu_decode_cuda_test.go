//go:build cuda

package densecpu

import (
	"os"
	"strconv"
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
	in, kern, out, bi, bo, calls := MatvecStats(true)
	t.Logf("matvec over the prefill: %d calls, copy in %.2fs (%.0f MB), kernel %.2fs, copy out %.2fs (%.0f MB)", calls, in, bi/1e6, kern, out, bo/1e6)
	t0 = time.Now()
	step()
	t.Logf("%s: context %d, prefill %.1fs (%.1fs in GPU calls), one decode step %.3fs", name, n, pre.Seconds(), gpu, time.Since(t0).Seconds())
	in, kern, out, bi, bo, calls = MatvecStats(true)
	t.Logf("matvec over the decode step: %d calls, copy in %.3fs (%.1f MB), kernel %.3fs, copy out %.3fs (%.1f MB)", calls, in, bi/1e6, kern, out, bo/1e6)
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
	defer st.Free()
	decodeAt(t, "qwen", 512, func(ids []int) { q.Step(st, ids) }, func() { q.Step(st, []int{760}) })
}

func TestPrefill512Gemma(t *testing.T) {
	g, err := OpenGemma(modelPath(t, "gemma-4-31B-it-UD-Q4_K_XL.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	if _, err := g.EnableGPU(); err != nil {
		t.Fatal(err)
	}
	st := g.NewState()
	defer st.Free()
	decodeAt(t, "gemma", 512, func(ids []int) { g.Step(st, ids) }, func() { g.Step(st, []int{760}) })
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
		st.Free()
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
		st.Free()
	}
}

// TestMaxContext fills a session toward STRATA_CTX tokens (model: STRATA_CTX_MODEL=qwen|gemma) and times a decode step
// each time the context doubles. Skipped unless STRATA_CTX is set.
func TestMaxContext(t *testing.T) {
	target, _ := strconv.Atoi(os.Getenv("STRATA_CTX"))
	if target == 0 {
		t.Skip("set STRATA_CTX")
	}
	var step func(ids []int)
	if os.Getenv("STRATA_CTX_MODEL") == "gemma" {
		g, err := OpenGemma(modelPath(t, "gemma-4-31B-it-UD-Q4_K_XL.gguf"))
		if err != nil {
			t.Fatal(err)
		}
		defer g.Close()
		if _, err := g.EnableGPU(); err != nil {
			t.Fatal(err)
		}
		st := g.NewState()
		defer st.Free()
		step = func(ids []int) { g.Step(st, ids) }
	} else {
		q, err := OpenQwen(modelPath(t, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"))
		if err != nil {
			t.Fatal(err)
		}
		defer q.Close()
		if _, err := q.EnableGPU(); err != nil {
			t.Fatal(err)
		}
		st := q.NewState()
		defer st.Free()
		step = func(ids []int) { q.Step(st, ids) }
	}
	t0, next := time.Now(), 4096
	for done := 0; done < target; done += 512 {
		ids := make([]int, 512)
		for i := range ids {
			ids[i] = 760 + (done+i)%50
		}
		step(ids)
		if done+512 >= next || done+512 >= target {
			t1 := time.Now()
			step([]int{760})
			done++ // the probe token is part of the context
			t.Logf("context %d: %.0fs elapsed, decode step %.3fs", done+512, t1.Sub(t0).Seconds(), time.Since(t1).Seconds())
			next *= 2
		}
	}
}
