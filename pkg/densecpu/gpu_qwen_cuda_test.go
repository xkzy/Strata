//go:build cuda

package densecpu

import (
	"math"
	"testing"
	"time"
)

func TestQwenResident(t *testing.T) {
	m, err := OpenQwen(modelPath(t, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.GreedyEncode("The capital of France is")
	ref := m.Forward(ids) // CPU
	t0 := time.Now()
	g, err := m.EnableResident(4096)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	t.Logf("upload %.1fs; segments %d", time.Since(t0).Seconds(), len(g.segs))
	for _, s := range g.segs {
		t.Logf("  device %d: %d layers", s.dev, s.n)
	}
	got := g.Step(ids)
	ProfDump()
	var md float64
	for i := range ref {
		md = math.Max(md, math.Abs(float64(ref[i]-got[i])))
	}
	t.Logf("top token cpu %q (%.3f) gpu %q (%.3f); max |logit diff| %.4f", m.Vocab[argmax(ref)], ref[argmax(ref)], m.Vocab[argmax(got)], got[argmax(got)], md)
	if argmax(ref) != argmax(got) {
		t.Fatal("top token differs")
	}
	ProfDump()
	// a prompt long enough (16+ tokens) for the tensor-core matmuls
	long := m.GreedyEncode("The quick brown fox jumps over the lazy dog. The capital of Germany is Berlin, and the capital of France is")
	t.Logf("long prompt: %d tokens", len(long))
	g.Reset()
	gotL := g.Step(long)
	refL := m.Forward(long)
	md = 0
	for i := range refL {
		md = math.Max(md, math.Abs(float64(refL[i]-gotL[i])))
	}
	t.Logf("long prompt: top cpu %q (%.3f) gpu %q (%.3f); max |logit diff| %.4f", m.Vocab[argmax(refL)], refL[argmax(refL)], m.Vocab[argmax(gotL)], gotL[argmax(gotL)], md)
	if len(long) < 16 || argmax(refL) != argmax(gotL) {
		t.Fatal("long prompt: too short or top token differs")
	}
	// 200 tokens in one call (128-column tiles) against two calls of 100 (32-column tiles): same math, so the logits must agree
	f200 := make([]int, 200)
	for i := range f200 {
		f200[i] = 760 + i%37
	}
	g.Reset()
	wide := g.Step(f200)
	g.Reset()
	g.chunk = 100
	narrow := g.Step(f200)
	g.chunk = 512
	md = 0
	for i := range wide {
		md = math.Max(md, math.Abs(float64(wide[i]-narrow[i])))
	}
	t.Logf("wide vs narrow tiles: max |logit diff| %.5f", md)
	if md > 0.05 || argmax(wide) != argmax(narrow) {
		t.Fatal("wide tiles disagree with narrow tiles")
	}
	g.Reset()
	g.Step(ids)
	// decode: 16 more tokens greedily, timed
	tok := argmax(got)
	t0 = time.Now()
	var out []int
	for i := 0; i < 16; i++ {
		out = append(out, tok)
		tok = argmax(g.Step([]int{tok}))
	}
	ProfDump()
	t.Logf("decode %.1f ms/token: %v", time.Since(t0).Seconds()*1000/16, out)
	// prefill speed
	g.Reset()
	filler := make([]int, 512)
	for i := range filler {
		filler[i] = 760 + i%50
	}
	t0 = time.Now()
	g.Step(filler)
	ProfDump()
	t.Logf("prefill 512 tokens: %.2fs", time.Since(t0).Seconds())
}
