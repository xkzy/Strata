//go:build cuda

package densecpu

import (
	"math"
	"os"
	"strconv"
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
	t.Logf("copy traffic GB/s: dev0 %.0f dev1 %.0f", DeviceBandwidth(0), DeviceBandwidth(1))
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

func TestQwenDecodeDepth(t *testing.T) {
	m, err := OpenQwen(modelPath(t, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := m.EnableResident(16384)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	filler := make([]int, 1024)
	for i := range filler {
		filler[i] = 760 + i%50
	}
	for _, depth := range depths() {
		g.Reset()
		t0 := time.Now()
		for g.pos < depth {
			g.Step(filler[:min(1024, depth-g.pos)])
		}
		pf := time.Since(t0).Seconds()
		ProfDump()
		g.Step([]int{760})
		t0 = time.Now()
		for i := 0; i < 8; i++ {
			g.Step([]int{760 + i})
		}
		t.Logf("depth %d: prefill %.1fs (%.0f tok/s), decode %.1f ms/token", depth, pf, float64(depth)/pf, time.Since(t0).Seconds()*1000/8)
	}
}

// the same last token through the split attention (n = 1, long context) and through the plain one (n = 10)
func TestQwenSplitAttention(t *testing.T) {
	m, err := OpenQwen(modelPath(t, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := m.EnableResident(4096)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	words := []string{"river", "mountain", "garden", "city", "winter", "story", "engine", "bridge", "market", "forest", "letter", "window", "island", "music", "harvest"}
	text := ""
	for i := 0; len(m.GreedyEncode(text)) < 1500; i++ {
		text += "The " + words[i%len(words)] + " near the " + words[(i*7+3)%len(words)] + " was quiet in the morning, and then the " + words[(i*5+1)%len(words)] + " changed. "
	}
	seq := m.GreedyEncode(text)[:1500]
	g.Reset()
	g.Step(seq[:1490])
	plain := g.Step(seq[1490:])
	g.Reset()
	g.Step(seq[:1499])
	split := g.Step(seq[1499:])
	var md float64
	for i := range plain {
		md = math.Max(md, math.Abs(float64(plain[i]-split[i])))
	}
	t.Logf("split vs plain attention: max |logit diff| %.5f, top %q / %q", md, m.Vocab[argmax(plain)], m.Vocab[argmax(split)])
	g.Reset()
	g.Step(seq[:1499])
	again := g.Step(seq[1499:])
	var mr float64
	for i := range again {
		mr = math.Max(mr, math.Abs(float64(again[i]-split[i])))
	}
	t.Logf("same run twice: max |logit diff| %.5f", mr)
	// the same single token with and without splitting the positions: only the order of the attention sums differs
	g.Reset()
	g.Step(seq[:1499])
	os.Setenv("DG_NOSPLIT", "1")
	one := g.Step(seq[1499:])
	os.Unsetenv("DG_NOSPLIT")
	var ms float64
	for i := range one {
		ms = math.Max(ms, math.Abs(float64(one[i]-split[i])))
	}
	t.Logf("same token, split vs one block: max |logit diff| %.5f", ms)
	// not exactly equal: the int8 inputs of every matmul turn the last bits of a different sum order into rounding steps
	if ms > 0.8 || argmax(one) != argmax(split) {
		t.Fatal("split attention differs from the single-block one")
	}
	// baseline: the same comparison where nothing is split (positions < 256) shows the noise of n = 10 against n = 1 alone
	g.Reset()
	g.Step(seq[:190])
	p0 := g.Step(seq[190:200])
	g.Reset()
	g.Step(seq[:199])
	s0 := g.Step(seq[199:200])
	var mb float64
	for i := range p0 {
		mb = math.Max(mb, math.Abs(float64(p0[i]-s0[i])))
	}
	t.Logf("baseline without splitting: max |logit diff| %.5f", mb)
	if mb > 0.8 {
		t.Fatal("baseline too noisy")
	}
	if argmax(plain) != argmax(split) {
		t.Fatal("split attention disagrees")
	}
}

func TestQwenBatchVsSingles(t *testing.T) {
	m, err := OpenQwen(modelPath(t, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.GreedyEncode("The quick brown fox jumps over the lazy dog and then the capital of France is")
	ref := m.Forward(ids)
	g, err := m.EnableResident(4096)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	batch := g.Step(ids)
	g.Reset()
	var single []float32
	for _, id := range ids {
		single = g.Step([]int{id})
	}
	d := func(a, b []float32) (md float64) {
		for i := range a {
			md = math.Max(md, math.Abs(float64(a[i]-b[i])))
		}
		return
	}
	t.Logf("%d tokens: batch vs cpu %.3f, singles vs cpu %.3f, batch vs singles %.3f; tops %q %q %q", len(ids), d(batch, ref), d(single, ref), d(batch, single),
		m.Vocab[argmax(ref)], m.Vocab[argmax(batch)], m.Vocab[argmax(single)])
}

func TestQwenSpecDecode(t *testing.T) {
	m, err := OpenQwen(modelPath(t, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := m.EnableResident(4096)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	if ok, err := g.EnableMTP(); !ok {
		t.Fatalf("no MTP block: %v", err)
	}
	prompts := []string{
		"The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is",
		"Write a short story about a lighthouse keeper who finds a message in a bottle. The story begins:",
		"Here is a Python function that returns the n-th Fibonacci number using recursion with memoization, followed by a short explanation of how it works:",
	}
	const N = 64
	for _, pr := range prompts {
		ids := m.GreedyEncode(pr)
		t0 := time.Now()
		plain, _ := g.Generate(ids, N, -1, false)
		tp := time.Since(t0)
		t0 = time.Now()
		sp, st := g.Generate(ids, N, -1, true)
		ts := time.Since(t0)
		same := 0
		for same < len(plain) && same < len(sp) && plain[same] == sp[same] {
			same++
		}
		t.Logf("%.30q: plain %.1f tok/s, spec %.1f tok/s; accepted %d of %d drafts; first %d of %d tokens identical", pr, float64(N)/tp.Seconds(), float64(N)/ts.Seconds(), st.Accepted, st.Cycles, same, N)
	}
}

func depths() []int {
	if os.Getenv("DG_PROF") != "" {
		return []int{8192}
	}
	return []int{1024, 4096, 8192, 15000}
}

// three chunks in one call (pipelined over the cards) against three calls of one chunk each: the same arithmetic, so identical logits
func TestQwenPipelineEqual(t *testing.T) {
	m, err := OpenQwen(modelPath(t, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := m.EnableResident(4096)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	text := ""
	for i := 0; len(m.GreedyEncode(text)) < 1300; i++ {
		text += "Chapter " + strconv.Itoa(i) + " begins with the harbor at dawn, and the boats leave one after another. "
	}
	seq := m.GreedyEncode(text)[:1300]
	g.Reset()
	piped := g.Step(seq)
	g.Reset()
	g.Step(seq[:512])
	g.Step(seq[512:1024])
	plain := g.Step(seq[1024:])
	var md float64
	for i := range piped {
		md = math.Max(md, math.Abs(float64(piped[i]-plain[i])))
	}
	t.Logf("pipelined vs sequential chunks: max |logit diff| %.6f", md)
	// the tensor-core attention against the scalar one on the same 1300 tokens (the scalar one is picked with DG_NOATTNMMA)
	os.Setenv("DG_NOATTNMMA", "1")
	g.Reset()
	scalar := g.Step(seq)
	os.Unsetenv("DG_NOATTNMMA")
	var ma float64
	for i := range piped {
		ma = math.Max(ma, math.Abs(float64(piped[i]-scalar[i])))
	}
	t.Logf("tensor-core vs scalar attention: max |logit diff| %.4f, top %q / %q", ma, m.Vocab[argmax(piped)], m.Vocab[argmax(scalar)])
	if ma > 0.8 || argmax(piped) != argmax(scalar) {
		t.Fatal("tensor-core attention disagrees with the scalar one")
	}
	if md != 0 {
		t.Fatal("pipelined result differs")
	}
}
