package densecpu

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

func flintPath(t *testing.T) string {
	if testing.Short() {
		t.Skip("slow")
	}
	p := filepath.Join(os.Getenv("HOME"), "Downloads/Strata-data/models/flint-UD-Q4_K_XL.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skip("model not present")
	}
	return p
}

// Slow (CPU pass over 23 GB) and needs the real file.
func TestFlintParis(t *testing.T) {
	m, err := OpenFlint(flintPath(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.GreedyEncode("The capital of France is")
	logits := m.Forward(ids)
	idx := make([]int, len(logits))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
	top := false
	for _, i := range idx[:5] {
		t.Logf("%q %.3f", m.Vocab[i], logits[i])
		top = top || strings.Contains(m.Vocab[i], "Paris")
	}
	if !top {
		t.Fatalf("Paris is not in the top 5 for %v", ids)
	}
}

func TestFlintChat(t *testing.T) {
	m, err := OpenFlint(flintPath(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.GreedyEncode("<|user|>What is the capital of France?<|assistant|>\n")
	st := m.NewState()
	var out []int
	for i := 0; i < 24; i++ {
		lg := m.Step(st, ids)
		best := 0
		for j, v := range lg {
			if v > lg[best] {
				best = j
			}
		}
		out = append(out, best)
		ids = []int{best}
	}
	t.Logf("%q", m.Decode(out))
}

// A fact placed beyond the 513-token window can only be read by the full-attention layers.
func TestFlintNeedle(t *testing.T) {
	m, err := OpenFlint(flintPath(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	var sb strings.Builder
	sb.WriteString("<|user|>Remember this: the vault code is 7319.\n")
	for i := 0; i < 45; i++ {
		sb.WriteString("The weather report for day " + strconv.Itoa(i+1) + " says clouds drift slowly over the hills while rivers run quietly.\n")
	}
	sb.WriteString("What is the vault code? Answer with the number only.<|assistant|>\n</think>")
	ids := m.GreedyEncode(sb.String())
	t.Logf("%d prompt tokens", len(ids))
	st := m.NewState()
	var out []int
	for i := 0; i < 8; i++ {
		lg := m.Step(st, ids)
		best := 0
		for j, v := range lg {
			if v > lg[best] {
				best = j
			}
		}
		out = append(out, best)
		ids = []int{best}
	}
	got := m.Decode(out)
	t.Logf("%q", got)
	if !strings.Contains(got, "7319") {
		t.Fatalf("needle lost: %q", got)
	}
}

// nll is the mean negative log-likelihood of text under the model, one token per step.
func (m *Flint) nll(text string) float64 {
	ids := m.GreedyEncode(text)
	st := m.NewState()
	var tot float64
	for i := 0; i+1 < len(ids); i++ {
		lg := m.Step(st, ids[i:i+1])
		mx := lg[0]
		for _, v := range lg {
			if v > mx {
				mx = v
			}
		}
		var z float64
		for _, v := range lg {
			z += math.Exp(float64(v - mx))
		}
		tot += math.Log(z) - float64(lg[ids[i+1]]-mx)
	}
	return tot / float64(len(ids)-1)
}

// Each guessed detail, flipped alone, must make plain English worse.
func TestFlintAblations(t *testing.T) {
	m, err := OpenFlint(flintPath(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	text := "The old lighthouse keeper climbed the spiral stairs every evening to light the lamp. Ships far out at sea relied on its steady beam to find the harbor, and the village below slept soundly because of it."
	base := m.nll(text)
	t.Logf("base nll %.3f (%d tokens)", base, len(m.GreedyEncode(text)))
	for name, a := range map[string]flintAblation{
		"no sink": {noSink: true}, "interleaved rope": {interleavedRope: true},
		"rope on full layers": {ropeEverywhere: true}, "no top-k norm": {noNormTopK: true}} {
		m.abl = a
		v := m.nll(text)
		t.Logf("%-20s nll %.3f (%+.3f)", name, v, v-base)
		if v < base-0.02 {
			t.Errorf("%s is better than the model's reading: %.3f < %.3f", name, v, base)
		}
	}
}

// TestFlintDumpLayers writes, for a few blocks, every step of a prefill followed by single-token decodes - the hidden rows
// going in and coming out, one cache per block - for the CUDA parity harness (flint_layer_parity).
// Set FLINT_FIXTURE_DIR to run it. File: int32 {magic, layer, n_embd, window, steps}, then per step int32 n, in, out (fp32).
func TestFlintDumpLayers(t *testing.T) {
	dir := os.Getenv("FLINT_FIXTURE_DIR")
	if dir == "" {
		t.Skip("FLINT_FIXTURE_DIR not set")
	}
	m, err := OpenFlint(flintPath(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.GreedyEncode("The old lighthouse keeper climbed the spiral stairs every evening to light the lamp")
	const prefill = 8
	chunks := [][]int{ids[:prefill]}
	for _, id := range ids[prefill:] {
		chunks = append(chunks, []int{id})
	}
	emb, eb := m.raw("token_embd.weight")
	ec := int(emb.Dims[0])
	elems, bytes, _ := blockInfo(emb.Type)
	embed := func(c []int) [][]float32 {
		h := make([][]float32, len(c))
		for i, tk := range c {
			h[i] = make([]float32, ec)
			if err := DequantRow(emb.Type, eb[tk*(ec/elems*bytes):], ec, h[i]); err != nil {
				t.Fatal(err)
			}
		}
		return h
	}
	clone := func(x [][]float32) [][]float32 {
		y := make([][]float32, len(x))
		for i := range x {
			y[i] = append([]float32(nil), x[i]...)
		}
		return y
	}
	type step struct{ in, out [][]float32 }
	layers := []int{0, 2, 4, 8, 19, 21, 42}
	want := map[int][]step{}
	for _, l := range layers {
		want[l] = make([]step, len(chunks))
	}
	st := m.NewState()
	pos := 0
	for si, c := range chunks {
		h := embed(c)
		for l := 0; l < 43; l++ {
			if _, ok := want[l]; ok {
				want[l][si].in = clone(h)
			}
			m.layer(st, l, h, pos)
			if _, ok := want[l]; ok {
				want[l][si].out = clone(h)
			}
		}
		pos += len(c)
	}
	write := func(l, window int, steps []step) {
		var raw []byte
		put := func(v uint32) { raw = binary.LittleEndian.AppendUint32(raw, v) }
		for _, v := range []uint32{0x464c4e32, uint32(l), uint32(ec), uint32(window), uint32(len(steps))} {
			put(v)
		}
		for _, sp := range steps {
			put(uint32(len(sp.in)))
			for _, rows := range [][][]float32{sp.in, sp.out} {
				for _, r := range rows {
					for _, v := range r {
						put(math.Float32bits(v))
					}
				}
			}
		}
		name := filepath.Join(dir, "flint_layer_"+strconv.Itoa(l)+"_w"+strconv.Itoa(window)+".bin")
		if err := os.WriteFile(name, raw, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, l := range layers {
		write(l, m.window, want[l])
	}
	// the same inputs through block 2 with a window of 3: decode must drop rows that fall out of the window
	var narrow []step
	st2, old := m.NewState(), m.window
	m.window, pos = 3, 0
	for _, sp := range want[2] {
		x := clone(sp.in)
		m.layer(st2, 2, x, pos)
		narrow = append(narrow, step{sp.in, x})
		pos += len(x)
	}
	m.window = old
	write(2, 3, narrow)
}

// TestFlintDumpReference writes a chat prompt, the model's greedy continuation and the top-10 logits at each step, for
// the CUDA full-model parity run (flint_model_parity). Set FLINT_FIXTURE_DIR to run it.
// File: int32 {magic, n_prompt, n_gen, topk}, prompt ids, then n_gen+1 steps of topk int32 ids and topk fp32 logits (the
// first step follows the prompt, each next one follows the previous greedy token).
func TestFlintDumpReference(t *testing.T) {
	dir := os.Getenv("FLINT_FIXTURE_DIR")
	if dir == "" {
		t.Skip("FLINT_FIXTURE_DIR not set")
	}
	m, err := OpenFlint(flintPath(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const topk, nGen = 10, 8
	prompt := m.GreedyEncode("<|user|>What is the capital of France?<|assistant|>\n")
	raw := binary.LittleEndian.AppendUint32(nil, 0x464c4e33)
	for _, v := range []int{len(prompt), nGen, topk} {
		raw = binary.LittleEndian.AppendUint32(raw, uint32(v))
	}
	for _, id := range prompt {
		raw = binary.LittleEndian.AppendUint32(raw, uint32(id))
	}
	st := m.NewState()
	feed := prompt
	var gen []int
	for s := 0; s <= nGen; s++ {
		lg := m.Step(st, feed)
		idx := make([]int, len(lg))
		for i := range idx {
			idx[i] = i
		}
		sort.SliceStable(idx, func(a, b int) bool { return lg[idx[a]] > lg[idx[b]] })
		for _, i := range idx[:topk] {
			raw = binary.LittleEndian.AppendUint32(raw, uint32(i))
		}
		for _, i := range idx[:topk] {
			raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(lg[i]))
		}
		gen = append(gen, idx[0])
		feed = []int{idx[0]}
	}
	// the prompt's last-row hidden state after every block, to see how the GPU's error grows with depth
	emb, eb := m.raw("token_embd.weight")
	ec := int(emb.Dims[0])
	elems, bytes, _ := blockInfo(emb.Type)
	h := make([][]float32, len(prompt))
	for i, tk := range prompt {
		h[i] = make([]float32, ec)
		if err := DequantRow(emb.Type, eb[tk*(ec/elems*bytes):], ec, h[i]); err != nil {
			t.Fatal(err)
		}
	}
	tr := binary.LittleEndian.AppendUint32(nil, 0x464c4e34)
	tr = binary.LittleEndian.AppendUint32(tr, uint32(m.nLayer))
	tr = binary.LittleEndian.AppendUint32(tr, uint32(ec))
	tst := m.NewState()
	for l := 0; l < m.nLayer; l++ {
		m.layer(tst, l, h, 0)
		for _, v := range h[len(h)-1] {
			tr = binary.LittleEndian.AppendUint32(tr, math.Float32bits(v))
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "flint_trace.bin"), tr, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("%q", m.Decode(gen))
	if err := os.WriteFile(filepath.Join(dir, "flint_reference.bin"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestFlintQ8Noise is the noise floor for the GPU comparison: the same teacher-forced run with the activations rounded to
// q8_1 (what the GPU kernels see), against the exact run. The total variation over the exact run's top-10 is the same
// measure flint_model_parity prints.
func TestFlintQ8Noise(t *testing.T) {
	m, err := OpenFlint(flintPath(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const nGen = 8
	prompt := m.GreedyEncode("<|user|>What is the capital of France?<|assistant|>\n")
	run := func(forced []int) (logits [][]float32, gen []int) {
		st := m.NewState()
		feed := prompt
		for s := 0; s <= nGen; s++ {
			lg := m.Step(st, feed)
			best := 0
			for i, v := range lg {
				if v > lg[best] {
					best = i
				}
			}
			logits, gen = append(logits, lg), append(gen, best)
			if forced != nil {
				best = forced[s]
			}
			feed = []int{best}
		}
		return
	}
	exact, gen := run(nil)
	quantActs = true
	defer func() { quantActs = false }()
	noisy, _ := run(gen)
	for s := range exact {
		idx := make([]int, len(exact[s]))
		for i := range idx {
			idx[i] = i
		}
		sort.SliceStable(idx, func(a, b int) bool { return exact[s][idx[a]] > exact[s][idx[b]] })
		top := idx[:10]
		best := 0
		for i, v := range noisy[s] {
			if v > noisy[s][best] {
				best = i
			}
		}
		var zr, zn float64
		for _, i := range top {
			zr += math.Exp(float64(exact[s][i] - exact[s][top[0]]))
			zn += math.Exp(float64(noisy[s][i] - noisy[s][best]))
		}
		var tv, dl float64
		for _, i := range top {
			tv += math.Abs(math.Exp(float64(exact[s][i]-exact[s][top[0]]))/zr - math.Exp(float64(noisy[s][i]-noisy[s][best]))/zn)
			dl = math.Max(dl, math.Abs(float64(exact[s][i]-noisy[s][i])))
		}
		t.Logf("step %d: top1 %v  max |dlogit| %.3f  TV %.3f", s, best == top[0], dl, tv/2)
	}
}
