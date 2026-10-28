package densecpu

import (
	"fmt"
	"math"
	"sort"
	"strconv"
)

// Flint is a flint GGUF: 49 blocks of GQA attention (sliding window + rope, or full window without rope, each with
// per-head sinks) followed by a dense or a sigmoid-routed shared+routed MoE feed-forward.
// The layer semantics are read from tensor names and metadata, not from a reference implementation.
type Flint struct {
	*weights
	nEmbd, nLayer, window int
	nExp, nUsed, nDense   int
	eps, base, expScale   float32
	local                 []bool // sliding-window layers, which are also the rope layers
	normTopK              bool
	abl                   flintAblation
	Vocab                 []string
}

// flintAblation flips one guessed detail at a time; tests use it to show the guess matters. All false is the model.
type flintAblation struct {
	noSink, interleavedRope, ropeEverywhere, noNormTopK bool
}

// rope rotates one head of layer l at pos.
func (m *Flint) rope(v []float32, l, pos int) {
	if !m.local[l] && !m.abl.ropeEverywhere {
		return
	}
	if !m.abl.interleavedRope {
		ropeNeox(v, pos, len(v), m.base, nil)
		return
	}
	for i := 0; i < len(v)/2; i++ {
		th := float64(pos) * math.Pow(float64(m.base), -2*float64(i)/float64(len(v)))
		s, c := math.Sincos(th)
		a, b := float64(v[2*i]), float64(v[2*i+1])
		v[2*i], v[2*i+1] = float32(a*c-b*s), float32(a*s+b*c)
	}
}

// FlintState is a running session; pass it to Step with only the new tokens.
type FlintState struct {
	pos int
	kv  []*kvCache
}

// NewState starts an empty session.
func (m *Flint) NewState() *FlintState {
	s := &FlintState{kv: make([]*kvCache, m.nLayer)}
	for l := range s.kv {
		s.kv[l] = &kvCache{}
	}
	return s
}

// Pos is the number of tokens the session has consumed.
func (s *FlintState) Pos() int { return s.pos }

func boolPattern(m *Flint, key string) ([]bool, error) {
	pat, _ := m.g.Metadata[key].([]any)
	if len(pat) != m.nLayer {
		return nil, fmt.Errorf("densecpu: %s has %d entries for %d layers", key, len(pat), m.nLayer)
	}
	out := make([]bool, len(pat))
	for i, p := range pat {
		out[i], _ = p.(bool)
	}
	return out, nil
}

// OpenFlint opens a flint GGUF.
func OpenFlint(path string) (*Flint, error) {
	w, err := openWeights(path, "flint")
	if err != nil {
		return nil, err
	}
	g := w.g
	m := &Flint{weights: w,
		nEmbd: metaI(g, "flint.embedding_length"), nLayer: metaI(g, "flint.block_count"),
		window: metaI(g, "flint.attention.sliding_window"),
		nExp:   metaI(g, "flint.expert_count"), nUsed: metaI(g, "flint.expert_used_count"),
		nDense: metaI(g, "flint.leading_dense_block_count"),
		eps:    metaF(g, "flint.attention.layer_norm_rms_epsilon"), base: metaF(g, "flint.rope.freq_base"),
		expScale: metaF(g, "flint.expert_weights_scale")}
	m.normTopK, _ = g.Metadata["flint.expert_weights_norm"].(bool)
	if m.local, err = boolPattern(m, "flint.attention.sliding_window_pattern"); err == nil {
		var rp []bool
		if rp, err = boolPattern(m, "flint.attention.rope_pattern"); err == nil {
			for i := range rp {
				if rp[i] != m.local[i] {
					err = fmt.Errorf("densecpu: rope_pattern differs from sliding_window_pattern at layer %d", i)
					break
				}
			}
		}
	}
	if err == nil && g.Metadata["flint.expert_gating_func"] != nil && metaI(g, "flint.expert_gating_func") != 2 {
		err = fmt.Errorf("densecpu: expert_gating_func %d is not sigmoid", metaI(g, "flint.expert_gating_func"))
	}
	if err != nil {
		m.Close()
		return nil, err
	}
	m.Vocab = vocabOf(g)
	return m, nil
}

// gpt2Bytes maps every byte to the printable rune the GPT-2 byte-level vocab uses for it.
var gpt2Bytes = func() [256]rune {
	var t [256]rune
	n := 0
	for b := 0; b < 256; b++ {
		if (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF) {
			t[b] = rune(b)
		} else {
			t[b] = rune(256 + n)
			n++
		}
	}
	return t
}()

// GreedyEncode is the test-helper encoder (longest vocab match, no merges) over GPT-2 byte-level text.
func (m *Flint) GreedyEncode(text string) []int {
	rs := make([]rune, 0, len(text))
	for i := 0; i < len(text); i++ {
		rs = append(rs, gpt2Bytes[text[i]])
	}
	return greedyEncode(m.Vocab, string(rs), " ")
}

// Decode turns token ids back into text.
func (m *Flint) Decode(ids []int) string {
	var inv = map[rune]byte{}
	for b, r := range gpt2Bytes {
		inv[r] = byte(b)
	}
	var out []byte
	for _, id := range ids {
		for _, r := range m.Vocab[id] {
			if b, ok := inv[r]; ok {
				out = append(out, b)
			} else {
				out = append(out, string(r)...)
			}
		}
	}
	return string(out)
}

// Forward runs the whole token sequence and returns the logits of the last position.
func (m *Flint) Forward(tokens []int) []float32 { return m.Step(m.NewState(), tokens) }

// expertMatmul applies expert e of the stacked tensor name ([cols, rows, nExp]) to X.
func (m *weights) expertMatmul(name string, e int, X [][]float32) [][]float32 {
	t, b := m.raw(name)
	if t == nil || len(t.Dims) != 3 {
		panic("densecpu: missing or non-3D tensor " + name)
	}
	cols, rows := int(t.Dims[0]), int(t.Dims[1])
	elems, bytes, err := blockInfo(t.Type)
	if err != nil {
		panic(err)
	}
	per := rows * (cols / elems * bytes)
	return cpuMatmul(t.Type, b[e*per:(e+1)*per], cols, rows, X)
}

// moe is the shared experts plus the sigmoid-routed top-k experts, per token.
func (m *Flint) moe(p string, X [][]float32) [][]float32 {
	logits := m.matmul(p+"ffn_gate_inp.weight", X)
	byExp := make([][]int, m.nExp)
	wts := make([][]float32, len(X))
	sel := make([][]int, len(X))
	for i, lg := range logits {
		sc := make([]float32, len(lg))
		idx := make([]int, len(lg))
		for j, v := range lg {
			sc[j], idx[j] = sigmoid(v), j
		}
		sort.SliceStable(idx, func(a, b int) bool { return sc[idx[a]] > sc[idx[b]] })
		sel[i] = idx[:m.nUsed]
		var sum float32
		for _, e := range sel[i] {
			wts[i] = append(wts[i], sc[e])
			sum += sc[e]
			byExp[e] = append(byExp[e], i)
		}
		for k := range wts[i] {
			if m.normTopK && !m.abl.noNormTopK {
				wts[i][k] /= sum
			}
			wts[i][k] *= m.expScale
		}
	}
	// shared experts: a single stacked SwiGLU
	sg, su := m.matmul(p+"ffn_gate_shexp.weight", X), m.matmul(p+"ffn_up_shexp.weight", X)
	for i := range sg {
		for j := range sg[i] {
			sg[i][j] = silu(sg[i][j]) * su[i][j]
		}
	}
	out := m.matmul(p+"ffn_down_shexp.weight", sg)
	for e, toks := range byExp {
		if len(toks) == 0 {
			continue
		}
		xs := make([][]float32, len(toks))
		for k, i := range toks {
			xs[k] = X[i]
		}
		g, u := m.expertMatmul(p+"ffn_gate_exps.weight", e, xs), m.expertMatmul(p+"ffn_up_exps.weight", e, xs)
		for k := range g {
			for j := range g[k] {
				g[k][j] = silu(g[k][j]) * u[k][j]
			}
		}
		d := m.expertMatmul(p+"ffn_down_exps.weight", e, g)
		for k, i := range toks {
			var w float32
			for s, se := range sel[i] {
				if se == e {
					w = wts[i][s]
				}
			}
			for j := range out[i] {
				out[i][j] += w * d[k][j]
			}
		}
	}
	return out
}

// layer runs block l over the hidden rows h (the new tokens, starting at position pos0) in place.
func (m *Flint) layer(st *FlintState, l int, h [][]float32, pos0 int) {
	n := len(h)
	p := "blk." + strconv.Itoa(l) + "."
	qn, kn, sinks := m.vec(p+"attn_q_norm.weight"), m.vec(p+"attn_k_norm.weight"), m.vec(p+"attn_sinks.weight")
	hd, nHead := len(qn), len(sinks)
	kt, _ := m.raw(p + "attn_k.weight")
	nKV := int(kt.Dims[1]) / hd
	an := m.vec(p + "attn_norm.weight")
	xn := make([][]float32, n)
	for i := range h {
		xn[i] = rms(h[i], an, m.eps)
	}
	pr := m.matmulMulti([]string{p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"}, xn)
	Q, K, V := pr[0], pr[1], pr[2]
	for i := 0; i < n; i++ {
		for hh := 0; hh < nHead; hh++ {
			q := rms(Q[i][hh*hd:(hh+1)*hd], qn, m.eps)
			m.rope(q, l, pos0+i)
			copy(Q[i][hh*hd:], q)
		}
		for hh := 0; hh < nKV; hh++ {
			k := rms(K[i][hh*hd:(hh+1)*hd], kn, m.eps)
			m.rope(k, l, pos0+i)
			copy(K[i][hh*hd:], k)
		}
	}
	c := st.kv[l]
	c.K, c.V = append(c.K, packHalf(K)...), append(c.V, packHalf(V)...)
	ctx := make([][]float32, n)
	for i := range ctx {
		ctx[i] = make([]float32, nHead*hd)
	}
	scale := float32(1 / math.Sqrt(float64(hd)))
	parFor(nHead, func(hh int) {
		kv := hh / (nHead / nKV)
		sc := make([]float32, pos0+n)
		for i := 0; i < n; i++ {
			last, lo := pos0+i, 0
			if m.local[l] && last-m.window+1 > 0 {
				lo = last - m.window + 1
			}
			qh := Q[i][hh*hd : (hh+1)*hd]
			mx := sinks[hh]
			if m.abl.noSink {
				mx = float32(math.Inf(-1))
			}
			for j := lo; j <= last; j++ {
				sc[j] = dotHalf(qh, c.K[j][kv*hd:(kv+1)*hd]) * scale
				if sc[j] > mx {
					mx = sc[j]
				}
			}
			sum := float32(math.Exp(float64(sinks[hh] - mx))) // the sink takes softmax mass but has no value
			if m.abl.noSink {
				sum = 0
			}
			for j := lo; j <= last; j++ {
				sc[j] = float32(math.Exp(float64(sc[j] - mx)))
				sum += sc[j]
			}
			oh := ctx[i][hh*hd : (hh+1)*hd]
			for j := lo; j <= last; j++ {
				axpyHalf(oh, sc[j]/sum, c.V[j][kv*hd:(kv+1)*hd])
			}
		}
	})
	o := m.matmul(p+"attn_output.weight", ctx)
	for i := range o {
		for j := range o[i] {
			h[i][j] += o[i][j]
		}
	}
	fnw := m.vec(p + "ffn_norm.weight")
	fx := make([][]float32, n)
	for i := range h {
		fx[i] = rms(h[i], fnw, m.eps)
	}
	var d [][]float32
	if l < m.nDense {
		d = m.ffn(p, fx, false)
	} else {
		d = m.moe(p, fx)
	}
	for i := range d {
		for j := range d[i] {
			h[i][j] += d[i][j]
		}
	}
}

// Step feeds only the new tokens to the session and returns the logits of the last one.
func (m *Flint) Step(st *FlintState, tokens []int) []float32 {
	n, pos0 := len(tokens), st.pos
	emb, eb := m.raw("token_embd.weight")
	ec := int(emb.Dims[0])
	elems, bytes, _ := blockInfo(emb.Type)
	rowBytes := ec / elems * bytes
	h := make([][]float32, n)
	for i, tk := range tokens {
		h[i] = make([]float32, ec)
		if err := DequantRow(emb.Type, eb[tk*rowBytes:], ec, h[i]); err != nil {
			panic(err)
		}
	}
	for l := 0; l < m.nLayer; l++ {
		m.layer(st, l, h, pos0)
	}
	st.pos += n
	last := rms(h[n-1], m.vec("output_norm.weight"), m.eps)
	return m.matmul("output.weight", [][]float32{last})[0]
}
