package densecpu

import (
	"fmt"
	"math"
	"strconv"
)

// Gemma is a gemma4 dense GGUF held open (weights memory-mapped, never copied).
type Gemma struct {
	*weights
	nEmbd, nLayer       int
	eps, softcap        float32
	window              int
	baseGlobal, baseSWA float32
	swa                 []bool
	ropeFreqs           []float32
	Vocab               []string
}

// OpenGemma opens a gemma4 GGUF.
func OpenGemma(path string) (*Gemma, error) {
	w, err := openWeights(path, "gemma4")
	if err != nil {
		return nil, err
	}
	g := w.g
	m := &Gemma{weights: w,
		nEmbd: metaI(g, "gemma4.embedding_length"), nLayer: metaI(g, "gemma4.block_count"),
		eps: metaF(g, "gemma4.attention.layer_norm_rms_epsilon"), softcap: metaF(g, "gemma4.final_logit_softcapping"),
		window:     metaI(g, "gemma4.attention.sliding_window"),
		baseGlobal: metaF(g, "gemma4.rope.freq_base"), baseSWA: metaF(g, "gemma4.rope.freq_base_swa")}
	pat, _ := g.Metadata["gemma4.attention.sliding_window_pattern"].([]any)
	if len(pat) != m.nLayer {
		m.Close()
		return nil, fmt.Errorf("densecpu: sliding_window_pattern has %d entries for %d layers", len(pat), m.nLayer)
	}
	for _, p := range pat {
		b, _ := p.(bool)
		m.swa = append(m.swa, b)
	}
	m.ropeFreqs = m.vec("rope_freqs.weight")
	m.Vocab = vocabOf(g)
	return m, nil
}

// Forward runs the whole token sequence and returns the logits of the last position.
func (m *Gemma) Forward(tokens []int) []float32 { return m.Step(m.NewState(), tokens) }

// Step feeds only the new tokens to the session and returns the logits of the last one.
func (m *Gemma) Step(st *GemmaState, tokens []int) []float32 {
	n := len(tokens)
	pos0 := st.pos
	emb, eb := m.raw("token_embd.weight")
	ec := int(emb.Dims[0])
	elems, bytes, _ := blockInfo(emb.Type)
	rowBytes := ec / elems * bytes
	h := make([][]float32, n)
	scale := float32(math.Sqrt(float64(m.nEmbd)))
	for i, tk := range tokens {
		h[i] = make([]float32, ec)
		if err := DequantRow(emb.Type, eb[tk*rowBytes:], ec, h[i]); err != nil {
			panic(err)
		}
		for j := range h[i] {
			h[i][j] *= scale
		}
	}
	for l := 0; l < m.nLayer; l++ {
		p := "blk." + strconv.Itoa(l) + "."
		qn, kn := m.vec(p+"attn_q_norm.weight"), m.vec(p+"attn_k_norm.weight")
		hd := len(qn)
		qt, _ := m.raw(p + "attn_q.weight")
		kt, _ := m.raw(p + "attn_k.weight")
		nHead, nKV := int(qt.Dims[1])/hd, int(kt.Dims[1])/hd
		base, ff := m.baseGlobal, m.ropeFreqs
		if m.swa[l] {
			base, ff = m.baseSWA, nil
		}
		an := m.vec(p + "attn_norm.weight")
		xn := make([][]float32, n)
		for i := range h {
			xn[i] = rms(h[i], an, m.eps)
		}
		names := []string{p + "attn_q.weight", p + "attn_k.weight"}
		hasV := false
		if t, _ := m.raw(p + "attn_v.weight"); t != nil {
			hasV = true
			names = append(names, p+"attn_v.weight")
		}
		pr := m.matmulMulti(names, xn)
		Q, K := pr[0], pr[1]
		var V [][]float32
		if hasV {
			V = pr[2]
		} else {
			V = make([][]float32, n) // global layers: V is the raw K projection
			for i := range K {
				V[i] = append([]float32(nil), K[i]...)
			}
		}
		for i := 0; i < n; i++ {
			for hh := 0; hh < nHead; hh++ {
				q := rms(Q[i][hh*hd:(hh+1)*hd], qn, m.eps)
				ropeNeox(q, pos0+i, hd, base, ff)
				copy(Q[i][hh*hd:], q)
			}
			for hh := 0; hh < nKV; hh++ {
				k := rms(K[i][hh*hd:(hh+1)*hd], kn, m.eps)
				ropeNeox(k, pos0+i, hd, base, ff)
				copy(K[i][hh*hd:], k)
				copy(V[i][hh*hd:], rms(V[i][hh*hd:(hh+1)*hd], nil, m.eps))
			}
		}
		c, b0 := st.kv[l], st.base[l] // c.K[r] is position b0+r
		c.K, c.V = append(c.K, packHalf(K)...), append(c.V, packHalf(V)...)
		ctx := make([][]float32, n)
		for i := range ctx {
			ctx[i] = make([]float32, nHead*hd)
		}
		window := 0
		if m.swa[l] {
			window = m.window
		}
		done := false
		if m.gpu != nil {
			if r, ok := m.gpu.attn(p, Q, c, pos0-b0, window, nHead, nKV, hd, 1); ok {
				ctx, done = r, true
			}
		}
		parFor(nHead, func(hh int) {
			if done {
				return
			}
			kv := hh / (nHead / nKV)
			sc := make([]float32, pos0+n-b0)
			for i := 0; i < n; i++ {
				last := pos0 + i
				lo := b0
				if m.swa[l] && last-m.window+1 > lo {
					lo = last - m.window + 1
				}
				qh := Q[i][hh*hd : (hh+1)*hd]
				mx := float32(math.Inf(-1))
				for j := lo; j <= last; j++ {
					s := dotHalf(qh, c.K[j-b0][kv*hd:(kv+1)*hd]) // attention scale is 1.0
					sc[j-lo] = s
					if s > mx {
						mx = s
					}
				}
				var sum float32
				for j := lo; j <= last; j++ {
					sc[j-lo] = float32(math.Exp(float64(sc[j-lo] - mx)))
					sum += sc[j-lo]
				}
				oh := ctx[i][hh*hd : (hh+1)*hd]
				for j := lo; j <= last; j++ {
					axpyHalf(oh, sc[j-lo]/sum, c.V[j-b0][kv*hd:(kv+1)*hd])
				}
			}
		})
		if m.swa[l] && len(c.K) > m.window { // nothing older than the window is ever read again
			drop := len(c.K) - m.window
			c.K, c.V = append([][]uint16(nil), c.K[drop:]...), append([][]uint16(nil), c.V[drop:]...)
			st.base[l] += drop
		}
		o := m.matmul(p+"attn_output.weight", ctx)
		pan := m.vec(p + "post_attention_norm.weight")
		a := make([][]float32, n)
		for i := range o {
			o[i] = rms(o[i], pan, m.eps)
			a[i] = make([]float32, ec)
			for j := range a[i] {
				a[i][j] = h[i][j] + o[i][j]
			}
		}
		fnw := m.vec(p + "ffn_norm.weight")
		fx := make([][]float32, n)
		for i := range a {
			fx[i] = rms(a[i], fnw, m.eps)
		}
		d := m.ffn(p, fx, true)
		pfw, ls := m.vec(p+"post_ffw_norm.weight"), m.vec(p + "layer_output_scale.weight")[0]
		for i := range d {
			d[i] = rms(d[i], pfw, m.eps)
			for j := range d[i] {
				h[i][j] = (a[i][j] + d[i][j]) * ls
			}
		}
	}
	st.pos += n
	last := rms(h[n-1], m.vec("output_norm.weight"), m.eps)
	head := "output.weight"
	if t, _ := m.raw(head); t == nil {
		head = "token_embd.weight" // tied
	}
	logits := m.matmul(head, [][]float32{last})[0]
	if m.softcap > 0 {
		for i, v := range logits {
			logits[i] = m.softcap * float32(math.Tanh(float64(v/m.softcap)))
		}
	}
	return logits
}
