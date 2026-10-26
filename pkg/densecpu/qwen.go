package densecpu

import (
	"fmt"
	"math"
	"strconv"
)

// Qwen is a qwen35 dense GGUF (gated-delta-net blocks with a full-attention block every fourth layer).
// The trailing MTP (nextn) block is not run.
type Qwen struct {
	*weights
	nLayer, nRot             int
	eps                      float32
	base                     float32
	headK, numK, numV, convK int
	dInner                   int
	Vocab                    []string
}

// OpenQwen opens a qwen35 GGUF.
func OpenQwen(path string) (*Qwen, error) {
	w, err := openWeights(path, "qwen35")
	if err != nil {
		return nil, err
	}
	g := w.g
	q := &Qwen{weights: w,
		nLayer: metaI(g, "qwen35.block_count") - metaI(g, "qwen35.nextn_predict_layers"),
		nRot:   metaI(g, "qwen35.rope.dimension_count"),
		eps:    metaF(g, "qwen35.attention.layer_norm_rms_epsilon"),
		base:   metaF(g, "qwen35.rope.freq_base"),
		headK:  metaI(g, "qwen35.ssm.state_size"), numK: metaI(g, "qwen35.ssm.group_count"),
		numV: metaI(g, "qwen35.ssm.time_step_rank"), convK: metaI(g, "qwen35.ssm.conv_kernel"),
		dInner: metaI(g, "qwen35.ssm.inner_size")}
	if q.nLayer <= 0 || q.numV%q.numK != 0 || q.dInner%q.numV != 0 || q.headK != q.dInner/q.numV {
		q.Close()
		return nil, fmt.Errorf("densecpu: unusable qwen35 geometry (layers %d, k heads %d, v heads %d, inner %d)", q.nLayer, q.numK, q.numV, q.dInner)
	}
	q.Vocab = vocabOf(g)
	return q, nil
}

// GreedyEncode encodes text with the GPT-2 byte-level space marker (U+0120).
func (m *Qwen) GreedyEncode(text string) []int { return greedyEncode(m.Vocab, text, "Ġ") }

func sigmoid(x float32) float32 { return float32(1 / (1 + math.Exp(-float64(x)))) }
func silu(x float32) float32    { return x * sigmoid(x) }
func softplus(x float32) float32 {
	if x > 20 {
		return x
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}

// l2 normalises v in place to x / sqrt(sum x^2 + eps).
func l2(v []float32, eps float32) {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	k := float32(1 / math.Sqrt(s+float64(eps)))
	for i := range v {
		v[i] *= k
	}
}

// gdn runs one gated-delta-net block over the new tokens, continuing from st (nil: zero conv and recurrent state).
func (m *Qwen) gdn(st *gdnState, p string, xn [][]float32) [][]float32 {
	n := len(xn)
	hv, hk := m.dInner/m.numV, m.headK
	qkv := m.matmul(p+"attn_qkv.weight", xn)
	z := m.matmul(p+"attn_gate.weight", xn)
	beta := m.matmul(p+"ssm_beta.weight", xn)
	alpha := m.matmul(p+"ssm_alpha.weight", xn)
	dt, a := m.vec(p+"ssm_dt.bias"), m.vec(p+"ssm_a")
	cw := m.vec(p + "ssm_conv1d.weight") // [convK, channels]: channel c's taps are cw[c*convK : (c+1)*convK]
	ch := len(qkv[0])

	if st == nil {
		st = &gdnState{}
	}
	if st.S == nil {
		st.S = make([][]float32, m.numV) // per value head: S[i*hv+j], i over key dim, j over value dim
		for h := range st.S {
			st.S[h] = make([]float32, hk*hv)
		}
		for k := 0; k < m.convK-1; k++ {
			st.tail = append(st.tail, make([]float32, ch))
		}
	}
	state := st.S
	ext := append(append([][]float32(nil), st.tail...), qkv...) // ext[t+k] is the row k-(convK-1) steps before new token t
	st.tail = ext[n:]
	conv := make([][]float32, n)
	parFor(n, func(t int) {
		conv[t] = make([]float32, ch)
		for c := 0; c < ch; c++ {
			var s float32
			for k := 0; k < m.convK; k++ {
				s += cw[c*m.convK+k] * ext[t+k][c]
			}
			conv[t][c] = silu(s)
		}
	})
	out := make([][]float32, n)
	qs := float32(1 / math.Sqrt(float64(hk)))
	for t := 0; t < n; t++ {
		qa, ka, va := conv[t][:m.numK*hk], conv[t][m.numK*hk:2*m.numK*hk], conv[t][2*m.numK*hk:]
		for h := 0; h < m.numK; h++ {
			l2(qa[h*hk:(h+1)*hk], m.eps)
			l2(ka[h*hk:(h+1)*hk], m.eps)
		}
		o := make([]float32, m.dInner)
		parFor(m.numV, func(h int) {
			kh := h % m.numK // the key heads are tiled across the value heads
			q, k, v := qa[kh*hk:(kh+1)*hk], ka[kh*hk:(kh+1)*hk], va[h*hv:(h+1)*hv]
			decay := float32(math.Exp(float64(softplus(alpha[t][h]+dt[h]) * a[h])))
			b := sigmoid(beta[t][h])
			S := state[h]
			for i := range S {
				S[i] *= decay
			}
			d := make([]float32, hv)
			for j := 0; j < hv; j++ {
				var sk float32
				for i := 0; i < hk; i++ {
					sk += S[i*hv+j] * k[i]
				}
				d[j] = (v[j] - sk) * b
			}
			for i := 0; i < hk; i++ {
				for j := 0; j < hv; j++ {
					S[i*hv+j] += k[i] * d[j]
				}
			}
			for j := 0; j < hv; j++ {
				var s float32
				for i := 0; i < hk; i++ {
					s += S[i*hv+j] * q[i] * qs
				}
				o[h*hv+j] = s
			}
		})
		out[t] = o
	}
	nw := m.vec(p + "ssm_norm.weight")
	for t := 0; t < n; t++ {
		for h := 0; h < m.numV; h++ {
			seg := rms(out[t][h*hv:(h+1)*hv], nw, m.eps)
			for j := range seg {
				out[t][h*hv+j] = seg[j] * silu(z[t][h*hv+j])
			}
		}
	}
	return m.matmul(p+"ssm_out.weight", out)
}

// attn runs one gated full-attention block over all tokens.
func (m *Qwen) attn(p string, xn [][]float32, posOff int, c *kvCache) [][]float32 {
	n := len(xn)
	qn, kn := m.vec(p+"attn_q_norm.weight"), m.vec(p+"attn_k_norm.weight")
	hd := len(qn)
	qt, _ := m.raw(p + "attn_q.weight")
	kt, _ := m.raw(p + "attn_k.weight")
	nHead, nKV := int(qt.Dims[1])/(2*hd), int(kt.Dims[1])/hd // attn_q carries the query and its gate, interleaved per head
	qf, K, V := m.matmul(p+"attn_q.weight", xn), m.matmul(p+"attn_k.weight", xn), m.matmul(p+"attn_v.weight", xn)
	if c == nil {
		c = &kvCache{}
	}
	base := len(c.K) // position of the first new token
	Q := make([][]float32, n)
	gate := make([][]float32, n)
	for i := 0; i < n; i++ {
		Q[i], gate[i] = make([]float32, nHead*hd), make([]float32, nHead*hd)
		for h := 0; h < nHead; h++ {
			q := rms(qf[i][h*2*hd:h*2*hd+hd], qn, m.eps)
			ropeNeox(q[:m.nRot], base+i+posOff, m.nRot, m.base, nil)
			copy(Q[i][h*hd:], q)
			copy(gate[i][h*hd:], qf[i][h*2*hd+hd:(h+1)*2*hd])
		}
		for h := 0; h < nKV; h++ {
			k := rms(K[i][h*hd:(h+1)*hd], kn, m.eps)
			ropeNeox(k[:m.nRot], base+i+posOff, m.nRot, m.base, nil)
			copy(K[i][h*hd:], k)
		}
	}
	c.K, c.V = append(c.K, packHalf(K)...), append(c.V, packHalf(V)...)
	scale := float32(1 / math.Sqrt(float64(hd)))
	ctx := make([][]float32, n)
	for i := 0; i < n; i++ {
		ctx[i] = make([]float32, nHead*hd)
		last := base + i // attends to positions 0..last
		parFor(nHead, func(h int) {
			kv := h / (nHead / nKV)
			sc := make([]float32, last+1)
			mx := float32(math.Inf(-1))
			for j := 0; j <= last; j++ {
				var s float32
				for d := 0; d < hd; d++ {
					s += Q[i][h*hd+d] * halfTab[c.K[j][kv*hd+d]]
				}
				sc[j] = s * scale
				if sc[j] > mx {
					mx = sc[j]
				}
			}
			var sum float32
			for j := range sc {
				sc[j] = float32(math.Exp(float64(sc[j] - mx)))
				sum += sc[j]
			}
			for j := 0; j <= last; j++ {
				w := sc[j] / sum
				for d := 0; d < hd; d++ {
					ctx[i][h*hd+d] += w * halfTab[c.V[j][kv*hd+d]]
				}
			}
		})
		for j := range ctx[i] {
			ctx[i][j] *= sigmoid(gate[i][j])
		}
	}
	return m.matmul(p+"attn_output.weight", ctx)
}

func (m *Qwen) hidden(tokens []int) [][]float32 { return m.hiddenAt(m.NewState(), tokens) }

// hiddenAt runs the new tokens on top of the session st and returns their post-final-norm hidden states.
func (m *Qwen) hiddenAt(st *QwenState, tokens []int) [][]float32 {
	n := len(tokens)
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
		p := "blk." + strconv.Itoa(l) + "."
		an := m.vec(p + "attn_norm.weight")
		xn := make([][]float32, n)
		for i := range h {
			xn[i] = rms(h[i], an, m.eps)
		}
		var y [][]float32
		if t, _ := m.raw(p + "ssm_out.weight"); t != nil {
			if st.gdn[l] == nil {
				st.gdn[l] = &gdnState{}
			}
			y = m.gdn(st.gdn[l], p, xn)
		} else {
			if st.kv[l] == nil {
				st.kv[l] = &kvCache{}
			}
			y = m.attn(p, xn, 0, st.kv[l])
		}
		pn := m.vec(p + "post_attention_norm.weight")
		fx := make([][]float32, n)
		for i := range h {
			for j := range h[i] {
				h[i][j] += y[i][j]
			}
			fx[i] = rms(h[i], pn, m.eps)
		}
		g, u := m.matmul(p+"ffn_gate.weight", fx), m.matmul(p+"ffn_up.weight", fx)
		for i := range g {
			for j := range g[i] {
				g[i][j] = silu(g[i][j]) * u[i][j]
			}
		}
		d := m.matmul(p+"ffn_down.weight", g)
		for i := range h {
			for j := range h[i] {
				h[i][j] += d[i][j]
			}
		}
	}
	on := m.vec("output_norm.weight")
	for i := range h {
		h[i] = rms(h[i], on, m.eps)
	}
	st.pos += n
	return h
}

// Forward runs the token sequence and returns the logits of the last position.
func (m *Qwen) Forward(tokens []int) []float32 { return m.Step(m.NewState(), tokens) }

// Step feeds only the new tokens to the session and returns the logits of the last one.
func (m *Qwen) Step(st *QwenState, tokens []int) []float32 {
	hs := m.hiddenAt(st, tokens)
	return m.matmul("output.weight", [][]float32{hs[len(hs)-1]})[0]
}

// MTP runs the multi-token-prediction block (blk.<nLayer>) over a sequence whose post-final-norm hidden states hs
// came from hidden(tokens): position i reads hs[i] and the embedding of tokens[i+1] and returns the logits of
// token i+2, for i = 0..n-2. posOff shifts the rope position of the block's own attention (i + posOff).
func (m *Qwen) MTP(tokens []int, hs [][]float32, posOff int) [][]float32 {
	p := "blk." + strconv.Itoa(m.nLayer) + "."
	n := len(tokens) - 1
	emb, eb := m.raw("token_embd.weight")
	ec := int(emb.Dims[0])
	elems, bytes, _ := blockInfo(emb.Type)
	rowBytes := ec / elems * bytes
	enorm, hnorm := m.vec(p+"nextn.enorm.weight"), m.vec(p+"nextn.hnorm.weight")
	in := make([][]float32, n)
	for i := 0; i < n; i++ {
		e := make([]float32, ec)
		if err := DequantRow(emb.Type, eb[tokens[i+1]*rowBytes:], ec, e); err != nil {
			panic(err)
		}
		in[i] = append(rms(e, enorm, m.eps), rms(hs[i], hnorm, m.eps)...) // [embedding | hidden]
	}
	x := m.matmul(p+"nextn.eh_proj.weight", in)
	an, pn := m.vec(p+"attn_norm.weight"), m.vec(p+"post_attention_norm.weight")
	xn := make([][]float32, n)
	for i := range x {
		xn[i] = rms(x[i], an, m.eps)
	}
	y := m.attn(p, xn, posOff, nil)
	fx := make([][]float32, n)
	for i := range x {
		for j := range x[i] {
			x[i][j] += y[i][j]
		}
		fx[i] = rms(x[i], pn, m.eps)
	}
	g, u := m.matmul(p+"ffn_gate.weight", fx), m.matmul(p+"ffn_up.weight", fx)
	for i := range g {
		for j := range g[i] {
			g[i][j] = silu(g[i][j]) * u[i][j]
		}
	}
	d := m.matmul(p+"ffn_down.weight", g)
	hn := m.vec(p + "nextn.shared_head_norm.weight")
	for i := range x {
		for j := range x[i] {
			x[i][j] += d[i][j]
		}
		x[i] = rms(x[i], hn, m.eps)
	}
	return m.matmul("output.weight", x)
}
