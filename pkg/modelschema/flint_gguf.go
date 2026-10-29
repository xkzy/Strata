// pkg/modelschema/flint_gguf.go - layer plan of a flint (sliding/full attention + shared/routed MoE) GGUF.
package modelschema

import (
	"fmt"
	"strconv"

	"strata/pkg/gguf"
)

// FlintPlan is the checked structure of a flint GGUF.
type FlintPlan struct {
	Layers             []DenseLayerKind // LayerSWA (rope, windowed) or LayerFullAttn (no rope), per block
	Window             int
	NDense             int // leading blocks with a dense FFN; the rest are MoE
	NExperts, NUsed    int
	NEmbd, Vocab       int
	Heads, KVHeads, HD int
}

// ReadFlintPlan opens path and returns its plan, or an error naming the first missing tensor or odd value.
func ReadFlintPlan(path string) (*FlintPlan, error) {
	g, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	defer g.Close()
	if a, _ := g.Metadata["general.architecture"].(string); a != "flint" {
		return nil, fmt.Errorf("flint plan: architecture %q is not flint", a)
	}
	get := func(k string) int { v, _ := metaInt(g, "flint."+k); return v }
	p := &FlintPlan{Window: get("attention.sliding_window"), NDense: get("leading_dense_block_count"),
		NExperts: get("expert_count"), NUsed: get("expert_used_count"), NEmbd: get("embedding_length")}
	n := get("block_count")
	pat, _ := g.Metadata["flint.attention.sliding_window_pattern"].([]any)
	rope, _ := g.Metadata["flint.attention.rope_pattern"].([]any)
	if n == 0 || len(pat) != n || len(rope) != n {
		return nil, fmt.Errorf("flint plan: block_count %d but patterns have %d and %d entries", n, len(pat), len(rope))
	}
	for l := 0; l < n; l++ {
		if pat[l] != rope[l] {
			return nil, fmt.Errorf("flint plan: layer %d: rope_pattern differs from sliding_window_pattern", l)
		}
		b := "blk." + strconv.Itoa(l) + "."
		need := []string{"attn_norm", "attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm", "attn_sinks", "ffn_norm"}
		if l < p.NDense {
			need = append(need, "ffn_gate", "ffn_up", "ffn_down")
		} else {
			need = append(need, "ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp")
		}
		for _, t := range need {
			if _, ok := g.GetTensor(b + t + ".weight"); !ok {
				return nil, fmt.Errorf("flint plan: missing %s%s.weight", b, t)
			}
		}
		if w, _ := pat[l].(bool); w {
			p.Layers = append(p.Layers, LayerSWA)
		} else {
			p.Layers = append(p.Layers, LayerFullAttn)
		}
	}
	te, ok := g.GetTensor("token_embd.weight")
	if !ok || len(te.Dims) != 2 {
		return nil, fmt.Errorf("flint plan: token_embd.weight missing or not 2-D")
	}
	p.Vocab = int(te.Dims[1])
	q, _ := g.GetTensor("blk.0.attn_q.weight")
	k, _ := g.GetTensor("blk.0.attn_k.weight")
	nq, _ := g.GetTensor("blk.0.attn_q_norm.weight")
	p.HD = int(nq.Dims[0])
	p.Heads, p.KVHeads = int(q.Dims[1])/p.HD, int(k.Dims[1])/p.HD
	return p, nil
}
