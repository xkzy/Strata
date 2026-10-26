// pkg/modelschema/dense_gguf.go - layer plan of a dense (non-MoE) GGUF, read from its tensor directory.
//
// The engine's decode path is built for the Qwen3.8-Flash-Next MoE. This is the first step toward dense
// qwen35 / gemma4 models: it reads the per-layer structure from the file and refuses a file whose tensors do not
// match what a dense decode loop would need, so that loop can be written against a checked plan.
package modelschema

import (
	"fmt"
	"strconv"

	"strata/pkg/gguf"
)

// DenseLayerKind is how one transformer block mixes tokens.
type DenseLayerKind string

const (
	LayerGDN        DenseLayerKind = "gdn"            // qwen35 gated delta net (ssm_* tensors)
	LayerFullAttn   DenseLayerKind = "full_attn"      // GQA attention over the whole context
	LayerGlobalAttn DenseLayerKind = "global_attn_kv" // gemma4 global layer: no attn_v, K is reused as V
	LayerSWA        DenseLayerKind = "sliding_attn"   // gemma4 sliding-window layer
)

// DensePlan is the checked structure of a dense GGUF.
type DensePlan struct {
	Arch      string
	Layers    []DenseLayerKind // the decoded blocks, in order
	MTPLayers int              // trailing nextn (multi-token prediction) blocks, not in Layers
	NEmbd     int
	NFF       int
	Vocab     int
	// Per kind: head count, KV head count and head dim, from the tensor shapes.
	Heads map[DenseLayerKind][3]int
}

func metaInt(g *gguf.GGUFFile, key string) (int, bool) {
	switch v := g.Metadata[key].(type) {
	case uint32:
		return int(v), true
	case uint64:
		return int(v), true
	case int32:
		return int(v), true
	case int64:
		return int(v), true
	}
	return 0, false
}

// ReadDensePlan opens path and returns its layer plan, or an error naming the first missing or odd tensor.
func ReadDensePlan(path string) (*DensePlan, error) {
	g, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	defer g.Close()
	arch, _ := g.Metadata["general.architecture"].(string)
	if arch != "qwen35" && arch != "gemma4" {
		return nil, fmt.Errorf("dense plan: architecture %q is not qwen35 or gemma4", arch)
	}
	n, ok := metaInt(g, arch+".block_count")
	if !ok {
		return nil, fmt.Errorf("dense plan: no %s.block_count", arch)
	}
	p := &DensePlan{Arch: arch, Heads: map[DenseLayerKind][3]int{}}
	// qwen35 counts the nextn block in block_count; its tensors are blk.<n-1>.nextn.*.
	if nl, _ := metaInt(g, arch+".nextn_predict_layers"); nl > 0 {
		p.MTPLayers = nl
		n -= nl
	}
	if p.NEmbd, ok = metaInt(g, arch+".embedding_length"); !ok {
		return nil, fmt.Errorf("dense plan: no %s.embedding_length", arch)
	}
	p.NFF, _ = metaInt(g, arch+".feed_forward_length")
	te, ok := g.GetTensor("token_embd.weight")
	if !ok || len(te.Dims) != 2 || int(te.Dims[0]) != p.NEmbd {
		return nil, fmt.Errorf("dense plan: token_embd.weight missing or not [n_embd, vocab]")
	}
	p.Vocab = int(te.Dims[1])

	has := func(l int, name string) bool { _, ok := g.GetTensor("blk." + strconv.Itoa(l) + "." + name); return ok }
	dim := func(l int, name string, i int) int {
		t, _ := g.GetTensor("blk." + strconv.Itoa(l) + "." + name)
		if t == nil || i >= len(t.Dims) {
			return 0
		}
		return int(t.Dims[i])
	}
	common := []string{"attn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "post_attention_norm.weight"}
	if arch == "gemma4" {
		common = append(common, "ffn_norm.weight", "post_ffw_norm.weight", "layer_output_scale.weight", "attn_q_norm.weight", "attn_k_norm.weight")
	}
	for l := 0; l < n; l++ {
		for _, name := range common {
			if !has(l, name) {
				return nil, fmt.Errorf("dense plan: blk.%d.%s missing", l, name)
			}
		}
		if dim(l, "ffn_gate.weight", 0) != p.NEmbd || dim(l, "ffn_gate.weight", 1) != p.NFF {
			return nil, fmt.Errorf("dense plan: blk.%d.ffn_gate.weight is not [n_embd, n_ff]", l)
		}
		var kind DenseLayerKind
		switch {
		case arch == "qwen35" && has(l, "ssm_out.weight"):
			for _, name := range []string{"attn_qkv.weight", "attn_gate.weight", "ssm_a", "ssm_alpha.weight", "ssm_beta.weight", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_norm.weight"} {
				if !has(l, name) {
					return nil, fmt.Errorf("dense plan: gdn blk.%d.%s missing", l, name)
				}
			}
			kind = LayerGDN
		default:
			for _, name := range []string{"attn_q.weight", "attn_k.weight", "attn_output.weight"} {
				if !has(l, name) {
					return nil, fmt.Errorf("dense plan: blk.%d.%s missing", l, name)
				}
			}
			switch {
			case arch == "qwen35":
				kind = LayerFullAttn
			case !has(l, "attn_v.weight"):
				kind = LayerGlobalAttn
			default:
				kind = LayerSWA
			}
			if _, seen := p.Heads[kind]; !seen {
				hd := dim(l, "attn_k_norm.weight", 0)
				if hd > 0 {
					// attn_q is [n_embd, heads*hd] (x2 for qwen35's gated query); attn_k is [n_embd, kv_heads*hd].
					q := dim(l, "attn_q.weight", 1)
					if arch == "qwen35" {
						q /= 2
					}
					p.Heads[kind] = [3]int{q / hd, dim(l, "attn_k.weight", 1) / hd, hd}
				}
			}
		}
		p.Layers = append(p.Layers, kind)
	}
	for i := 0; i < p.MTPLayers; i++ {
		l := n + i
		for _, name := range []string{"nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight"} {
			if !has(l, name) {
				return nil, fmt.Errorf("dense plan: blk.%d.%s missing", l, name)
			}
		}
	}
	return p, nil
}
