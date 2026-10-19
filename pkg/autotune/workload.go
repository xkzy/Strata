// pkg/autotune/workload.go - the workload Strata actually executes, and the memory it needs.
//
// Geometry comes from the same GGUF metadata the engine reads; when a field is absent the value stays 0 and the
// solver refuses to claim a configuration that depends on it. The envelope (prompt/output lengths, concurrency)
// starts from a conservative default and is replaced by observed request statistics when they exist.
package autotune

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"strata/pkg/gguf"
)

// Geometry is the model shape the cost model needs. WeightBytes is the on-disk size of the weights.
type Geometry struct {
	Architecture    string `json:"architecture,omitempty"`
	Layers          int    `json:"layers"`
	Hidden          int    `json:"hidden"`
	Heads           int    `json:"heads"`
	KVHeads         int    `json:"kv_heads"`
	HeadDim         int    `json:"head_dim"`
	Experts         int    `json:"experts,omitempty"`
	ExpertsPerToken int    `json:"experts_per_token,omitempty"`
	ExpertInterDim  int    `json:"expert_inter_dim,omitempty"`
	DenseInterDim   int    `json:"dense_inter_dim,omitempty"`
	Vocab           int    `json:"vocab,omitempty"`
	Context         int    `json:"context,omitempty"`
	Quant           string `json:"quant,omitempty"`
	WeightBytes     uint64 `json:"weight_bytes,omitempty"`
	KVBytesPerElem  int    `json:"kv_bytes_per_elem,omitempty"` // engine default fp16 = 2; --kv int8 = 1
	NativePack      bool   `json:"native_pack,omitempty"`       // a GGUF-native pack (AVX2) rather than the canonical Q2_0 pack (AVX-512)
}

// KVBytesPerToken is K and V over every layer at one context position.
func (g Geometry) KVBytesPerToken() uint64 {
	kvHeads := g.KVHeads
	if kvHeads <= 0 {
		kvHeads = g.Heads
	}
	if kvHeads <= 0 || g.HeadDim <= 0 || g.Layers <= 0 {
		return 0
	}
	elems := g.KVBytesPerElem
	if elems <= 0 {
		elems = 2 // the engine's fp16 default
	}
	return uint64(g.Layers) * uint64(kvHeads) * uint64(g.HeadDim) * 2 * uint64(elems)
}

// ActiveParams is the number of weights read for one token: every dense weight plus the routed experts. It is
// the denominator of the decode roofline. 0 when the geometry is incomplete.
func (g Geometry) ActiveParams() uint64 {
	if g.Layers <= 0 || g.Hidden <= 0 {
		return 0
	}
	dense := uint64(g.DenseInterDim)
	if dense == 0 {
		dense = uint64(g.Hidden) * 4
	}
	perLayer := uint64(g.Hidden)*uint64(dense)*3 + uint64(g.Hidden)*uint64(g.Hidden)*4
	if g.Experts > 0 && g.ExpertsPerToken > 0 && g.ExpertInterDim > 0 {
		perLayer += uint64(g.ExpertsPerToken) * uint64(g.ExpertInterDim) * uint64(g.Hidden) * 3
	}
	params := perLayer * uint64(g.Layers)
	if g.Vocab > 0 {
		params += uint64(g.Vocab) * uint64(g.Hidden) * 2
	}
	return params
}

// TotalParams is every weight in the model, used to turn the on-disk weight bytes into bytes per parameter.
// 0 when the geometry is incomplete.
func (g Geometry) TotalParams() uint64 {
	if g.Layers <= 0 || g.Hidden <= 0 {
		return 0
	}
	dense := uint64(g.DenseInterDim)
	if dense == 0 {
		dense = uint64(g.Hidden) * 4
	}
	perLayer := uint64(g.Hidden)*uint64(g.Hidden)*4 + uint64(g.Hidden)*dense*3
	if g.Experts > 0 && g.ExpertInterDim > 0 {
		perLayer += uint64(g.Experts) * uint64(g.ExpertInterDim) * uint64(g.Hidden) * 3
	}
	total := perLayer * uint64(g.Layers)
	if g.Vocab > 0 {
		total += uint64(g.Vocab) * uint64(g.Hidden) * 2
	}
	return total
}

// Envelope is the workload the plan must serve. Percentiles are p90, not means: planning for the mean is how a
// configuration that "usually fits" falls over.
type Envelope struct {
	PromptTokens int     `json:"prompt_tokens"`
	OutputTokens int     `json:"output_tokens"`
	Concurrency  int     `json:"concurrency"`
	TTFTTargetMS float64 `json:"ttft_target_ms,omitempty"`
	Observed     int     `json:"observed_requests,omitempty"`
	Source       string  `json:"source"` // default | observed
}

// DefaultEnvelope is deliberately unambitious: a 2K prompt, a 256-token answer, one user. It is used only when
// no request statistics exist yet.
func DefaultEnvelope(maxContext int) Envelope {
	prompt, out := 2048, 256
	if maxContext > 0 && prompt+out > maxContext {
		prompt = maxContext / 2
		out = maxContext - prompt
		if out > 256 {
			out = 256
		}
	}
	return Envelope{PromptTokens: prompt, OutputTokens: out, Concurrency: 1, Source: "default"}
}

// EnvelopeFromStats takes observed prompt and output lengths and returns the p90 of each (nearest rank, so a
// small sample cannot interpolate a value it never saw), with at least one request's worth.
func EnvelopeFromStats(prompts, outputs []int, maxContext int) Envelope {
	if len(prompts) == 0 && len(outputs) == 0 {
		return DefaultEnvelope(maxContext)
	}
	e := Envelope{Source: "observed", Observed: len(prompts)}
	if len(prompts) > 0 {
		e.PromptTokens = nearestRank(prompts, 0.90)
	}
	if len(outputs) > 0 {
		e.OutputTokens = nearestRank(outputs, 0.90)
	}
	if e.Observed < len(outputs) {
		e.Observed = len(outputs)
	}
	e.Concurrency = 1 // the server's scheduler reports concurrent users separately; one is the safe floor
	if maxContext > 0 {
		if e.PromptTokens > maxContext {
			e.PromptTokens = maxContext
		}
		if e.PromptTokens+e.OutputTokens > maxContext {
			e.OutputTokens = maxContext - e.PromptTokens
			if e.OutputTokens < 0 {
				e.OutputTokens = 0
			}
		}
	}
	if e.PromptTokens <= 0 {
		e.PromptTokens = 1
	}
	return e
}

func nearestRank(v []int, p float64) int {
	if len(v) == 0 {
		return 0
	}
	s := append([]int(nil), v...)
	sort.Ints(s)
	rank := int(math.Ceil(p * float64(len(s))))
	if rank < 1 {
		rank = 1
	}
	if rank > len(s) {
		rank = len(s)
	}
	return s[rank-1]
}

// Workload is the geometry, the envelope and the KV budget for the physical window.
type Workload struct {
	Geometry   Geometry `json:"geometry"`
	Envelope   Envelope `json:"envelope"`
	MaxContext int      `json:"max_context"`
	KVBytes    uint64   `json:"kv_bytes"` // for MaxContext at the envelope's concurrency
}

// NewWorkload fills the derived fields.
func NewWorkload(g Geometry, e Envelope, maxContext int) Workload {
	if maxContext <= 0 {
		maxContext = g.Context
	}
	if maxContext <= 0 {
		maxContext = 32768
	}
	conc := e.Concurrency
	if conc < 1 {
		conc = 1
	}
	return Workload{
		Geometry:   g,
		Envelope:   e,
		MaxContext: maxContext,
		KVBytes:    g.KVBytesPerToken() * uint64(maxContext) * uint64(conc),
	}
}

// Fingerprint identifies the model+geometry a record belongs to. The tokenizer and template are deliberately
// absent: they do not change what a configuration costs.
func (w Workload) Fingerprint() string {
	g := w.Geometry
	s := fmt.Sprintf("schema=%d arch=%s layers=%d hidden=%d heads=%d kvh=%d hd=%d experts=%d topk=%d inter=%d vocab=%d ctx=%d weights=%d kv=%d",
		CapabilitySchema, g.Architecture, g.Layers, g.Hidden, g.Heads, g.KVHeads, g.HeadDim,
		g.Experts, g.ExpertsPerToken, g.ExpertInterDim, g.Vocab, w.MaxContext, g.WeightBytes, g.KVBytesPerToken())
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])[:24]
}

// GeometryFromGGUF reads the shape from a GGUF header. It is best effort: unreadable or absent keys leave fields
// at 0 rather than inventing a value, and the returned error is only for a file that cannot be opened/parsed.
func GeometryFromGGUF(path string) (Geometry, error) {
	f, err := gguf.Open(path)
	if err != nil {
		return Geometry{}, err
	}
	defer f.Close()
	g := Geometry{NativePack: true}
	g.Architecture, _ = f.Metadata["general.architecture"].(string)
	arch := g.Architecture

	u := func(keys ...string) int {
		for _, k := range keys {
			if v, ok := metaInt(f.Metadata, k); ok {
				return v
			}
		}
		return 0
	}
	g.Layers = u(arch+".block_count", "block_count", "llama.block_count")
	g.Hidden = u(arch+".embedding_length", "embedding_length", "n_embd")
	g.Heads = u(arch+".attention.head_count", "attention.head_count", "n_head")
	g.KVHeads = u(arch+".attention.head_count_kv", "attention.head_count_kv", "n_head_kv")
	g.HeadDim = u(arch+".attention.key_length", "attention.key_length", "n_embd_head_k")
	if g.HeadDim == 0 && g.Hidden > 0 && g.Heads > 0 {
		g.HeadDim = g.Hidden / g.Heads
	}
	g.Experts = u(arch+".expert_count", "expert_count", "n_expert")
	g.ExpertsPerToken = u(arch+".expert_used_count", "expert_used_count", "n_expert_used")
	g.ExpertInterDim = u(arch+".expert_feed_forward_length", "expert_ffn_length", "n_ff_exp")
	g.DenseInterDim = u(arch+".feed_forward_length", "feed_forward_length", "n_ff")
	g.Vocab = u(arch+".vocab_size", "vocab_size")
	if g.Vocab == 0 {
		if arr, ok := f.Metadata["tokenizer.ggml.tokens"]; ok {
			if s, ok := arr.([]any); ok {
				g.Vocab = len(s)
			}
		}
	}
	g.Context = u(arch+".context_length", "context_length", "n_ctx_train")
	if q, ok := f.Metadata["general.file_type"].(uint32); ok {
		g.Quant = quantName(q)
	}
	var total uint64
	for _, t := range f.Tensors {
		total += tensorBytes(t)
	}
	g.WeightBytes = total
	if g.KVHeads <= 0 {
		g.KVHeads = g.Heads
	}
	g.KVBytesPerElem = 2
	return g, nil
}

// GeometryFromPack is the native Q2_0 pack path: the pack index has no GGUF header, so the CPU-side defaults
// (48 layers, 64 experts, top-8, 128-wide heads) are used and the caller may override with measured/known
// values. WeightBytes must be supplied by the caller (SumPackBytes) or it stays 0.
func GeometryFromPack(layers, hidden, heads, kvHeads, headDim, experts, topK, inter int) Geometry {
	return Geometry{
		Architecture: "qwen-moe-pack", Layers: layers, Hidden: hidden, Heads: heads, KVHeads: kvHeads,
		HeadDim: headDim, Experts: experts, ExpertsPerToken: topK, ExpertInterDim: inter,
		KVBytesPerElem: 2, NativePack: false,
	}
}

// ApplyKVFlag records the engine's --kv choice (fp16 default, int8 halves it).
func (g Geometry) ApplyKVFlag(args []string) Geometry {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "--kv" {
			switch strings.ToLower(args[i+1]) {
			case "int8", "q8":
				g.KVBytesPerElem = 1
			case "fp16", "f16":
				g.KVBytesPerElem = 2
			}
		}
	}
	return g
}

func metaInt(m map[string]any, key string) (int, bool) {
	v, ok := m[key]
	if !ok {
		return 0, false
	}
	switch n := v.(type) {
	case uint8:
		return int(n), true
	case int8:
		return int(n), true
	case uint16:
		return int(n), true
	case int16:
		return int(n), true
	case uint32:
		return int(n), true
	case int32:
		return int(n), true
	case uint64:
		return int(n), true
	case int64:
		return int(n), true
	case int:
		return n, true
	case float32:
		return int(n), true
	case float64:
		return int(n), true
	}
	return 0, false
}

// tensorBytes is numel times the type's bits per element, rounded up. Values are the published GGML block sizes;
// an unknown type counts 16 bits, which is never below a real k-quant and only matters when the type is unknown.
func tensorBytes(t *gguf.TensorInfo) uint64 {
	var numel uint64 = 1
	for _, d := range t.Dims {
		numel *= d
	}
	bpe := map[uint32]float64{
		gguf.GGMLTypeF32: 4, gguf.GGMLTypeF16: 2, gguf.GGMLTypeBF16: 2, gguf.GGMLTypeF64: 8,
		gguf.GGMLTypeQ4_0: 0.5625, gguf.GGMLTypeQ4_1: 0.625, gguf.GGMLTypeQ5_0: 0.6875,
		gguf.GGMLTypeQ5_1: 0.75, gguf.GGMLTypeQ8_0: 1.0625, gguf.GGMLTypeQ8_1: 1.125,
		gguf.GGMLTypeQ2_K: 0.3438, gguf.GGMLTypeQ3_K: 0.4375, gguf.GGMLTypeQ4_K: 0.5625,
		gguf.GGMLTypeQ5_K: 0.6875, gguf.GGMLTypeQ6_K: 0.8203, gguf.GGMLTypeQ8_K: 1.0625,
		gguf.GGMLTypeIQ2_XXS: 0.2578, gguf.GGMLTypeIQ2_XS: 0.2891, gguf.GGMLTypeIQ3_XXS: 0.3828,
		gguf.GGMLTypeIQ1_S: 0.1953, gguf.GGMLTypeIQ4_NL: 0.5625, gguf.GGMLTypeIQ3_S: 0.4375,
		gguf.GGMLTypeIQ2_S: 0.3125, gguf.GGMLTypeIQ4_XS: 0.5313, gguf.GGMLTypeIQ1_M: 0.2188,
		gguf.GGMLTypeMXFP4: 0.5313, gguf.GGMLTypeI8: 1, gguf.GGMLTypeI16: 2, gguf.GGMLTypeI32: 4, gguf.GGMLTypeI64: 8,
	}
	bits, ok := bpe[t.Type]
	if !ok {
		bits = 2
	}
	return uint64(math.Ceil(float64(numel) * bits / 8))
}

func quantName(fileType uint32) string {
	switch fileType {
	case 0:
		return "f32"
	case 1:
		return "f16"
	case 2:
		return "q4_0"
	case 3:
		return "q4_1"
	case 7:
		return "q8_0"
	case 8:
		return "q5_0"
	case 9:
		return "q5_1"
	case 10:
		return "q2_k"
	case 11:
		return "q3_k"
	case 12:
		return "q4_k"
	case 13:
		return "q5_k"
	case 14:
		return "q6_k"
	case 15:
		return "q8_k"
	case 24:
		return "iq1_s"
	case 29:
		return "iq1_m"
	case 30:
		return "bf16"
	}
	return "q" + strconv.FormatUint(uint64(fileType), 10)
}
