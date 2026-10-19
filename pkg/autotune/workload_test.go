package autotune

import (
	"path/filepath"
	"testing"

	"strata/pkg/gguf"
)

func writeGGUF(t *testing.T) string {
	t.Helper()
	w := gguf.NewWriter()
	w.SetMetadata("general.architecture", "qwen3moe")
	w.SetMetadata("qwen3moe.block_count", uint32(24))
	w.SetMetadata("qwen3moe.embedding_length", uint32(2048))
	w.SetMetadata("qwen3moe.attention.head_count", uint32(16))
	w.SetMetadata("qwen3moe.attention.head_count_kv", uint32(8))
	w.SetMetadata("qwen3moe.attention.key_length", uint32(128))
	w.SetMetadata("qwen3moe.expert_count", uint32(64))
	w.SetMetadata("qwen3moe.expert_used_count", uint32(4))
	w.SetMetadata("qwen3moe.expert_feed_forward_length", uint32(1024))
	w.SetMetadata("qwen3moe.feed_forward_length", uint32(8192))
	w.SetMetadata("qwen3moe.context_length", uint32(32768))
	w.SetMetadata("qwen3moe.vocab_size", uint32(32000))
	w.SetMetadata("general.file_type", uint32(12)) // q4_k
	w.AddTensor("token_embd.weight", []uint64{2048, 32000}, gguf.GGMLTypeQ4_K, make([]byte, 1024))
	w.AddTensor("blk.0.ffn_gate_exps.weight", []uint64{2048, 1024, 64}, gguf.GGMLTypeQ4_K, make([]byte, 1024))
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := w.WriteFile(path); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestGeometryFromGGUF(t *testing.T) {
	g, err := GeometryFromGGUF(writeGGUF(t))
	if err != nil {
		t.Fatal(err)
	}
	if g.Layers != 24 || g.Hidden != 2048 || g.Heads != 16 || g.KVHeads != 8 || g.HeadDim != 128 {
		t.Errorf("shape: %+v", g)
	}
	if g.Experts != 64 || g.ExpertsPerToken != 4 || g.ExpertInterDim != 1024 || g.Vocab != 32000 {
		t.Errorf("MoE shape: %+v", g)
	}
	if g.Context != 32768 || g.Quant != "q4_k" || !g.NativePack || g.KVBytesPerElem != 2 {
		t.Errorf("metadata: %+v", g)
	}
	if g.WeightBytes == 0 {
		t.Error("weight bytes must come from the tensor catalogue")
	}
	// 24 layers * 8 KV heads * 128 dim * (K and V) * 2 bytes.
	if got, want := g.KVBytesPerToken(), uint64(24*8*128*2*2); got != want {
		t.Errorf("KV bytes/token = %d, want %d", got, want)
	}
	if g.ActiveParams() == 0 || g.TotalParams() <= g.ActiveParams() {
		t.Errorf("params: active=%d total=%d", g.ActiveParams(), g.TotalParams())
	}
}

func TestGeometryFromGGUFMissingFieldsStayZero(t *testing.T) {
	w := gguf.NewWriter()
	w.SetMetadata("general.architecture", "mystery")
	path := filepath.Join(t.TempDir(), "m.gguf")
	if err := w.WriteFile(path); err != nil {
		t.Fatal(err)
	}
	g, err := GeometryFromGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	if g.Layers != 0 || g.Hidden != 0 || g.KVBytesPerToken() != 0 {
		t.Errorf("unknown fields must stay 0, not be invented: %+v", g)
	}
	if _, err := GeometryFromGGUF(filepath.Join(t.TempDir(), "absent.gguf")); err == nil {
		t.Error("a missing file is an error")
	}
}

func TestGeometryKVFlagAndFingerprint(t *testing.T) {
	g := GeometryFromPack(48, 2048, 16, 8, 128, 64, 8, 1024)
	if g.NativePack {
		t.Error("the canonical pack is not native")
	}
	g = g.ApplyKVFlag([]string{"--kv", "int8"})
	if g.KVBytesPerElem != 1 {
		t.Errorf("--kv int8 must halve KV: %+v", g)
	}
	w1 := NewWorkload(g, DefaultEnvelope(32768), 32768)
	g2 := g.ApplyKVFlag([]string{"--kv", "fp16"})
	w2 := NewWorkload(g2, DefaultEnvelope(32768), 32768)
	if w1.Fingerprint() == w2.Fingerprint() {
		t.Error("a KV format change must change the workload fingerprint")
	}
	if w1.KVBytes != g.KVBytesPerToken()*32768 {
		t.Errorf("KV budget: %d", w1.KVBytes)
	}
}

func TestEnvelopeFromStats(t *testing.T) {
	prompts := []int{100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}
	outputs := []int{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
	e := EnvelopeFromStats(prompts, outputs, 32768)
	if e.PromptTokens != 900 || e.OutputTokens != 90 {
		t.Errorf("p90: %+v", e)
	}
	if e.Source != "observed" || e.Observed != 10 {
		t.Errorf("source: %+v", e)
	}
	// The physical context clamps the envelope, not the other way round.
	e = EnvelopeFromStats([]int{40000}, []int{2000}, 32768)
	if e.PromptTokens+e.OutputTokens > 32768 || e.OutputTokens < 0 {
		t.Errorf("context clamp: %+v", e)
	}
	if got := EnvelopeFromStats(nil, nil, 2048); got.Source != "default" || got.PromptTokens+got.OutputTokens > 2048 {
		t.Errorf("default: %+v", got)
	}
}

func TestObjectiveFollowsTheWorkload(t *testing.T) {
	w := testWorkload()
	w.Envelope.PromptTokens, w.Envelope.OutputTokens, w.Envelope.Concurrency = 100, 100, 1
	if got := ChooseObjective(w); got != ObjDecodeTokS {
		t.Errorf("ordinary chat: %s", got)
	}
	w.Envelope.PromptTokens, w.Envelope.OutputTokens = 4000, 64
	if got := ChooseObjective(w); got != ObjTTFT {
		t.Errorf("long prompt: %s", got)
	}
	w.Envelope.Concurrency = 4
	if got := ChooseObjective(w); got != ObjThroughput {
		t.Errorf("many users: %s", got)
	}
}
