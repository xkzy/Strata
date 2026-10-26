//go:build cuda

package densecpu

import (
	"testing"
	"time"
)

func gpuReport(t *testing.T, st GPUStats) {
	t.Logf("on GPU: %d tensors, %.2f GiB (per device %v); on CPU: %d tensors, %.2f GiB",
		st.GPUTensors, float64(st.GPUBytes)/(1<<30), st.PerDevice, st.CPUTensors, float64(st.CPUBytes)/(1<<30))
}

func TestGemmaOnGPU(t *testing.T) {
	m, err := OpenGemma(modelPath(t, "gemma-4-31B-it-UD-Q4_K_XL.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	t0 := time.Now()
	st, err := m.EnableGPU()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("upload %.1fs", time.Since(t0).Seconds())
	gpuReport(t, st)
	ids := append([]int{2}, m.GreedyEncode("The capital of France is")...)
	logits := m.Forward(ids)
	if got := m.Vocab[argmax(logits)]; got != "▁Paris" {
		t.Fatalf("top token %q, want ▁Paris", got)
	}
	t.Logf("prompt logit of ▁Paris: %.3f (CPU reference 27.484)", logits[argmax(logits)])
	m.Forward([]int{2})
	t0 = time.Now()
	m.Forward([]int{2})
	t.Logf("gemma 1-token pass on GPU: %.3fs (CPU fused: 1.45s)", time.Since(t0).Seconds())
}

func TestQwenOnGPU(t *testing.T) {
	m, err := OpenQwen(modelPath(t, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	t0 := time.Now()
	st, err := m.EnableGPU()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("upload %.1fs", time.Since(t0).Seconds())
	gpuReport(t, st)
	logits := m.Forward(m.GreedyEncode("The capital of France is"))
	if got := m.Vocab[argmax(logits)]; got != "ĠParis" {
		t.Fatalf("top token %q, want ĠParis", got)
	}
	t.Logf("prompt logit of ĠParis: %.3f (CPU reference 16.356)", logits[argmax(logits)])
	m.Forward([]int{760})
	GPUCallNanos.Store(0)
	GPUCalls.Store(0)
	t0 = time.Now()
	m.Forward([]int{760})
	el := time.Since(t0)
	t.Logf("qwen 1-token pass on GPU: %.3fs (CPU fused: 1.01s); %d GPU calls took %.3fs, the rest (%.3fs) is Go code",
		el.Seconds(), GPUCalls.Load(), float64(GPUCallNanos.Load())/1e9, el.Seconds()-float64(GPUCallNanos.Load())/1e9)
}
