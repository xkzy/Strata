//go:build cuda

package densecpu

import (
	"testing"
	"time"
)

// TestDecodeBandwidth times repeated one-token matmuls on real Qwen weights and prints the weight bytes read per second.
func TestDecodeBandwidth(t *testing.T) {
	q, err := OpenQwen(modelPath(t, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	if _, err := q.EnableGPU(); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.60.ffn_gate.weight", "blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.0.attn_qkv.weight", "output.weight"} {
		tn, _ := q.raw(name)
		r := q.gpu.res[name]
		if tn == nil || r.ptr == nil {
			t.Logf("%s: not resident", name)
			continue
		}
		x := [][]float32{make([]float32, tn.Dims[0])}
		for i := range x[0] {
			x[0][i] = float32(i%7) - 3
		}
		q.gpu.matmul(name, tn, x)
		MatvecStats(true)
		const reps = 100
		for i := 0; i < reps; i++ {
			q.gpu.matmul(name, tn, x)
		}
		_, kern, _, _, _, _ := MatvecStats(true)
		bytes := float64(r.rowBytes) * float64(r.rows)
		t.Logf("%-26s type %2d dev %d %6.1f MB: %6.1f us per call (kernel+sync) = %5.0f GB/s", name, tn.Type, r.dev, bytes/1e6, kern/reps*1e6, bytes*reps/kern/1e9)
	}
}

// TestDecodeSteps prefills 2048 filler tokens, then times 30 one-token steps (profile with -cpuprofile).
func TestDecodeSteps(t *testing.T) {
	q, err := OpenQwen(modelPath(t, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	if _, err := q.EnableGPU(); err != nil {
		t.Fatal(err)
	}
	st := q.NewState()
	defer st.Free()
	for done := 0; done < 2048; done += 512 {
		ids := make([]int, 512)
		for i := range ids {
			ids[i] = 760 + (done+i)%50
		}
		q.Step(st, ids)
	}
	t0 := time.Now()
	for i := 0; i < 30; i++ {
		q.Step(st, []int{760 + i})
	}
	t.Logf("context 2048: %.1f ms per decode step", time.Since(t0).Seconds()/30*1000)
}
