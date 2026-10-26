//go:build cuda

package densecpu

import (
	"fmt"
	"strings"
	"testing"
)

// TestQwenPlacement prints which card each layer's FFN weights landed on.
func TestQwenPlacement(t *testing.T) {
	q, err := OpenQwen(modelPath(t, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	if _, err := q.EnableGPU(); err != nil {
		t.Fatal(err)
	}
	var sb strings.Builder
	split := 0
	for l := 0; l < q.nLayer; l++ {
		p := fmt.Sprintf("blk.%d.", l)
		g, u, d := q.gpu.res[p+"ffn_gate.weight"], q.gpu.res[p+"ffn_up.weight"], q.gpu.res[p+"ffn_down.weight"]
		fmt.Fprintf(&sb, "%d%d%d ", g.dev, u.dev, d.dev)
		if g.dev != u.dev || g.dev != d.dev || g.ptr == nil {
			split++
		}
	}
	t.Logf("ffn gate/up/down device per layer: %s", sb.String())
	t.Logf("%d of %d layers have their FFN weights on more than one card", split, q.nLayer)
}
