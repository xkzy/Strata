package modelschema

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadFlintPlan(t *testing.T) {
	path := filepath.Join(os.Getenv("HOME"), "Downloads/Strata-data/models/flint-UD-Q4_K_XL.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skip("model not present")
	}
	p, err := ReadFlintPlan(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Layers) != 49 || p.NDense != 2 || p.NExperts != 256 || p.NUsed != 6 || p.Heads != 32 || p.KVHeads != 4 || p.HD != 128 || p.Vocab != 150144 {
		t.Fatalf("unexpected plan %+v", p)
	}
	if p.Layers[0] != LayerFullAttn || p.Layers[2] != LayerSWA || p.Layers[48] != LayerFullAttn {
		t.Fatalf("layer kinds %v", p.Layers)
	}
}
