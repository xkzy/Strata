package modelschema

import (
	"os"
	"path/filepath"
	"testing"
)

// The real files are 17 GB each and not in the repo: skip when they are not on this machine.
func densePath(t *testing.T, name string) string {
	p := filepath.Join(os.Getenv("HOME"), "Downloads/Strata-data/models", name)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("%s not present", name)
	}
	return p
}

func TestReadDensePlan_Qwen35(t *testing.T) {
	p, err := ReadDensePlan(densePath(t, "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Layers) != 64 || p.MTPLayers != 1 {
		t.Fatalf("layers=%d mtp=%d, want 64 and 1", len(p.Layers), p.MTPLayers)
	}
	for l, k := range p.Layers {
		want := LayerGDN
		if l%4 == 3 {
			want = LayerFullAttn
		}
		if k != want {
			t.Fatalf("layer %d is %s, want %s", l, k, want)
		}
	}
	if h := p.Heads[LayerFullAttn]; h != [3]int{24, 4, 256} {
		t.Fatalf("full attention heads/kv/dim = %v", h)
	}
}

func TestReadDensePlan_Gemma4(t *testing.T) {
	p, err := ReadDensePlan(densePath(t, "gemma-4-31B-it-UD-Q4_K_XL.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Layers) != 60 {
		t.Fatalf("layers=%d, want 60", len(p.Layers))
	}
	for l, k := range p.Layers {
		want := LayerSWA
		if l%6 == 5 {
			want = LayerGlobalAttn
		}
		if k != want {
			t.Fatalf("layer %d is %s, want %s", l, k, want)
		}
	}
	if h := p.Heads[LayerSWA]; h != [3]int{32, 16, 256} {
		t.Fatalf("sliding heads/kv/dim = %v", h)
	}
	if h := p.Heads[LayerGlobalAttn]; h != [3]int{32, 4, 512} {
		t.Fatalf("global heads/kv/dim = %v", h)
	}
}

func TestReadDensePlan_RejectsOtherArch(t *testing.T) {
	if _, err := ReadDensePlan(filepath.Join(t.TempDir(), "missing.gguf")); err == nil {
		t.Fatal("expected an error for a missing file")
	}
}
