package virtualcontext

import (
	"testing"
)

func TestKVBlockKeySensitivity(t *testing.T) {
	k1 := KVBlockKey{ModelHash: 100, TokenizerID: 1, TokenContentHash: 0x1234, StartTokenPos: 0, EndTokenPos: 4096, LayerStart: 0, LayerEnd: 32, Format: FormatFP16, RoPEType: 1}
	k2 := KVBlockKey{ModelHash: 100, TokenizerID: 1, TokenContentHash: 0x1234, StartTokenPos: 4096, EndTokenPos: 8192, LayerStart: 0, LayerEnd: 32, Format: FormatFP16, RoPEType: 1}
	k3 := KVBlockKey{ModelHash: 100, TokenizerID: 1, TokenContentHash: 0x5678, StartTokenPos: 0, EndTokenPos: 4096, LayerStart: 0, LayerEnd: 32, Format: FormatFP16, RoPEType: 1}

	if k1 == k2 {
		t.Fatal("expected distinct keys for different start/end positions")
	}
	if k1 == k3 {
		t.Fatal("expected distinct keys for different token hashes")
	}
}

func TestKVPageTableTierTransitions(t *testing.T) {
	table := NewKVPageTable(KVTierBudgets{HostRAMBytes: 16 << 30, NVMeBytes: 64 << 30})

	p := &KVPage{
		ID:            101,
		SessionID:     1,
		Key:           KVBlockKey{ModelHash: 100, TokenizerID: 1, TokenContentHash: 0xAAA, StartTokenPos: 0, EndTokenPos: 4096, LayerStart: 0, LayerEnd: 32, Format: FormatFP16, RoPEType: 1},
		Residency:     ResidencyRecompute,
		ByteSize:      8 * 1024 * 1024,
		TokenCount:    4096,
		PrefillCostMs: 150.0,
	}

	if !table.RegisterPage(p) {
		t.Fatal("failed to register page")
	}

	st := table.Stats()
	if st.TotalPages != 1 || st.TierPages[ResidencyRecompute] != 1 {
		t.Fatalf("unexpected stats: %+v", st)
	}

	// Promote to Host RAM
	if !table.UpdateResidency(101, ResidencyHostRAM) {
		t.Fatal("failed to update residency to Host RAM")
	}

	st = table.Stats()
	if st.Promotions != 1 || st.TierPages[ResidencyHostRAM] != 1 || st.TierPages[ResidencyRecompute] != 0 {
		t.Fatalf("unexpected stats after promotion: %+v", st)
	}

	// Promote to GPU VRAM
	if !table.UpdateResidency(101, ResidencyGPUVRAM) {
		t.Fatal("failed to update residency to GPU VRAM")
	}

	found := table.Lookup(p.Key)
	if found == nil || found.ID != 101 {
		t.Fatalf("lookup failed: %+v", found)
	}

	st = table.Stats()
	if st.Hits[ResidencyGPUVRAM] != 1 {
		t.Fatalf("expected 1 GPU VRAM hit, got %+v", st.Hits)
	}
}

func TestKVPageTableCostAwareEviction(t *testing.T) {
	table := NewKVPageTable(KVTierBudgets{HostRAMBytes: 16 << 30, NVMeBytes: 64 << 30})

	cheap := &KVPage{
		ID:              201,
		Key:             KVBlockKey{ModelHash: 100, TokenizerID: 1, TokenContentHash: 0x111, StartTokenPos: 0, EndTokenPos: 512, LayerStart: 0, LayerEnd: 32, Format: FormatFP16, RoPEType: 1},
		Residency:       ResidencyHostRAM,
		ByteSize:        2 * 1024 * 1024,
		PrefillCostMs:   5.0,
		AccessFrequency: 1,
		LastAccessTick:  100,
	}

	expensive := &KVPage{
		ID:              202,
		Key:             KVBlockKey{ModelHash: 100, TokenizerID: 1, TokenContentHash: 0x222, StartTokenPos: 512, EndTokenPos: 8192, LayerStart: 0, LayerEnd: 32, Format: FormatFP16, RoPEType: 1},
		Residency:       ResidencyHostRAM,
		ByteSize:        8 * 1024 * 1024,
		PrefillCostMs:   600.0,
		AccessFrequency: 10,
		LastAccessTick:  100,
	}

	table.RegisterPage(cheap)
	table.RegisterPage(expensive)

	freed := table.EvictTier(ResidencyHostRAM, 2*1024*1024, 100)
	if freed < 2*1024*1024 {
		t.Fatalf("expected >= 2MB freed, got %d", freed)
	}

	if cheap.Residency != ResidencyCompressedRAM {
		t.Fatalf("expected cheap page to be demoted to Compressed RAM, got %v", cheap.Residency)
	}
	if expensive.Residency != ResidencyHostRAM {
		t.Fatalf("expected expensive page to remain in Host RAM, got %v", expensive.Residency)
	}
}
