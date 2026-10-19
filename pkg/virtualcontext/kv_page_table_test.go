package virtualcontext

import (
	"sync"
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

// Lookup and GetPage hand out copies: a reader must not race the table moving the page between tiers.
func TestLookupReturnsACopyUnderRace(t *testing.T) {
	tbl := NewKVPageTable(KVTierBudgets{})
	key := KVBlockKey{StartTokenPos: 1}
	tbl.RegisterPage(&KVPage{ID: 7, ByteSize: 100, Residency: ResidencyGPUVRAM, Key: key, DependentPages: []uint64{1, 2}})

	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 1000; j++ {
				if p := tbl.Lookup(key); p != nil {
					_ = p.Residency
					_ = len(p.DependentPages)
				}
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < 1000; j++ {
			tbl.UpdateResidency(7, ResidencyHostRAM)
			tbl.UpdateResidency(7, ResidencyGPUVRAM)
		}
	}()
	wg.Wait()

	p := tbl.GetPage(7)
	if p == nil || p.ID != 7 {
		t.Fatalf("GetPage = %+v", p)
	}
	p.Residency = ResidencyNVMeDisk // changing a copy must not change the stored page
	if got := tbl.GetPage(7).Residency; got == ResidencyNVMeDisk {
		t.Fatal("GetPage aliases the stored page")
	}
	p.DependentPages[0] = 99
	if got := tbl.Lookup(key); got.DependentPages[0] != 1 {
		t.Fatalf("GetPage shares DependentPages with the stored page: %v", got.DependentPages)
	}
}

func TestRegisterPageTwiceCountsOnce(t *testing.T) {
	tbl := NewKVPageTable(KVTierBudgets{})
	tbl.RegisterPage(&KVPage{ID: 7, ByteSize: 100, Residency: ResidencyGPUVRAM, Key: KVBlockKey{StartTokenPos: 1}})
	tbl.RegisterPage(&KVPage{ID: 7, ByteSize: 100, Residency: ResidencyHostRAM, Key: KVBlockKey{StartTokenPos: 2}})
	st := tbl.Stats()
	if st.TierPages[ResidencyGPUVRAM] != 0 || st.TierBytes[ResidencyGPUVRAM] != 0 {
		t.Fatalf("old tier still counted: %+v", st)
	}
	if st.TierPages[ResidencyHostRAM] != 1 || st.TierBytes[ResidencyHostRAM] != 100 || st.TotalPages != 1 {
		t.Fatalf("new tier counted wrong: %+v", st)
	}
	if tbl.Lookup(KVBlockKey{StartTokenPos: 1}) != nil {
		t.Fatal("old key still resolves")
	}
}
