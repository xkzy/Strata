// Package virtualcontext provides virtual context and KV page table management for long-context inference.
package virtualcontext

import (
	"math"
	"sort"
	"sync"
)

type KVResidency uint8

const (
	ResidencyGPUVRAM       KVResidency = 0
	ResidencyHostRAM       KVResidency = 1
	ResidencyCompressedRAM KVResidency = 2
	ResidencyNVMeDisk      KVResidency = 3
	ResidencyRecompute     KVResidency = 4
)

func (r KVResidency) String() string {
	switch r {
	case ResidencyGPUVRAM:
		return "GPU_VRAM"
	case ResidencyHostRAM:
		return "HOST_RAM"
	case ResidencyCompressedRAM:
		return "COMPRESSED_RAM"
	case ResidencyNVMeDisk:
		return "NVME_DISK"
	case ResidencyRecompute:
		return "RECOMPUTE"
	default:
		return "UNKNOWN"
	}
}

type KVFormat uint8

const (
	FormatFP16     KVFormat = 0
	FormatBF16     KVFormat = 1
	FormatFP8_E4M3 KVFormat = 2
	FormatINT8     KVFormat = 3
	FormatQ4_0     KVFormat = 4
	FormatHybrid   KVFormat = 5
)

type KVBlockKey struct {
	ModelHash        uint64
	TokenizerID      uint64
	TokenContentHash uint64
	StartTokenPos    uint32
	EndTokenPos      uint32
	LayerStart       uint16
	LayerEnd         uint16
	Format           KVFormat
	RoPEType         uint8
}

type KVPage struct {
	ID              uint64      `json:"id"`
	SessionID       uint64      `json:"session_id"`
	Key             KVBlockKey  `json:"key"`
	Residency       KVResidency `json:"residency"`
	ByteSize        uint64      `json:"byte_size"`
	TokenCount      uint32      `json:"token_count"`
	PrefillCostMs   float32     `json:"prefill_cost_ms"`
	LastAccessTick  uint64      `json:"last_access_tick"`
	AccessFrequency uint32      `json:"access_frequency"`
	Pinned          bool        `json:"pinned"`
	DependentPages  []uint64    `json:"dependent_pages,omitempty"`
	SourceSymbol    string      `json:"source_symbol,omitempty"`
}

func (p *KVPage) RetentionScore(currentTick uint64, recencyLambda float64) float64 {
	if p.Pinned {
		return math.Inf(1)
	}
	costWeight := float64(p.PrefillCostMs) * float64(p.AccessFrequency+1)
	sizeKB := math.Max(1.0, float64(p.ByteSize)/1024.0)
	score := costWeight / sizeKB

	var deltaTicks float64
	if currentTick >= p.LastAccessTick {
		deltaTicks = float64(currentTick - p.LastAccessTick)
	}
	recencyDecay := math.Exp(-recencyLambda * deltaTicks)
	score *= recencyDecay * (1.0 + 0.15*float64(len(p.DependentPages)))
	return score
}

type KVTierBudgets struct {
	VRAMBytes          uint64
	HostRAMBytes       uint64
	CompressedRAMBytes uint64
	NVMeBytes          uint64
}

type KVTableStats struct {
	TotalPages  uint64    `json:"total_pages"`
	TierPages   [5]uint64 `json:"tier_pages"`
	TierBytes   [5]uint64 `json:"tier_bytes"`
	Lookups     uint64    `json:"lookups"`
	Hits        [5]uint64 `json:"hits"`
	Promotions  uint64    `json:"promotions"`
	Demotions   uint64    `json:"demotions"`
	Evictions   uint64    `json:"evictions"`
	PageFaults  uint64    `json:"page_faults"`
}

type KVPageTable struct {
	budgets   KVTierBudgets
	mu        sync.RWMutex
	keyIndex  map[KVBlockKey]*KVPage
	pageIndex map[uint64]*KVPage
	stats     KVTableStats
}

func NewKVPageTable(budgets KVTierBudgets) *KVPageTable {
	return &KVPageTable{
		budgets:   budgets,
		keyIndex:  make(map[KVBlockKey]*KVPage),
		pageIndex: make(map[uint64]*KVPage),
	}
}

// Lookup returns the page for key, or nil. The result is a copy: UpdateResidency, EvictTier, Pin and RecordAccess
// change the stored page under the table's lock, so handing out the pointer would let a reader race them.
func (t *KVPageTable) Lookup(key KVBlockKey) *KVPage {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.stats.Lookups++
	p, ok := t.keyIndex[key]
	if !ok {
		t.stats.PageFaults++
		return nil
	}
	if int(p.Residency) < 5 {
		t.stats.Hits[p.Residency]++
	}
	return copyPage(p)
}

// GetPage returns a copy of the page with id, or nil (see Lookup).
func (t *KVPageTable) GetPage(id uint64) *KVPage {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return copyPage(t.pageIndex[id])
}

// copyPage copies a stored page (and its dependent list) for a caller outside the lock.
func copyPage(p *KVPage) *KVPage {
	if p == nil {
		return nil
	}
	c := *p
	if len(p.DependentPages) > 0 {
		c.DependentPages = append([]uint64(nil), p.DependentPages...)
	}
	return &c
}

// RegisterPage adds p to the table. The table keeps p itself, not a copy: do not change it after registering it;
// move it between tiers with UpdateResidency / EvictTier and read it back with Lookup or GetPage (which copy it).
func (t *KVPageTable) RegisterPage(p *KVPage) bool {
	if p == nil || p.ID == 0 {
		return false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if old, ok := t.pageIndex[p.ID]; ok {   // registered again: take the old entry out first, or the tier counts add up twice
		if t.keyIndex[old.Key] == old {
			delete(t.keyIndex, old.Key)
		}
		if tier := int(old.Residency); tier < 5 {
			if t.stats.TierPages[tier] > 0 {
				t.stats.TierPages[tier]--
			}
			if t.stats.TierBytes[tier] >= old.ByteSize {
				t.stats.TierBytes[tier] -= old.ByteSize
			} else {
				t.stats.TierBytes[tier] = 0
			}
		}
	}
	t.keyIndex[p.Key] = p
	t.pageIndex[p.ID] = p
	t.stats.TotalPages = uint64(len(t.pageIndex))

	tierIdx := int(p.Residency)
	if tierIdx < 5 {
		t.stats.TierPages[tierIdx]++
		t.stats.TierBytes[tierIdx] += p.ByteSize
	}
	return true
}

func (t *KVPageTable) UpdateResidency(id uint64, targetTier KVResidency) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	p, ok := t.pageIndex[id]
	if !ok {
		return false
	}
	if p.Residency == targetTier {
		return true
	}

	oldTier := int(p.Residency)
	newTier := int(targetTier)

	if oldTier < 5 {
		if t.stats.TierPages[oldTier] > 0 {
			t.stats.TierPages[oldTier]--
		}
		if t.stats.TierBytes[oldTier] >= p.ByteSize {
			t.stats.TierBytes[oldTier] -= p.ByteSize
		} else {
			t.stats.TierBytes[oldTier] = 0
		}
	}

	if newTier < oldTier {
		t.stats.Promotions++
	} else if newTier > oldTier {
		t.stats.Demotions++
	}

	p.Residency = targetTier

	if newTier < 5 {
		t.stats.TierPages[newTier]++
		t.stats.TierBytes[newTier] += p.ByteSize
	}
	return true
}

func (t *KVPageTable) Pin(id uint64, pinned bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if p, ok := t.pageIndex[id]; ok {
		p.Pinned = pinned
	}
}

func (t *KVPageTable) RecordAccess(id uint64, currentTick uint64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if p, ok := t.pageIndex[id]; ok {
		p.LastAccessTick = currentTick
		p.AccessFrequency++
	}
}

func (t *KVPageTable) EstimateWorkingSet(currentTick, windowTicks uint64) []uint64 {
	t.mu.RLock()
	defer t.mu.RUnlock()

	type scored struct {
		score float64
		id    uint64
	}
	var candidates []scored

	for id, p := range t.pageIndex {
		if p.Pinned {
			candidates = append(candidates, scored{score: math.Inf(1), id: id})
			continue
		}
		var delta uint64
		if currentTick >= p.LastAccessTick {
			delta = currentTick - p.LastAccessTick
		}
		if delta <= windowTicks {
			candidates = append(candidates, scored{score: p.RetentionScore(currentTick, 0.001), id: id})
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	res := make([]uint64, len(candidates))
	for i, c := range candidates {
		res[i] = c.id
	}
	return res
}

func (t *KVPageTable) EvictTier(tier KVResidency, bytesToFree, currentTick uint64) uint64 {
	t.mu.Lock()
	defer t.mu.Unlock()

	type candidate struct {
		score float64
		page  *KVPage
	}
	var candidates []candidate

	for _, p := range t.pageIndex {
		if p.Residency == tier && !p.Pinned {
			candidates = append(candidates, candidate{score: p.RetentionScore(currentTick, 0.001), page: p})
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score < candidates[j].score // lowest score evicted first
	})

	var freed uint64
	colderTier := tier + 1
	if colderTier > ResidencyRecompute {
		colderTier = ResidencyRecompute
	}

	for _, c := range candidates {
		if freed >= bytesToFree {
			break
		}
		p := c.page
		oldTier := int(p.Residency)
		newTier := int(colderTier)

		if oldTier < 5 {
			if t.stats.TierPages[oldTier] > 0 {
				t.stats.TierPages[oldTier]--
			}
			if t.stats.TierBytes[oldTier] >= p.ByteSize {
				t.stats.TierBytes[oldTier] -= p.ByteSize
			} else {
				t.stats.TierBytes[oldTier] = 0
			}
		}

		freed += p.ByteSize
		p.Residency = colderTier

		if newTier < 5 {
			t.stats.TierPages[newTier]++
			t.stats.TierBytes[newTier] += p.ByteSize
		}

		t.stats.Demotions++
		t.stats.Evictions++
	}
	return freed
}

func (t *KVPageTable) Stats() KVTableStats {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.stats
}
