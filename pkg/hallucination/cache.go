// pkg/hallucination/cache.go - Dependency-Aware Verification Cache
package hallucination

import (
	"crypto/sha256"
	"fmt"
	"sync"
	"time"
)

// CacheEntry stores the deterministic verification result for a claim
type CacheEntry struct {
	ClaimID       string           `json:"claim_id"`
	Status        ClaimStatus      `json:"status"`
	EvidenceRefs  []string         `json:"evidence_refs"`
	Explanation   ClaimExplanation `json:"explanation"`
	SourceVersion string           `json:"source_version"`
	ExpiresAt     time.Time        `json:"expires_at"`
}

// VerificationCache provides fast lookup for deterministic verification results
type VerificationCache struct {
	mu            sync.RWMutex
	entries       map[string]CacheEntry
	sourceVersion string
	ttl           time.Duration
	maxEntries    int
}

// NewVerificationCache creates a cache with given TTL and maximum size
func NewVerificationCache(ttl time.Duration, maxEntries int) *VerificationCache {
	if ttl <= 0 {
		ttl = 30 * time.Minute
	}
	if maxEntries <= 0 {
		maxEntries = 10000
	}
	return &VerificationCache{
		entries:       make(map[string]CacheEntry),
		sourceVersion: "v1.0.0",
		ttl:           ttl,
		maxEntries:    maxEntries,
	}
}

// Key computes the deterministic cache key for a claim
func (vc *VerificationCache) Key(c *Claim) string {
	raw := fmt.Sprintf("%s:%s:%s:%s:%s", c.Type, c.Subject, c.Predicate, c.Object, c.SourceSpan)
	h := sha256.Sum256([]byte(raw))
	return fmt.Sprintf("%x", h[:16])
}

// Get retrieves a cached result if valid and not expired
func (vc *VerificationCache) Get(c *Claim) (CacheEntry, bool) {
	vc.mu.RLock()
	defer vc.mu.RUnlock()

	key := vc.Key(c)
	entry, found := vc.entries[key]
	if !found {
		return CacheEntry{}, false
	}

	if time.Now().After(entry.ExpiresAt) {
		return CacheEntry{}, false
	}

	if entry.SourceVersion != vc.sourceVersion {
		return CacheEntry{}, false
	}

	return entry, true
}

// Put stores a verification outcome in cache
func (vc *VerificationCache) Put(c *Claim, status ClaimStatus, evidenceRefs []string, exp ClaimExplanation) {
	vc.mu.Lock()
	defer vc.mu.Unlock()

	// Simple eviction if max reached
	if len(vc.entries) >= vc.maxEntries {
		// Evict oldest or clear half
		for k := range vc.entries {
			delete(vc.entries, k)
			if len(vc.entries) < vc.maxEntries*3/4 {
				break
			}
		}
	}

	key := vc.Key(c)
	vc.entries[key] = CacheEntry{
		ClaimID:       c.ID,
		Status:        status,
		EvidenceRefs:  evidenceRefs,
		Explanation:   exp,
		SourceVersion: vc.sourceVersion,
		ExpiresAt:     time.Now().Add(vc.ttl),
	}
}

// InvalidateOnSourceChange updates source version and invalidates old cache entries
func (vc *VerificationCache) InvalidateOnSourceChange(newVersion string) {
	vc.mu.Lock()
	defer vc.mu.Unlock()

	vc.sourceVersion = newVersion
	vc.entries = make(map[string]CacheEntry)
}

// Clear flushes all entries
func (vc *VerificationCache) Clear() {
	vc.mu.Lock()
	defer vc.mu.Unlock()

	vc.entries = make(map[string]CacheEntry)
}
