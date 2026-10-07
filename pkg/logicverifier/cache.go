// pkg/logicverifier/cache.go - Multi-Tenant Verification Result Cache
package logicverifier

import (
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

type cacheEntry struct {
	key       string
	result    VerificationResult
	expiresAt time.Time
}

type VerificationCache struct {
	mu       sync.RWMutex
	capacity int
	ttl      time.Duration
	items    map[string]*list.Element
	lru      *list.List
}

func NewVerificationCache(capacity int, ttl time.Duration) *VerificationCache {
	if capacity <= 0 {
		capacity = 10000
	}
	if ttl <= 0 {
		ttl = 1 * time.Hour
	}
	return &VerificationCache{
		capacity: capacity,
		ttl:      ttl,
		items:    make(map[string]*list.Element),
		lru:      list.New(),
	}
}

// ComputeCanonicalHash generates a deterministic key for content-addressed lookup
func ComputeCanonicalHash(claim VerificationClaim) string {
	h := sha256.New()
	tenant := claim.TenantID
	if tenant == "" {
		tenant = "default"
	}

	h.Write([]byte(tenant + "|"))
	h.Write([]byte(string(claim.Type) + "|"))
	h.Write([]byte(strings.TrimSpace(claim.Expression) + "|"))
	h.Write([]byte(strings.TrimSpace(claim.ClaimedValue) + "|"))

	// Sorted premises
	sortedPremises := make([]string, len(claim.Premises))
	copy(sortedPremises, claim.Premises)
	sort.Strings(sortedPremises)
	for _, p := range sortedPremises {
		h.Write([]byte(strings.TrimSpace(p) + ";"))
	}
	h.Write([]byte("|"))

	// Sorted constraints
	sortedConstraints := make([]string, len(claim.Constraints))
	copy(sortedConstraints, claim.Constraints)
	sort.Strings(sortedConstraints)
	for _, c := range sortedConstraints {
		h.Write([]byte(strings.TrimSpace(c) + ";"))
	}
	h.Write([]byte("|"))

	h.Write([]byte(strings.TrimSpace(claim.SchemaJSON) + "|"))
	h.Write([]byte(strings.TrimSpace(claim.UnitExpression) + "|"))
	h.Write([]byte(strings.TrimSpace(claim.ClaimedUnit) + "|"))
	h.Write([]byte(fmt.Sprintf("%g:%g:%v", claim.Tolerance.AbsTol, claim.Tolerance.RelTol, claim.Tolerance.ExactOnly)))

	return hex.EncodeToString(h.Sum(nil))
}

func (c *VerificationCache) Get(key string) (VerificationResult, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	elem, found := c.items[key]
	if !found {
		return VerificationResult{}, false
	}

	entry := elem.Value.(*cacheEntry)
	if time.Now().After(entry.expiresAt) {
		c.lru.Remove(elem)
		delete(c.items, key)
		return VerificationResult{}, false
	}

	c.lru.MoveToFront(elem)
	res := entry.result
	res.CacheHit = true
	return res, true
}

func (c *VerificationCache) Put(key string, res VerificationResult) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, found := c.items[key]; found {
		c.lru.MoveToFront(elem)
		entry := elem.Value.(*cacheEntry)
		entry.result = res
		entry.expiresAt = time.Now().Add(c.ttl)
		return
	}

	for c.lru.Len() >= c.capacity {
		back := c.lru.Back()
		if back != nil {
			c.lru.Remove(back)
			oldEntry := back.Value.(*cacheEntry)
			delete(c.items, oldEntry.key)
		}
	}

	entry := &cacheEntry{
		key:       key,
		result:    res,
		expiresAt: time.Now().Add(c.ttl),
	}
	elem := c.lru.PushFront(entry)
	c.items[key] = elem
}

func (c *VerificationCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items = make(map[string]*list.Element)
	c.lru.Init()
}
