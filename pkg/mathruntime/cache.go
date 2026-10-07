package mathruntime

import (
	"fmt"
	"sync"
)

type MathResultCache struct {
	maxEntries int
	mu         sync.Mutex
	cache      map[string]MathResult
	tenantMap  map[string]map[string]struct{}
	lru        []string
}

func NewMathResultCache(maxEntries int) *MathResultCache {
	if maxEntries <= 0 {
		maxEntries = 10000
	}
	return &MathResultCache{
		maxEntries: maxEntries,
		cache:      make(map[string]MathResult),
		tenantMap:  make(map[string]map[string]struct{}),
		lru:        make([]string, 0, maxEntries),
	}
}

func (c *MathResultCache) MakeCacheKey(req MathRequest) string {
	parser := NewExpressionParser()
	canon := parser.Canonicalize(req.Expression)
	return fmt.Sprintf("%s|%s|%s|%s|%s|%d|%s|%d",
		req.Operation, req.Mode, canon, req.Variable, req.Point, req.Order, req.Assumptions, req.PrecisionDigits)
}

func (c *MathResultCache) Get(req MathRequest) (MathResult, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := c.MakeCacheKey(req)
	res, found := c.cache[key]
	if found {
		res.CacheHit = true
		return res, true
	}
	return MathResult{}, false
}

func (c *MathResultCache) Put(req MathRequest, res MathResult) {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := c.MakeCacheKey(req)

	if len(c.cache) >= c.maxEntries && len(c.lru) > 0 {
		oldest := c.lru[0]
		c.lru = c.lru[1:]
		delete(c.cache, oldest)
	}

	c.cache[key] = res
	c.lru = append(c.lru, key)

	if req.TenantID != "" {
		if c.tenantMap[req.TenantID] == nil {
			c.tenantMap[req.TenantID] = make(map[string]struct{})
		}
		c.tenantMap[req.TenantID][key] = struct{}{}
	}
}

func (c *MathResultCache) InvalidateTenant(tenantID string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	keys, ok := c.tenantMap[tenantID]
	if !ok {
		return
	}

	for k := range keys {
		delete(c.cache, k)
	}
	delete(c.tenantMap, tenantID)
}

func (c *MathResultCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cache = make(map[string]MathResult)
	c.tenantMap = make(map[string]map[string]struct{})
	c.lru = make([]string, 0, c.maxEntries)
}

func (c *MathResultCache) Size() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.cache)
}
