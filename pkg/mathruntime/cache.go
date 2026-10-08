package mathruntime

import (
	"fmt"
	"sync"
)

// MathResultCache keeps computed results (bounded, oldest out) and each session's variables. Callers choose the session
// ids and the tenant, so everything kept here has a limit (limits.go): without them one caller could grow the memory of
// the whole server by rotating session ids or sending ever-new expressions.
type MathResultCache struct {
	maxEntries   int
	mu           sync.Mutex
	cache        map[string]MathResult
	tenantMap    map[string]map[string]struct{} // tenant -> the cache keys it stored
	keyTenants   map[string]map[string]struct{} // cache key -> the tenants that stored it (to drop an evicted key)
	sessionVars  map[string]map[string]string
	sessionOrder []string // session ids, oldest first
	lru          []string // cache keys, oldest first
}

func NewMathResultCache(maxEntries int) *MathResultCache {
	if maxEntries <= 0 {
		maxEntries = 10000
	}
	return &MathResultCache{
		maxEntries:  maxEntries,
		cache:       make(map[string]MathResult),
		tenantMap:   make(map[string]map[string]struct{}),
		keyTenants:  make(map[string]map[string]struct{}),
		sessionVars: make(map[string]map[string]string),
		lru:         make([]string, 0, maxEntries),
	}
}

// SetSessionVar keeps a variable of a session. Too many sessions drop the oldest; a session holds at most
// maxSessionVars variables (an existing one can still be changed) and each value at most maxSessionValueLen characters:
// a larger one is not kept.
func (c *MathResultCache) SetSessionVar(sessionID, name, value string) {
	if len(value) > maxSessionValueLen || len(name) > 128 || len(sessionID) > 256 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	vars := c.sessionVars[sessionID]
	if vars == nil {
		if len(c.sessionVars) >= maxSessions && len(c.sessionOrder) > 0 {
			delete(c.sessionVars, c.sessionOrder[0])
			c.sessionOrder = c.sessionOrder[1:]
		}
		vars = make(map[string]string)
		c.sessionVars[sessionID] = vars
		c.sessionOrder = append(c.sessionOrder, sessionID)
	}
	if _, exists := vars[name]; !exists && len(vars) >= maxSessionVars {
		return
	}
	vars[name] = value
}

func (c *MathResultCache) GetSessionVar(sessionID, name string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.sessionVars[sessionID] == nil {
		return "", false
	}
	val, ok := c.sessionVars[sessionID][name]
	return val, ok
}

func (c *MathResultCache) GetSessionVars(sessionID string) map[string]string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.sessionVars[sessionID] == nil {
		return nil
	}
	res := make(map[string]string)
	for k, v := range c.sessionVars[sessionID] {
		res[k] = v
	}
	return res
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

// forget drops a key from every index (the caller holds the lock and has dealt with c.cache and c.lru).
func (c *MathResultCache) forget(key string) {
	for t := range c.keyTenants[key] {
		delete(c.tenantMap[t], key)
		if len(c.tenantMap[t]) == 0 {
			delete(c.tenantMap, t)
		}
	}
	delete(c.keyTenants, key)
}

func (c *MathResultCache) Put(req MathRequest, res MathResult) {
	if len(res.ExactResult)+len(res.NumericResult)+len(res.RawResult) > maxCachedResultLen {
		return // not worth keeping, and 10,000 of them would be gigabytes
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	key := c.MakeCacheKey(req)
	if _, exists := c.cache[key]; !exists {
		if len(c.cache) >= c.maxEntries && len(c.lru) > 0 {
			oldest := c.lru[0]
			c.lru = c.lru[1:]
			delete(c.cache, oldest)
			c.forget(oldest)
		}
		c.lru = append(c.lru, key)
	}
	c.cache[key] = res
	if req.TenantID != "" {
		if c.tenantMap[req.TenantID] == nil {
			c.tenantMap[req.TenantID] = make(map[string]struct{})
		}
		c.tenantMap[req.TenantID][key] = struct{}{}
		if c.keyTenants[key] == nil {
			c.keyTenants[key] = make(map[string]struct{})
		}
		c.keyTenants[key][req.TenantID] = struct{}{}
	}
}

func (c *MathResultCache) InvalidateTenant(tenantID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	keys, ok := c.tenantMap[tenantID]
	if !ok {
		return
	}
	gone := make(map[string]struct{}, len(keys))
	for k := range keys {
		delete(c.cache, k)
		gone[k] = struct{}{}
	}
	for k := range gone {
		c.forget(k)
	}
	kept := c.lru[:0] // the eviction list must not keep the dropped keys
	for _, k := range c.lru {
		if _, dropped := gone[k]; !dropped {
			kept = append(kept, k)
		}
	}
	c.lru = kept
	delete(c.tenantMap, tenantID)
}

func (c *MathResultCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache = make(map[string]MathResult)
	c.tenantMap = make(map[string]map[string]struct{})
	c.keyTenants = make(map[string]map[string]struct{})
	c.lru = make([]string, 0, c.maxEntries)
}

func (c *MathResultCache) Size() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.cache)
}
