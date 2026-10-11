// pkg/engineipc/multi_orchestrator.go - Multi-Model Dynamic Engine Orchestrator
package engineipc

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

// MultiEngineManager coordinates multiple engine processes and routes requests
type MultiEngineManager struct {
	mu            sync.RWMutex
	engines       map[string]*EngineOrchestrator
	configs       map[string]EngineConfig
	tokenizers    map[string]TextCodec
	activeModel   string
	lazyLoad      bool
	idleTimeout   time.Duration
	lastActivity  map[string]time.Time
}

// NewMultiEngineManager creates a coordinator for multi-model serving
func NewMultiEngineManager(lazyLoad bool, idleTimeout time.Duration) *MultiEngineManager {
	return &MultiEngineManager{
		engines:      make(map[string]*EngineOrchestrator),
		configs:      make(map[string]EngineConfig),
		tokenizers:   make(map[string]TextCodec),
		lazyLoad:     lazyLoad,
		idleTimeout:  idleTimeout,
		lastActivity: make(map[string]time.Time),
	}
}

// RegisterModel associates a model ID with its launch configuration
func (m *MultiEngineManager) RegisterModel(modelID string, cfg EngineConfig, tok TextCodec) {
	m.mu.Lock()
	defer m.mu.Unlock()

	id := strings.ToLower(strings.TrimSpace(modelID))
	m.configs[id] = cfg
	if tok != nil {
		m.tokenizers[id] = tok
	} else {
		m.tokenizers[id] = NewByteTokenizer()
	}

	if m.activeModel == "" {
		m.activeModel = id
	}
}

// GetOrchestrator retrieves or initializes the engine orchestrator for the requested model
func (m *MultiEngineManager) GetOrchestrator(modelID string) (*EngineOrchestrator, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	id := strings.ToLower(strings.TrimSpace(modelID))
	if id == "" {
		id = m.activeModel
	}

	cfg, exists := m.configs[id]
	if !exists {
		// Fallback to active model if requested model is not explicitly configured
		if m.activeModel != "" {
			id = m.activeModel
			cfg = m.configs[id]
		} else {
			return nil, fmt.Errorf("model %q is not registered", modelID)
		}
	}

	m.lastActivity[id] = time.Now()

	// Return running orchestrator if already created
	if orch, ok := m.engines[id]; ok {
		return orch, nil
	}

	// Create and register new orchestrator
	orch := NewEngineOrchestrator(cfg)
	if tok, ok := m.tokenizers[id]; ok && tok != nil {
		orch.SetTokenizer(tok)
	}
	m.engines[id] = orch
	m.activeModel = id

	return orch, nil
}

// ActiveModel returns the currently active model ID
func (m *MultiEngineManager) ActiveModel() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.activeModel
}

// SetActiveModel switches the primary active model
func (m *MultiEngineManager) SetActiveModel(modelID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	id := strings.ToLower(strings.TrimSpace(modelID))
	if _, ok := m.configs[id]; !ok {
		return fmt.Errorf("model %q is not registered", modelID)
	}
	m.activeModel = id
	return nil
}

// ListModels returns all registered model IDs and their run states
func (m *MultiEngineManager) ListModels() map[string]string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	res := make(map[string]string)
	for id := range m.configs {
		if orch, ok := m.engines[id]; ok {
			res[id] = orch.State().String()
		} else {
			res[id] = StateStopped.String()
		}
	}
	return res
}

// StopAll stops all active engine instances
func (m *MultiEngineManager) StopAll() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, orch := range m.engines {
		if orch != nil {
			orch.Stop()
		}
	}
	m.engines = make(map[string]*EngineOrchestrator)
}

// PruneIdle stops engines that have been idle past idleTimeout
func (m *MultiEngineManager) PruneIdle() int {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.idleTimeout <= 0 {
		return 0
	}

	now := time.Now()
	pruned := 0

	for id, last := range m.lastActivity {
		// Do not prune the currently active primary model if it's the only one
		if id == m.activeModel && len(m.engines) <= 1 {
			continue
		}

		if now.Sub(last) > m.idleTimeout {
			if orch, ok := m.engines[id]; ok && orch != nil {
				orch.Stop()
				delete(m.engines, id)
				pruned++
			}
		}
	}

	return pruned
}
