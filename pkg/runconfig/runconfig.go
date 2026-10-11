// pkg/runconfig/runconfig.go - Server and Model Run Configuration in Go
package runconfig

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

type SamplingConfig struct {
	Temperature float64 `json:"temperature"`
	TopP        float64 `json:"top_p"`
	TopK        int     `json:"top_k"`
	MinP        float64 `json:"min_p"`
}

type ModelEntry struct {
	ID                   string          `json:"id,omitempty"`
	Name                 string          `json:"name,omitempty"`
	ModelPath            string          `json:"model_path"`
	EngineBinary         string          `json:"engine_binary,omitempty"`
	MaxContext           int             `json:"max_context,omitempty"`
	VirtualLimit         int             `json:"virtual_limit,omitempty"`
	Abliterate           bool            `json:"abliterate,omitempty"`
	AbliterationStrength float64         `json:"abliteration_strength,omitempty"`
	AbliterationVector   string          `json:"abliteration_vector,omitempty"`
	Sampling             *SamplingConfig `json:"sampling,omitempty"`
	Enabled              bool            `json:"enabled,omitempty"`
}

type StrataConfig struct {
	Host                  string                 `json:"host"`
	Port                  int                    `json:"port"`
	APIKey                string                 `json:"api_key"`
	Model                 string                 `json:"model"`
	ModelPath             string                 `json:"model_path"`
	DefaultModel          string                 `json:"default_model,omitempty"`
	Models                map[string]ModelEntry  `json:"models,omitempty"`
	EngineBinary          string                 `json:"engine_binary"`
	MaxContext            int                    `json:"max_context"`
	VirtualLimit          int                    `json:"virtual_limit"`
	FitMaxTokens          bool                   `json:"fit_max_tokens"`
	ReasoningBudgetTokens int                    `json:"reasoning_budget_tokens"`
	AnthropicThinking     string                 `json:"anthropic_thinking"` // "model" or "on_request"
	PowerPolicy           string                 `json:"power_policy"`       // "MAX_THROUGHPUT", "LOW_LATENCY", "BALANCED", "ENERGY_SAVING"
	ResourceAdapt         bool                   `json:"resource_adapt"`
	PoolWorkers           int                    `json:"pool_workers,omitempty"`
	PcieFrac              float64                `json:"pcie_frac,omitempty"`
	GPUs                  string                 `json:"gpus,omitempty"`
	Threads               int                    `json:"threads,omitempty"`
	Abliterate            bool                   `json:"abliterate,omitempty"`
	AbliterationStrength  float64                `json:"abliteration_strength,omitempty"`
	AbliterationVector    string                 `json:"abliteration_vector,omitempty"`
	Sampling              SamplingConfig         `json:"sampling"`
	MCPServers            map[string]interface{} `json:"mcp_servers,omitempty"`
	AllowedHosts          []string               `json:"allowed_hosts,omitempty"`
	CORSOrigins           []string               `json:"cors_origins,omitempty"`
}

// GetActiveModel returns the active model ID and model path with multi-model fallback
func (c StrataConfig) GetActiveModel() (string, string) {
	if c.DefaultModel != "" && c.Models != nil {
		if entry, ok := c.Models[c.DefaultModel]; ok {
			id := entry.ID
			if id == "" {
				id = c.DefaultModel
			}
			return id, entry.ModelPath
		}
	}
	if c.Model != "" || c.ModelPath != "" {
		m := c.Model
		if m == "" {
			m = "Qwen3.8-Flash-Next"
		}
		return m, c.ModelPath
	}
	if len(c.Models) > 0 {
		for key, entry := range c.Models {
			id := entry.ID
			if id == "" {
				id = key
			}
			return id, entry.ModelPath
		}
	}
	return "Qwen3.8-Flash-Next", ""
}

// GetModelEntry returns a ModelEntry by ID or falls back to legacy configuration
func (c StrataConfig) GetModelEntry(id string) (ModelEntry, bool) {
	if c.Models != nil {
		if entry, ok := c.Models[id]; ok {
			if entry.ID == "" {
				entry.ID = id
			}
			return entry, true
		}
	}
	if id == "" || id == c.Model || id == "default" {
		return ModelEntry{
			ID:                   c.Model,
			ModelPath:            c.ModelPath,
			EngineBinary:         c.EngineBinary,
			MaxContext:           c.MaxContext,
			VirtualLimit:         c.VirtualLimit,
			Abliterate:           c.Abliterate,
			AbliterationStrength: c.AbliterationStrength,
			AbliterationVector:   c.AbliterationVector,
			Sampling:             &c.Sampling,
			Enabled:              true,
		}, true
	}
	return ModelEntry{}, false
}

// AllModels returns all configured models as a map
func (c StrataConfig) AllModels() map[string]ModelEntry {
	res := make(map[string]ModelEntry)
	if c.Models != nil {
		for k, v := range c.Models {
			if v.ID == "" {
				v.ID = k
			}
			res[k] = v
		}
	}
	if len(res) == 0 && (c.Model != "" || c.ModelPath != "") {
		id := c.Model
		if id == "" {
			id = "Qwen3.8-Flash-Next"
		}
		res[id] = ModelEntry{
			ID:                   id,
			ModelPath:            c.ModelPath,
			EngineBinary:         c.EngineBinary,
			MaxContext:           c.MaxContext,
			VirtualLimit:         c.VirtualLimit,
			Abliterate:           c.Abliterate,
			AbliterationStrength: c.AbliterationStrength,
			AbliterationVector:   c.AbliterationVector,
			Sampling:             &c.Sampling,
			Enabled:              true,
		}
	}
	return res
}

func DefaultConfig() StrataConfig {
	return StrataConfig{
		Host:                  "127.0.0.1",
		Port:                  8080,
		Model:                 "Qwen3.8-Flash-Next",
		MaxContext:            32768,
		VirtualLimit:          2000000,
		FitMaxTokens:          true,
		ReasoningBudgetTokens: 0,
		AnthropicThinking:     "model",
		PowerPolicy:           "BALANCED",
		ResourceAdapt:         true,
		Sampling: SamplingConfig{
			Temperature: 0.0,
			TopP:        0.95,
			TopK:        40,
			MinP:        0.05,
		},
		MCPServers: make(map[string]interface{}),
	}
}

type ConfigManager struct {
	mu       sync.RWMutex
	filePath string
	config   StrataConfig
}

func NewConfigManager(filePath string) *ConfigManager {
	cm := &ConfigManager{
		filePath: filePath,
		config:   DefaultConfig(),
	}
	if filePath != "" {
		_ = cm.LoadFromFile(filePath)
	}
	return cm
}

func (cm *ConfigManager) LoadFromFile(path string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var cfg StrataConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return fmt.Errorf("invalid json config: %w", err)
	}

	cm.filePath = path
	cm.config = cfg
	return nil
}

func (cm *ConfigManager) SaveToFile(path string) error {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	data, err := json.MarshalIndent(cm.config, "", "  ")
	if err != nil {
		return err
	}

	if path == "" {
		path = cm.filePath
	}
	if path == "" {
		path = "strata-config.json"
	}

	return os.WriteFile(path, data, 0644)
}

func (cm *ConfigManager) Get() StrataConfig {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.config
}

func (cm *ConfigManager) Update(cfg StrataConfig) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.config = cfg
}
