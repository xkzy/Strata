// pkg/modelschema/registry.go - Thread-safe Model Registry for Multi-Model Strata Engine
package modelschema

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

// ModelRuntimeProfile defines a configured runtime instance of a model
type ModelRuntimeProfile struct {
	ID                   string                 `json:"id"`
	ManifestID           string                 `json:"manifest_id"`
	ModelPath            string                 `json:"model_path"`
	EngineBinary         string                 `json:"engine_binary,omitempty"`
	MaxContext           int                    `json:"max_context"`
	VirtualLimit         int                    `json:"virtual_limit"`
	Abliterate           bool                   `json:"abliterate,omitempty"`
	AbliterationStrength float64                `json:"abliteration_strength,omitempty"`
	AbliterationVector   string                 `json:"abliteration_vector,omitempty"`
	ExtraArgs            map[string]interface{} `json:"extra_args,omitempty"`
	Enabled              bool                   `json:"enabled"`
	Status               string                 `json:"status"` // "ready", "loaded", "unloaded", "loading", "error"
}

// ModelRegistry manages available model manifests and runtime instances
type ModelRegistry struct {
	mu        sync.RWMutex
	manifests map[string]*ModelManifest
	aliasMap  map[string]string
	runtimes  map[string]*ModelRuntimeProfile
}

// NewRegistry creates a registry pre-loaded with Strata's native architectural models
func NewRegistry() *ModelRegistry {
	r := &ModelRegistry{
		manifests: make(map[string]*ModelManifest),
		aliasMap:  make(map[string]string),
		runtimes:  make(map[string]*ModelRuntimeProfile),
	}
	r.registerBuiltins()
	return r
}

func (r *ModelRegistry) registerBuiltins() {
	// 1. Qwen3.8-Flash-Next (Default Hybrid QSA MoE)
	_ = r.RegisterManifest(ModelManifest{
		ID:           "qwen3.8-flash-next",
		Name:         "Qwen3.8-Flash-Next",
		Family:       FamilyQwen,
		Architecture: ArchHybridQSA,
		Attention:    AttentionFullMHA,
		MoE: &MoETopology{
			NumExperts:  64,
			TopK:        8,
			RoutingType: "softmax",
		},
		Resources: ResourceSpec{
			MinRAMBytes:          12 << 30, // 12 GiB
			MinVRAMBytes:         4 << 30,  // 4 GiB
			LayersTotal:          40,
			HiddenDim:            3584,
			NumHeads:             28,
			NumKVHeads:           4,
			MaxContextTokens:     32768,
			DefaultContextTokens: 32768,
			SupportsAbliteration: true,
		},
		Aliases:         []string{"qwen", "qwen3.8", "flash-next", "strata"},
		Description:     "Strata optimized Qwen3.8-Flash-Next MoE with hybrid QSA/GDN attention",
		HuggingFaceRepo: "Qwen/Qwen3.8-Flash-Next-GGUF",
	})

	// 2. Qwen3.8-Flash-Next-Coder
	_ = r.RegisterManifest(ModelManifest{
		ID:           "qwen3.8-flash-next-coder",
		Name:         "Qwen3.8-Flash-Next-Coder",
		Family:       FamilyQwen,
		Architecture: ArchHybridQSA,
		Attention:    AttentionFullMHA,
		MoE: &MoETopology{
			NumExperts:  64,
			TopK:        8,
			RoutingType: "softmax",
		},
		Resources: ResourceSpec{
			MinRAMBytes:          12 << 30,
			MinVRAMBytes:         4 << 30,
			LayersTotal:          40,
			HiddenDim:            3584,
			NumHeads:             28,
			NumKVHeads:           4,
			MaxContextTokens:     32768,
			DefaultContextTokens: 32768,
			SupportsAbliteration: true,
		},
		Aliases:         []string{"coder", "qwen-coder", "qwen3.8-flash-next-coder-iq1_m"},
		Description:     "Strata optimized Qwen3.8-Flash-Next-Coder with programming specialization",
		HuggingFaceRepo: "Qwen/Qwen3.8-Flash-Next-Coder-GGUF",
	})

	// 3. Xiaomi MiMo-V2.6-Flash-RL (Asymmetric SWA MoE)
	_ = r.RegisterManifest(ModelManifest{
		ID:           "mimo-v2.6-flash-rl",
		Name:         "Xiaomi MiMo-V2.6-Flash-RL",
		Family:       FamilyMiMo,
		Architecture: ArchAsymmetricMoE,
		Attention:    AttentionAsymmetricSWA,
		MoE: &MoETopology{
			NumExperts:  256,
			TopK:        8,
			RoutingType: "sigmoid",
		},
		Resources: ResourceSpec{
			MinRAMBytes:          16 << 30, // 16 GiB
			MinVRAMBytes:         6 << 30,  // 6 GiB
			LayersTotal:          48,
			HiddenDim:            4096,
			NumHeads:             32,
			NumKVHeads:           8,
			HeadDimQK:            192,
			HeadDimV:             128,
			MaxContextTokens:     131072,
			DefaultContextTokens: 32768,
			SupportsMTP:          true,
			SupportsAbliteration: true,
		},
		Aliases:         []string{"mimo", "mimo-v2.6", "mimo-flash", "mimo-rl", "mimo-v2.6-flash"},
		Description:     "Xiaomi MiMo-V2.6-Flash-RL with asymmetric QK/V dimensions and 128-token rolling SWA buffer",
		HuggingFaceRepo: "XiaomiMiMo/MiMo-V2.6-Flash-RL",
		DefaultGGUF:     "MiMo-V2.6-Flash-RL-Q2_K.gguf",
	})

	// 4. Mamba State-Space Model
	_ = r.RegisterManifest(ModelManifest{
		ID:           "mamba-2.8b",
		Name:         "Mamba-2.8B Selective SSM",
		Family:       FamilyMamba,
		Architecture: ArchStateSpaceModel,
		Attention:    AttentionSelectiveSSM,
		SSM: &SSMTopology{
			StateDim:       16,
			InnerDim:       5120,
			ConvKernel:     4,
			TimeStepRank:   160,
			Discretization: "zoh",
		},
		Resources: ResourceSpec{
			MinRAMBytes:          6 << 30, // 6 GiB
			MinVRAMBytes:         2 << 30, // 2 GiB
			LayersTotal:          64,
			HiddenDim:            2560,
			NumHeads:             1,
			NumKVHeads:           1,
			MaxContextTokens:     1048576, // 1M linear context with O(1) memory
			DefaultContextTokens: 65536,
			SupportsAbliteration: false,
		},
		Aliases:         []string{"mamba", "mamba-2", "mamba-ssm", "mamba-2.8b-slim"},
		Description:     "Hardware-aware selective state-space model with strictly O(1) constant recurrence memory",
		HuggingFaceRepo: "state-spaces/mamba-2.8b-slimpj",
	})

	// 5. DeepSeek-V3 / DeepSeek-R1 (MLA MoE)
	_ = r.RegisterManifest(ModelManifest{
		ID:           "deepseek-v3",
		Name:         "DeepSeek-V3 / R1",
		Family:       FamilyDeepSeek,
		Architecture: ArchMultiHeadLatentAttention,
		Attention:    AttentionMLA,
		MoE: &MoETopology{
			NumExperts:    256,
			TopK:          8,
			SharedExperts: 1,
			RoutingType:   "sigmoid",
		},
		Resources: ResourceSpec{
			MinRAMBytes:          32 << 30,
			MinVRAMBytes:         12 << 30,
			LayersTotal:          61,
			HiddenDim:            7168,
			NumHeads:             128,
			NumKVHeads:           128,
			MaxContextTokens:     131072,
			DefaultContextTokens: 32768,
			SupportsMTP:          true,
			SupportsAbliteration: true,
		},
		Aliases:         []string{"deepseek", "deepseek-r1", "deepseek-coder"},
		Description:     "Multi-head Latent Attention (MLA) architecture with DeepSeekMoE routing",
		HuggingFaceRepo: "deepseek-ai/DeepSeek-V3-GGUF",
	})

	// 6. OpenAI GPT-OSS-20B (Open-Weight 21B MoE Reasoning)
	_ = r.RegisterManifest(ModelManifest{
		ID:           "gpt-oss-20b",
		Name:         "OpenAI GPT-OSS-20B",
		Family:       FamilyOpenAI,
		Architecture: ArchSparseMoE,
		Attention:    AttentionFullMHA,
		MoE: &MoETopology{
			NumExperts:  32,
			TopK:        4,
			RoutingType: "softmax",
		},
		Resources: ResourceSpec{
			MinRAMBytes:          14 << 30, // 14 GiB
			MinVRAMBytes:         6 << 30,  // 6 GiB
			LayersTotal:          32,
			HiddenDim:            4096,
			NumHeads:             32,
			NumKVHeads:           8,
			MaxContextTokens:     131072,
			DefaultContextTokens: 32768,
			SupportsMTP:          true,
			SupportsAbliteration: true,
		},
		Aliases:         []string{"gpt-oss", "gpt-oss-20b", "openai/gpt-oss-20b", "openai-gpt-oss-20b", "unsloth/gpt-oss-20b-gguf", "unsloth/gpt-oss-20b", "gpt-oss-20b-gguf", "unsloth-gpt-oss-20b"},
		Description:     "OpenAI open-weight 21B MoE reasoning model with 3.6B active parameters, SwiGLU activations, and CoT reasoning",
		HuggingFaceRepo: "unsloth/gpt-oss-20b-GGUF",
		DefaultGGUF:     "gpt-oss-20b-UD-Q4_K_XL.gguf",
	})
}

// RegisterManifest registers a new model manifest into the catalog
func (r *ModelRegistry) RegisterManifest(m ModelManifest) error {
	if err := m.Validate(); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	id := strings.ToLower(m.ID)
	r.manifests[id] = &m
	r.aliasMap[id] = id

	for _, alias := range m.Aliases {
		a := strings.ToLower(strings.TrimSpace(alias))
		if a != "" {
			r.aliasMap[a] = id
		}
	}
	return nil
}

// GetManifest retrieves a manifest by ID or alias
func (r *ModelRegistry) GetManifest(idOrAlias string) (*ModelManifest, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	key := strings.ToLower(strings.TrimSpace(idOrAlias))
	if canonical, ok := r.aliasMap[key]; ok {
		if m, exists := r.manifests[canonical]; exists {
			return m, true
		}
	}
	if m, ok := r.manifests[key]; ok {
		return m, true
	}
	return nil, false
}

// ListManifests returns all registered manifests
func (r *ModelRegistry) ListManifests() []ModelManifest {
	r.mu.RLock()
	defer r.mu.RUnlock()

	res := make([]ModelManifest, 0, len(r.manifests))
	for _, m := range r.manifests {
		res = append(res, *m)
	}
	return res
}

// RegisterRuntime registers or updates a model runtime profile
func (r *ModelRegistry) RegisterRuntime(p ModelRuntimeProfile) error {
	if strings.TrimSpace(p.ID) == "" {
		return fmt.Errorf("runtime profile ID cannot be empty")
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	id := strings.ToLower(p.ID)
	r.runtimes[id] = &p
	r.aliasMap[id] = id
	return nil
}

// GetRuntime retrieves a model runtime profile by ID
func (r *ModelRegistry) GetRuntime(id string) (*ModelRuntimeProfile, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	key := strings.ToLower(strings.TrimSpace(id))
	if canonical, ok := r.aliasMap[key]; ok {
		if p, exists := r.runtimes[canonical]; exists {
			return p, true
		}
	}
	if p, ok := r.runtimes[key]; ok {
		return p, true
	}
	return nil, false
}

// ListRuntimes returns all configured runtime profiles
func (r *ModelRegistry) ListRuntimes() []ModelRuntimeProfile {
	r.mu.RLock()
	defer r.mu.RUnlock()

	res := make([]ModelRuntimeProfile, 0, len(r.runtimes))
	for _, p := range r.runtimes {
		res = append(res, *p)
	}
	return res
}

// ResolveModel resolves an identifier or alias to a canonical ID, falling back to defaultID
func (r *ModelRegistry) ResolveModel(requested string, defaultID string) string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	req := strings.ToLower(strings.TrimSpace(requested))
	if req == "" {
		return defaultID
	}
	if canonical, ok := r.aliasMap[req]; ok {
		return canonical
	}
	// Return the requested string if unrecognized
	return requested
}

// OpenAIList produces an OpenAI `/v1/models` data list
func (r *ModelRegistry) OpenAIList(created int64, defaultVirt int) []map[string]interface{} {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if created <= 0 {
		created = time.Now().Unix()
	}

	seen := make(map[string]bool)
	var list []map[string]interface{}

	// Add runtime profiles first
	for _, prof := range r.runtimes {
		if !prof.Enabled && prof.Status == "disabled" {
			continue
		}
		seen[strings.ToLower(prof.ID)] = true
		ctx := prof.MaxContext
		virt := prof.VirtualLimit
		if virt <= 0 {
			virt = defaultVirt
		}
		if ctx <= 0 {
			ctx = 32768
		}
		entry := map[string]interface{}{
			"id":                      prof.ID,
			"object":                  "model",
			"created":                 created,
			"owned_by":                "strata",
			"context_window":          virt,
			"max_context_length":      virt,
			"context_length":          virt,
			"max_tokens":              virt,
			"max_input_tokens":        virt,
			"max_output_tokens":       16384,
			"physical_context_window": ctx,
			"physical_context_length": ctx,
			"virtual_context_limit":   virt,
			"virtual_context_tokens":  virt,
			"runtime_status":          prof.Status,
			"permission": []map[string]interface{}{
				{
					"id":                   "modelperm-" + prof.ID,
					"object":               "model_permission",
					"created":              created,
					"allow_create_engine":  false,
					"allow_sampling":       true,
					"allow_logprobs":       true,
					"allow_search_indices": false,
					"allow_view":           true,
					"allow_fine_tuning":    false,
					"organization":         "*",
					"group":                nil,
					"is_blocking":          false,
				},
			},
		}
		list = append(list, entry)
	}

	// Add manifests not shadowed by a runtime profile
	for _, m := range r.manifests {
		idLower := strings.ToLower(m.ID)
		if !seen[idLower] {
			seen[idLower] = true
			list = append(list, m.ToOpenAIModel(created, defaultVirt))
		}
	}

	return list
}

// OpenAIEntry produces a single OpenAI model entry
func (r *ModelRegistry) OpenAIEntry(id string, created int64, defaultCtx int, defaultVirt int) map[string]interface{} {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if created <= 0 {
		created = time.Now().Unix()
	}
	key := strings.ToLower(strings.TrimSpace(id))
	canonical := key
	if c, ok := r.aliasMap[key]; ok {
		canonical = c
	}

	if prof, ok := r.runtimes[canonical]; ok {
		ctx := prof.MaxContext
		virt := prof.VirtualLimit
		if virt <= 0 {
			virt = defaultVirt
		}
		if ctx <= 0 {
			ctx = defaultCtx
		}
		return map[string]interface{}{
			"id":                      prof.ID,
			"object":                  "model",
			"created":                 created,
			"owned_by":                "strata",
			"context_window":          virt,
			"max_context_length":      virt,
			"context_length":          virt,
			"max_tokens":              virt,
			"max_input_tokens":        virt,
			"max_output_tokens":       16384,
			"physical_context_window": ctx,
			"physical_context_length": ctx,
			"virtual_context_limit":   virt,
			"virtual_context_tokens":  virt,
			"runtime_status":          prof.Status,
			"permission": []map[string]interface{}{
				{
					"id":                   "modelperm-" + prof.ID,
					"object":               "model_permission",
					"created":              created,
					"allow_create_engine":  false,
					"allow_sampling":       true,
					"allow_logprobs":       true,
					"allow_search_indices": false,
					"allow_view":           true,
					"allow_fine_tuning":    false,
					"organization":         "*",
					"group":                nil,
					"is_blocking":          false,
				},
			},
		}
	}

	if m, ok := r.manifests[canonical]; ok {
		return m.ToOpenAIModel(created, defaultVirt)
	}

	// Fallback generic entry
	virt := defaultVirt
	if virt <= 0 {
		virt = 2000000
	}
	ctx := defaultCtx
	if ctx <= 0 {
		ctx = 32768
	}
	return map[string]interface{}{
		"id":                      id,
		"object":                  "model",
		"created":                 created,
		"owned_by":                "strata",
		"context_window":          virt,
		"max_context_length":      virt,
		"context_length":          virt,
		"max_tokens":              virt,
		"max_input_tokens":        virt,
		"max_output_tokens":       16384,
		"physical_context_window": ctx,
		"physical_context_length": ctx,
		"virtual_context_limit":   virt,
		"virtual_context_tokens":  virt,
		"permission": []map[string]interface{}{
			{
				"id":                   "modelperm-" + id,
				"object":               "model_permission",
				"created":              created,
				"allow_create_engine":  false,
				"allow_sampling":       true,
				"allow_logprobs":       true,
				"allow_search_indices": false,
				"allow_view":           true,
				"allow_fine_tuning":    false,
				"organization":         "*",
				"group":                nil,
				"is_blocking":          false,
			},
		},
	}
}
