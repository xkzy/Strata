// pkg/modelschema/schema.go - Structured schema definitions for multi-model Strata engine
package modelschema

import (
	"errors"
	"strings"
)

// ModelFamily defines the foundational model architecture family
type ModelFamily string

const (
	FamilyQwen     ModelFamily = "qwen"
	FamilyMiMo     ModelFamily = "mimo"
	FamilyMamba    ModelFamily = "mamba"
	FamilyDeepSeek ModelFamily = "deepseek"
	FamilyLlama    ModelFamily = "llama"
	FamilyMistral  ModelFamily = "mistral"
	FamilyOpenAI   ModelFamily = "openai"
	FamilyGeneric  ModelFamily = "generic"
)

// Architecture classifies the core compute structure
type Architecture string

const (
	ArchHybridQSA               Architecture = "hybrid_qsa"                 // QSA + GDN hybrid recurrence
	ArchDenseTransformer        Architecture = "dense_transformer"          // Standard dense transformer
	ArchSparseMoE               Architecture = "sparse_moe"                 // Sparse Mixture-of-Experts
	ArchAsymmetricMoE           Architecture = "asymmetric_moe"             // Asymmetric QK/V MoE with SWA (e.g. MiMo-V2.6)
	ArchStateSpaceModel         Architecture = "state_space_model"          // Selective State-Space Model (e.g. Mamba-1/2)
	ArchMultiHeadLatentAttention Architecture = "multi_head_latent_attention" // MLA + DeepSeekMoE
)

// AttentionKind defines the attention / recurrence mechanism
type AttentionKind string

const (
	AttentionFullMHA       AttentionKind = "full_mha"
	AttentionGQA           AttentionKind = "gqa"
	AttentionSlidingWindow AttentionKind = "sliding_window"
	AttentionAsymmetricSWA AttentionKind = "asymmetric_swa"
	AttentionSelectiveSSM  AttentionKind = "selective_ssm"
	AttentionMLA           AttentionKind = "mla"
)

// MoETopology defines Mixture-of-Experts geometry
type MoETopology struct {
	NumExperts             int    `json:"num_experts"`
	TopK                   int    `json:"top_k"`
	ExpertIntermediateDim  int    `json:"expert_intermediate_dim,omitempty"`
	SharedExperts          int    `json:"shared_experts,omitempty"`
	RoutingType            string `json:"routing_type"` // "softmax", "sigmoid", "topk", "none"
	DenseLayers            []int  `json:"dense_layers,omitempty"`
	MoELayers              []int  `json:"moe_layers,omitempty"`
}

// SSMTopology defines Selective State-Space Model geometry
type SSMTopology struct {
	StateDim       int    `json:"state_dim"`       // d_state (e.g. 16)
	InnerDim       int    `json:"inner_dim"`       // d_inner (e.g. 5120)
	ConvKernel     int    `json:"conv_kernel"`     // 1D depthwise conv kernel (e.g. 4)
	TimeStepRank   int    `json:"dt_rank"`         // d_delta rank
	Discretization string `json:"discretization"` // "zoh" (Zero-Order Hold)
}

// ResourceSpec specifies compute and memory envelopes
type ResourceSpec struct {
	MinRAMBytes          int64 `json:"min_ram_bytes"`
	MinVRAMBytes         int64 `json:"min_vram_bytes"`
	RecommendedVRAMBytes int64 `json:"recommended_vram_bytes,omitempty"`
	LayersTotal          int   `json:"layers_total"`
	HiddenDim            int   `json:"hidden_dim"`
	NumHeads             int   `json:"num_heads"`
	NumKVHeads           int   `json:"num_kv_heads"`
	HeadDimQK            int   `json:"head_dim_qk,omitempty"`
	HeadDimV             int   `json:"head_dim_v,omitempty"`
	MaxContextTokens     int   `json:"max_context_tokens"`
	DefaultContextTokens int   `json:"default_context_tokens"`
	SupportsMTP          bool  `json:"supports_mtp,omitempty"`
	SupportsAbliteration bool  `json:"supports_abliteration,omitempty"`
}

// ModelManifest represents the canonical schema of a supported model
type ModelManifest struct {
	ID              string        `json:"id"`
	Name            string        `json:"name"`
	Family          ModelFamily   `json:"family"`
	Architecture    Architecture  `json:"architecture"`
	Attention       AttentionKind `json:"attention"`
	MoE             *MoETopology  `json:"moe,omitempty"`
	SSM             *SSMTopology  `json:"ssm,omitempty"`
	Resources       ResourceSpec  `json:"resources"`
	Aliases         []string      `json:"aliases,omitempty"`
	Description     string        `json:"description,omitempty"`
	HuggingFaceRepo string        `json:"huggingface_repo,omitempty"`
	DefaultGGUF     string        `json:"default_gguf,omitempty"`
}

// Validate checks whether manifest has required attributes
func (m *ModelManifest) Validate() error {
	if strings.TrimSpace(m.ID) == "" {
		return errors.New("model manifest ID cannot be empty")
	}
	if strings.TrimSpace(m.Name) == "" {
		return errors.New("model manifest Name cannot be empty")
	}
	if m.Family == "" {
		return errors.New("model manifest Family must be specified")
	}
	if m.Architecture == "" {
		return errors.New("model manifest Architecture must be specified")
	}
	if m.Resources.MaxContextTokens <= 0 {
		return errors.New("model manifest MaxContextTokens must be > 0")
	}
	if m.Architecture == ArchStateSpaceModel && m.SSM == nil {
		return errors.New("state space model must provide SSM topology")
	}
	if (m.Architecture == ArchSparseMoE || m.Architecture == ArchAsymmetricMoE) && m.MoE == nil {
		return errors.New("MoE model must provide MoE topology")
	}
	return nil
}

// Matches checks if an identifier matches ID or any alias (case-insensitive)
func (m *ModelManifest) Matches(identifier string) bool {
	id := strings.ToLower(strings.TrimSpace(identifier))
	if id == "" {
		return false
	}
	if strings.ToLower(m.ID) == id {
		return true
	}
	for _, alias := range m.Aliases {
		if strings.ToLower(alias) == id {
			return true
		}
	}
	return false
}

// ToOpenAIModel formats manifest into OpenAI-compatible model entry
func (m *ModelManifest) ToOpenAIModel(created int64, defaultVirt int) map[string]interface{} {
	virt := defaultVirt
	if virt < m.Resources.MaxContextTokens {
		virt = m.Resources.MaxContextTokens
	}
	return map[string]interface{}{
		"id":                      m.ID,
		"object":                  "model",
		"created":                 created,
		"owned_by":                "strata",
		"family":                  string(m.Family),
		"architecture":            string(m.Architecture),
		"attention":               string(m.Attention),
		"context_window":          virt,
		"max_context_length":      virt,
		"context_length":          virt,
		"max_tokens":              virt,
		"max_input_tokens":        virt,
		"max_output_tokens":       16384,
		"physical_context_window": m.Resources.MaxContextTokens,
		"physical_context_length": m.Resources.MaxContextTokens,
		"virtual_context_limit":   virt,
		"virtual_context_tokens":  virt,
		"permission": []map[string]interface{}{
			{
				"id":                   "modelperm-" + m.ID,
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
