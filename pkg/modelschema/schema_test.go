// pkg/modelschema/schema_test.go - Unit Tests for Multi-Model Schema and Registry
package modelschema

import (
	"testing"
)

func TestModelManifest_Validation(t *testing.T) {
	// Valid manifest
	valid := ModelManifest{
		ID:           "test-model",
		Name:         "Test Model",
		Family:       FamilyQwen,
		Architecture: ArchDenseTransformer,
		Attention:    AttentionFullMHA,
		Resources: ResourceSpec{
			MaxContextTokens: 32768,
		},
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("expected valid manifest, got error: %v", err)
	}

	// Empty ID
	invalid := valid
	invalid.ID = ""
	if err := invalid.Validate(); err == nil {
		t.Fatal("expected error for empty ID, got nil")
	}

	// Missing SSM for state space model
	ssmInvalid := ModelManifest{
		ID:           "test-mamba",
		Name:         "Test Mamba",
		Family:       FamilyMamba,
		Architecture: ArchStateSpaceModel,
		Attention:    AttentionSelectiveSSM,
		Resources: ResourceSpec{
			MaxContextTokens: 65536,
		},
	}
	if err := ssmInvalid.Validate(); err == nil {
		t.Fatal("expected error for SSM model without SSM topology, got nil")
	}

	// Missing MoE topology for MoE model
	moeInvalid := ModelManifest{
		ID:           "test-moe",
		Name:         "Test MoE",
		Family:       FamilyQwen,
		Architecture: ArchSparseMoE,
		Attention:    AttentionFullMHA,
		Resources: ResourceSpec{
			MaxContextTokens: 32768,
		},
	}
	if err := moeInvalid.Validate(); err == nil {
		t.Fatal("expected error for MoE model without MoE topology, got nil")
	}
}

func TestModelRegistry_Builtins(t *testing.T) {
	reg := NewRegistry()

	// 1. Qwen
	qwen, ok := reg.GetManifest("qwen3.8-flash-next")
	if !ok || qwen == nil {
		t.Fatal("failed to find built-in qwen3.8-flash-next")
	}
	if qwen.Family != FamilyQwen || qwen.Architecture != ArchHybridQSA {
		t.Fatalf("unexpected qwen properties: family=%s, arch=%s", qwen.Family, qwen.Architecture)
	}

	// 2. MiMo-V2.6-Flash-RL
	mimo, ok := reg.GetManifest("mimo-v2.6-flash-rl")
	if !ok || mimo == nil {
		t.Fatal("failed to find built-in mimo-v2.6-flash-rl")
	}
	if mimo.Family != FamilyMiMo || mimo.Architecture != ArchAsymmetricMoE {
		t.Fatalf("unexpected mimo properties: family=%s, arch=%s", mimo.Family, mimo.Architecture)
	}
	if mimo.Resources.HeadDimQK != 192 || mimo.Resources.HeadDimV != 128 {
		t.Fatalf("unexpected mimo asymmetric head dims: QK=%d, V=%d", mimo.Resources.HeadDimQK, mimo.Resources.HeadDimV)
	}
	if !mimo.Resources.SupportsMTP {
		t.Fatal("expected MiMo to support MTP speculative drafting")
	}

	// 3. Mamba
	mamba, ok := reg.GetManifest("mamba-2.8b")
	if !ok || mamba == nil {
		t.Fatal("failed to find built-in mamba-2.8b")
	}
	if mamba.Family != FamilyMamba || mamba.Architecture != ArchStateSpaceModel {
		t.Fatalf("unexpected mamba properties: family=%s, arch=%s", mamba.Family, mamba.Architecture)
	}
	if mamba.SSM == nil || mamba.SSM.StateDim != 16 || mamba.SSM.InnerDim != 5120 {
		t.Fatalf("unexpected mamba SSM topology: %+v", mamba.SSM)
	}

	// 4. DeepSeek
	deepseek, ok := reg.GetManifest("deepseek-v3")
	if !ok || deepseek == nil {
		t.Fatal("failed to find built-in deepseek-v3")
	}
	if deepseek.Family != FamilyDeepSeek || deepseek.Architecture != ArchMultiHeadLatentAttention {
		t.Fatalf("unexpected deepseek properties: family=%s, arch=%s", deepseek.Family, deepseek.Architecture)
	}
}

func TestModelRegistry_AliasesAndResolve(t *testing.T) {
	reg := NewRegistry()

	// Alias lookups
	tests := []struct {
		input    string
		expected string
	}{
		{"qwen", "qwen3.8-flash-next"},
		{"strata", "qwen3.8-flash-next"},
		{"coder", "qwen3.8-flash-next-coder"},
		{"mimo", "mimo-v2.6-flash-rl"},
		{"mimo-v2.6", "mimo-v2.6-flash-rl"},
		{"mamba", "mamba-2.8b"},
		{"mamba-ssm", "mamba-2.8b"},
		{"deepseek", "deepseek-v3"},
		{"deepseek-r1", "deepseek-v3"},
	}

	for _, tt := range tests {
		resolved := reg.ResolveModel(tt.input, "fallback")
		if resolved != tt.expected {
			t.Errorf("ResolveModel(%q) = %q; expected %q", tt.input, resolved, tt.expected)
		}

		m, ok := reg.GetManifest(tt.input)
		if !ok || m.ID != tt.expected {
			t.Errorf("GetManifest(%q) failed: got %v, expected %q", tt.input, m, tt.expected)
		}
	}

	// Unknown resolution fallback
	if got := reg.ResolveModel("", "default-id"); got != "default-id" {
		t.Fatalf("expected default-id on empty input, got %s", got)
	}
	if got := reg.ResolveModel("custom-unregistered", "default-id"); got != "custom-unregistered" {
		t.Fatalf("expected custom-unregistered preserved, got %s", got)
	}
}

func TestModelRegistry_RuntimeProfilesAndOpenAI(t *testing.T) {
	reg := NewRegistry()

	prof := ModelRuntimeProfile{
		ID:           "mimo-custom",
		ManifestID:   "mimo-v2.6-flash-rl",
		ModelPath:    "/models/mimo.gguf",
		MaxContext:   65536,
		VirtualLimit: 2000000,
		Enabled:      true,
		Status:       "ready",
	}

	if err := reg.RegisterRuntime(prof); err != nil {
		t.Fatalf("failed to register runtime profile: %v", err)
	}

	gotProf, ok := reg.GetRuntime("mimo-custom")
	if !ok || gotProf.MaxContext != 65536 {
		t.Fatalf("failed to retrieve runtime profile: %+v", gotProf)
	}

	// OpenAI List
	list := reg.OpenAIList(1234567890, 2000000)
	if len(list) < 5 {
		t.Fatalf("expected at least 5 models in OpenAI list, got %d", len(list))
	}

	// Check runtime profile appears in OpenAI list
	foundCustom := false
	for _, item := range list {
		if item["id"] == "mimo-custom" {
			foundCustom = true
			if item["virtual_context_limit"] != 2000000 {
				t.Errorf("unexpected virtual_context_limit: %v", item["virtual_context_limit"])
			}
			if item["runtime_status"] != "ready" {
				t.Errorf("unexpected runtime_status: %v", item["runtime_status"])
			}
		}
	}
	if !foundCustom {
		t.Fatal("custom runtime profile not found in OpenAI list")
	}

	// OpenAI Entry
	entry := reg.OpenAIEntry("mimo-custom", 1234567890, 32768, 2000000)
	if entry["id"] != "mimo-custom" || entry["context_window"] != 2000000 {
		t.Fatalf("unexpected OpenAI entry: %+v", entry)
	}
}
