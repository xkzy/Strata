// include/strata/models/model_adapter.hpp - Generic MoE Model Interface & Adapter Architecture
//
// Decouples model architectures from execution logic. Models (Qwen, Mixtral, DeepSeek,
// GLM, custom MoE) provide geometry, routing algorithms, tensor manifests, and layer configurations
// through this interface without hardcoding values in the runtime.
#pragma once

#include "strata/core/weights.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::models {

enum class RoutingType {
    kTopKSoftmax = 0,
    kTopKSigmoid,
    kExpertChoice,
    kSharedRoutedBias,
    kCustomRouting
};

enum class AttentionType {
    kStandardMHA = 0,
    kGroupedQueryAttn,
    kMultiHeadLatentAttn,   // DeepSeek MLA
    kSparseHybridAttn,      // QSA + GDN hybrid
    kSlidingWindowAttn      // Interleaved SWA (e.g. MiMo-V2.6)
};

enum class LayerType {
    kStandardMoE = 0,
    kHybridDenseMoE,
    kSharedExpertMoE,
    kPureDense
};

struct MoEModelConfig {
    std::string model_name = "Generic-MoE";
    std::string architecture = "generic_moe";

    int64_t n_embd = 2560;
    int64_t n_layers = 48;
    int64_t n_head = 24;
    int64_t n_head_kv = 2;
    int64_t head_dim = 256;

    // MoE geometry
    int64_t n_expert = 512;
    int64_t active_experts = 10;
    int64_t n_shared_experts = 0;
    int64_t n_ff = 640;
    int64_t shared_n_ff = 0;

    // Speculative Multi-Token Prediction (MTP) & SWA extensions
    int64_t mtp_layers = 0;
    int64_t sliding_window = 4096;

    // Hybrid/Recurrence extensions (if applicable)
    int64_t hybrid_interval = 4;
    int64_t ssm_state_size = 0;
    int64_t ssm_k_heads = 0;
    int64_t ssm_v_heads = 0;
    int64_t ssm_d_conv = 0;
    int64_t ssm_conv_channels = 0;
    int64_t ssm_value_dim = 0;

    // Attention parameters
    int64_t idx_q_heads = 4;
    int64_t idx_key_dim = 128;

    // Gated residual & norm
    int64_t hc = 4;
    int64_t hc_lr = 320;
    float norm_eps = 1e-6f;

    // Enums
    RoutingType routing_type = RoutingType::kTopKSigmoid;
    AttentionType attention_type = AttentionType::kSparseHybridAttn;

    int64_t hc_dim() const { return hc * n_embd; }
};

class MoEModelAdapter {
public:
    virtual ~MoEModelAdapter() = default;

    virtual const MoEModelConfig& config() const = 0;
    virtual std::string architecture_name() const = 0;

    // Tensor name resolution per layer & role
    virtual std::string tensor_name(int64_t layer, const std::string& role) const = 0;

    // Validate whether a loaded weight table matches this model's required tensor shapes
    virtual bool validate_weights(const core::WeightTable& table, std::string& err) const = 0;

    // Layer type classification
    virtual LayerType layer_type(int64_t layer) const = 0;
    virtual bool is_full_attention_layer(int64_t layer) const = 0;

    // Generic routing execution
    virtual void route_token(const float* routing_logits, int64_t n_expert, int64_t top_k,
                             int32_t* selected_experts_out, float* weights_out) const = 0;

    // Sliding Window Attention (SWA) support across all models
    virtual bool supports_swa() const {
        return config().sliding_window > 0;
    }

    virtual int64_t sliding_window_size() const {
        return config().sliding_window;
    }

    virtual std::unique_ptr<MoEModelAdapter> clone() const = 0;
};

// Model Registry for extensible MoE architectures
class ModelAdapterRegistry {
public:
    static ModelAdapterRegistry& instance();

    void register_factory(const std::string& arch_name,
                          std::function<std::unique_ptr<MoEModelAdapter>(const MoEModelConfig*)> factory);

    std::unique_ptr<MoEModelAdapter> create(const std::string& arch_name,
                                            const MoEModelConfig* custom_cfg = nullptr) const;

    std::vector<std::string> available_architectures() const;

    // Auto-detect architecture from loaded weights
    std::unique_ptr<MoEModelAdapter> detect_and_create(const core::WeightTable& table) const;

private:
    ModelAdapterRegistry();
    std::unordered_map<std::string, std::function<std::unique_ptr<MoEModelAdapter>(const MoEModelConfig*)>> factories_;
};

} // namespace strata::models
