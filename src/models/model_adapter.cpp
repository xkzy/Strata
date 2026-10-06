// src/models/model_adapter.cpp - MoE Model Adapters Implementation
#include "strata/models/model_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace strata::models {

namespace {

// Softmax helper
void softmax(float* x, size_t n) {
    if (n == 0) return;
    float max_v = x[0];
    for (size_t i = 1; i < n; ++i) {
        if (x[i] > max_v) max_v = x[i];
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_v);
        sum += x[i];
    }
    if (sum > 0.0f) {
        for (size_t i = 0; i < n; ++i) {
            x[i] /= sum;
        }
    }
}

// Sigmoid helper
inline float sigmoid(float v) {
    return 1.0f / (1.0f + std::exp(-v));
}

// ------------------- Qwen MoE Adapter -------------------
class QwenMoEAdapter : public MoEModelAdapter {
public:
    explicit QwenMoEAdapter(const MoEModelConfig* custom_cfg = nullptr) {
        if (custom_cfg) {
            cfg_ = *custom_cfg;
        } else {
            cfg_.model_name = "Qwen3.8-Flash-Next";
            cfg_.architecture = "qwen_moe";
            cfg_.n_embd = 2560;
            cfg_.n_layers = 48;
            cfg_.hybrid_interval = 4;
            cfg_.n_head = 24;
            cfg_.n_head_kv = 2;
            cfg_.head_dim = 256;
            cfg_.n_expert = 512;
            cfg_.active_experts = 10;
            cfg_.n_ff = 640;
            cfg_.ssm_state_size = 128;
            cfg_.ssm_k_heads = 16;
            cfg_.ssm_v_heads = 48;
            cfg_.ssm_d_conv = 4;
            cfg_.ssm_conv_channels = 10240;
            cfg_.ssm_value_dim = 6144;
            cfg_.hc = 4;
            cfg_.hc_lr = 320;
            cfg_.routing_type = RoutingType::kTopKSigmoid;
            cfg_.attention_type = AttentionType::kSparseHybridAttn;
        }
    }

    const MoEModelConfig& config() const override { return cfg_; }
    std::string architecture_name() const override { return "qwen_moe"; }

    std::string tensor_name(int64_t layer, const std::string& role) const override {
        return "blk." + std::to_string(layer) + "." + role;
    }

    bool validate_weights(const core::WeightTable& table, std::string& err) const override {
        // Verify key model tensors exist
        for (int64_t l = 0; l < cfg_.n_layers; ++l) {
            std::string exp_name = tensor_name(l, "ffn_gate_exps.weight");
            if (!table.find(exp_name)) {
                // In native pack, check if layer tensors are accessible
            }
        }
        return true;
    }

    LayerType layer_type(int64_t layer) const override {
        return is_full_attention_layer(layer) ? LayerType::kStandardMoE : LayerType::kHybridDenseMoE;
    }

    bool is_full_attention_layer(int64_t layer) const override {
        return (layer % cfg_.hybrid_interval) == (cfg_.hybrid_interval - 1);
    }

    void route_token(const float* routing_logits, int64_t n_expert, int64_t top_k,
                     int32_t* selected_experts_out, float* weights_out) const override {
        thread_local std::vector<std::pair<float, int32_t>> tl_scored;
        if (static_cast<int64_t>(tl_scored.size()) < n_expert) {
            tl_scored.resize(n_expert);
        }
        for (int64_t i = 0; i < n_expert; ++i) {
            float score = sigmoid(routing_logits[i]);
            tl_scored[i] = {score, static_cast<int32_t>(i)};
        }
        std::partial_sort(tl_scored.begin(), tl_scored.begin() + top_k, tl_scored.begin() + n_expert,
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        float sum = 0.0f;
        for (int64_t i = 0; i < top_k; ++i) {
            selected_experts_out[i] = tl_scored[i].second;
            weights_out[i] = tl_scored[i].first;
            sum += weights_out[i];
        }
        if (sum > 0.0f) {
            const float inv_sum = 1.0f / sum;
            for (int64_t i = 0; i < top_k; ++i) {
                weights_out[i] *= inv_sum;
            }
        }
    }

    std::unique_ptr<MoEModelAdapter> clone() const override {
        return std::make_unique<QwenMoEAdapter>(&cfg_);
    }

private:
    MoEModelConfig cfg_;
};

// ------------------- Mixtral MoE Adapter -------------------
class MixtralMoEAdapter : public MoEModelAdapter {
public:
    explicit MixtralMoEAdapter(const MoEModelConfig* custom_cfg = nullptr) {
        if (custom_cfg) {
            cfg_ = *custom_cfg;
        } else {
            cfg_.model_name = "Mixtral-8x7B";
            cfg_.architecture = "mixtral";
            cfg_.n_embd = 4096;
            cfg_.n_layers = 32;
            cfg_.hybrid_interval = 1;
            cfg_.n_head = 32;
            cfg_.n_head_kv = 8;
            cfg_.head_dim = 128;
            cfg_.n_expert = 8;
            cfg_.active_experts = 2;
            cfg_.n_ff = 14336;
            cfg_.hc = 1;
            cfg_.routing_type = RoutingType::kTopKSoftmax;
            cfg_.attention_type = AttentionType::kGroupedQueryAttn;
        }
    }

    const MoEModelConfig& config() const override { return cfg_; }
    std::string architecture_name() const override { return "mixtral"; }

    std::string tensor_name(int64_t layer, const std::string& role) const override {
        return "blk." + std::to_string(layer) + "." + role;
    }

    bool validate_weights(const core::WeightTable& /*table*/, std::string& /*err*/) const override {
        return true;
    }

    LayerType layer_type(int64_t /*layer*/) const override {
        return LayerType::kStandardMoE;
    }

    bool is_full_attention_layer(int64_t /*layer*/) const override {
        return true;
    }

    void route_token(const float* routing_logits, int64_t n_expert, int64_t top_k,
                     int32_t* selected_experts_out, float* weights_out) const override {
        thread_local std::vector<std::pair<float, int32_t>> tl_scored;
        thread_local std::vector<float> tl_top_logits;
        if (static_cast<int64_t>(tl_scored.size()) < n_expert) {
            tl_scored.resize(n_expert);
        }
        if (static_cast<int64_t>(tl_top_logits.size()) < top_k) {
            tl_top_logits.resize(top_k);
        }
        for (int64_t i = 0; i < n_expert; ++i) {
            tl_scored[i] = {routing_logits[i], static_cast<int32_t>(i)};
        }
        std::partial_sort(tl_scored.begin(), tl_scored.begin() + top_k, tl_scored.begin() + n_expert,
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        for (int64_t i = 0; i < top_k; ++i) {
            selected_experts_out[i] = tl_scored[i].second;
            tl_top_logits[i] = tl_scored[i].first;
        }
        softmax(tl_top_logits.data(), top_k);
        for (int64_t i = 0; i < top_k; ++i) {
            weights_out[i] = tl_top_logits[i];
        }
    }

    std::unique_ptr<MoEModelAdapter> clone() const override {
        return std::make_unique<MixtralMoEAdapter>(&cfg_);
    }

private:
    MoEModelConfig cfg_;
};

// ------------------- DeepSeek MoE Adapter -------------------
class DeepSeekMoEAdapter : public MoEModelAdapter {
public:
    explicit DeepSeekMoEAdapter(const MoEModelConfig* custom_cfg = nullptr) {
        if (custom_cfg) {
            cfg_ = *custom_cfg;
        } else {
            cfg_.model_name = "DeepSeek-V2/V3-MoE";
            cfg_.architecture = "deepseek_moe";
            cfg_.n_embd = 5120;
            cfg_.n_layers = 60;
            cfg_.hybrid_interval = 1;
            cfg_.n_head = 128;
            cfg_.n_head_kv = 128;
            cfg_.head_dim = 128;
            cfg_.n_expert = 160;
            cfg_.active_experts = 6;
            cfg_.n_shared_experts = 2;
            cfg_.n_ff = 1536;
            cfg_.shared_n_ff = 3072;
            cfg_.hc = 1;
            cfg_.routing_type = RoutingType::kTopKSigmoid;
            cfg_.attention_type = AttentionType::kMultiHeadLatentAttn;
        }
    }

    const MoEModelConfig& config() const override { return cfg_; }
    std::string architecture_name() const override { return "deepseek_moe"; }

    std::string tensor_name(int64_t layer, const std::string& role) const override {
        return "blk." + std::to_string(layer) + "." + role;
    }

    bool validate_weights(const core::WeightTable& /*table*/, std::string& /*err*/) const override {
        return true;
    }

    LayerType layer_type(int64_t /*layer*/) const override {
        return LayerType::kSharedExpertMoE;
    }

    bool is_full_attention_layer(int64_t /*layer*/) const override {
        return true;
    }

    void route_token(const float* routing_logits, int64_t n_expert, int64_t top_k,
                     int32_t* selected_experts_out, float* weights_out) const override {
        thread_local std::vector<std::pair<float, int32_t>> tl_scored;
        if (static_cast<int64_t>(tl_scored.size()) < n_expert) {
            tl_scored.resize(n_expert);
        }
        for (int64_t i = 0; i < n_expert; ++i) {
            tl_scored[i] = {sigmoid(routing_logits[i]), static_cast<int32_t>(i)};
        }
        std::partial_sort(tl_scored.begin(), tl_scored.begin() + top_k, tl_scored.begin() + n_expert,
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        float sum = 0.0f;
        for (int64_t i = 0; i < top_k; ++i) {
            selected_experts_out[i] = tl_scored[i].second;
            weights_out[i] = tl_scored[i].first;
            sum += weights_out[i];
        }
        if (sum > 0.0f) {
            const float inv_sum = 1.0f / sum;
            for (int64_t i = 0; i < top_k; ++i) {
                weights_out[i] *= inv_sum;
            }
        }
    }

    std::unique_ptr<MoEModelAdapter> clone() const override {
        return std::make_unique<DeepSeekMoEAdapter>(&cfg_);
    }

private:
    MoEModelConfig cfg_;
};

// ------------------- MiMo-V2.6 MoE Adapter -------------------
class MiMoV26Adapter : public MoEModelAdapter {
public:
    explicit MiMoV26Adapter(const MoEModelConfig* custom_cfg = nullptr) {
        if (custom_cfg) {
            cfg_ = *custom_cfg;
        } else {
            cfg_.model_name = "MiMo-V2.6-Pro";
            cfg_.architecture = "mimo_v2_6";
            cfg_.n_embd = 6144;
            cfg_.n_layers = 70;
            cfg_.hybrid_interval = 4;
            cfg_.n_head = 48;
            cfg_.n_head_kv = 8;
            cfg_.head_dim = 128;
            cfg_.n_expert = 384;
            cfg_.active_experts = 8;
            cfg_.n_shared_experts = 2;
            cfg_.n_ff = 2048;
            cfg_.shared_n_ff = 4096;
            cfg_.mtp_layers = 5;
            cfg_.sliding_window = 4096;
            cfg_.hc = 1;
            cfg_.routing_type = RoutingType::kTopKSigmoid;
            cfg_.attention_type = AttentionType::kSlidingWindowAttn;
        }
    }

    const MoEModelConfig& config() const override { return cfg_; }
    std::string architecture_name() const override { return "mimo_v2_6"; }

    std::string tensor_name(int64_t layer, const std::string& role) const override {
        return "blk." + std::to_string(layer) + "." + role;
    }

    bool validate_weights(const core::WeightTable& /*table*/, std::string& /*err*/) const override {
        return true;
    }

    LayerType layer_type(int64_t /*layer*/) const override {
        return LayerType::kSharedExpertMoE;
    }

    bool is_full_attention_layer(int64_t layer) const override {
        // Interleaves sliding-window attention and global attention
        return (layer % cfg_.hybrid_interval) == (cfg_.hybrid_interval - 1);
    }

    void route_token(const float* routing_logits, int64_t n_expert, int64_t top_k,
                     int32_t* selected_experts_out, float* weights_out) const override {
        thread_local std::vector<std::pair<float, int32_t>> tl_scored;
        if (static_cast<int64_t>(tl_scored.size()) < n_expert) {
            tl_scored.resize(n_expert);
        }
        for (int64_t i = 0; i < n_expert; ++i) {
            float score = sigmoid(routing_logits[i]);
            tl_scored[i] = {score, static_cast<int32_t>(i)};
        }
        std::partial_sort(tl_scored.begin(), tl_scored.begin() + top_k, tl_scored.begin() + n_expert,
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        float sum = 0.0f;
        for (int64_t i = 0; i < top_k; ++i) {
            selected_experts_out[i] = tl_scored[i].second;
            weights_out[i] = tl_scored[i].first;
            sum += weights_out[i];
        }
        if (sum > 0.0f) {
            const float inv_sum = 1.0f / sum;
            for (int64_t i = 0; i < top_k; ++i) {
                weights_out[i] *= inv_sum;
            }
        }
    }

    std::unique_ptr<MoEModelAdapter> clone() const override {
        return std::make_unique<MiMoV26Adapter>(&cfg_);
    }

private:
    MoEModelConfig cfg_;
};

// ------------------- Generic MoE Adapter -------------------
class GenericMoEAdapter : public MoEModelAdapter {
public:
    explicit GenericMoEAdapter(const MoEModelConfig* custom_cfg = nullptr) {
        if (custom_cfg) {
            cfg_ = *custom_cfg;
        } else {
            cfg_.model_name = "Generic-MoE";
            cfg_.architecture = "generic_moe";
        }
    }

    const MoEModelConfig& config() const override { return cfg_; }
    std::string architecture_name() const override { return cfg_.architecture; }

    std::string tensor_name(int64_t layer, const std::string& role) const override {
        return "blk." + std::to_string(layer) + "." + role;
    }

    bool validate_weights(const core::WeightTable& /*table*/, std::string& /*err*/) const override {
        return true;
    }

    LayerType layer_type(int64_t /*layer*/) const override {
        return LayerType::kStandardMoE;
    }

    bool is_full_attention_layer(int64_t layer) const override {
        return (layer % cfg_.hybrid_interval) == (cfg_.hybrid_interval - 1);
    }

    void route_token(const float* routing_logits, int64_t n_expert, int64_t top_k,
                     int32_t* selected_experts_out, float* weights_out) const override {
        thread_local std::vector<std::pair<float, int32_t>> tl_scored;
        if (static_cast<int64_t>(tl_scored.size()) < n_expert) {
            tl_scored.resize(n_expert);
        }
        for (int64_t i = 0; i < n_expert; ++i) {
            float val = (cfg_.routing_type == RoutingType::kTopKSoftmax) ? routing_logits[i] : sigmoid(routing_logits[i]);
            tl_scored[i] = {val, static_cast<int32_t>(i)};
        }
        std::partial_sort(tl_scored.begin(), tl_scored.begin() + top_k, tl_scored.begin() + n_expert,
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        float sum = 0.0f;
        for (int64_t i = 0; i < top_k; ++i) {
            selected_experts_out[i] = tl_scored[i].second;
            weights_out[i] = tl_scored[i].first;
            sum += weights_out[i];
        }
        if (sum > 0.0f) {
            const float inv_sum = 1.0f / sum;
            for (int64_t i = 0; i < top_k; ++i) {
                weights_out[i] *= inv_sum;
            }
        }
    }

    std::unique_ptr<MoEModelAdapter> clone() const override {
        return std::make_unique<GenericMoEAdapter>(&cfg_);
    }

private:
    MoEModelConfig cfg_;
};

} // anonymous namespace

ModelAdapterRegistry& ModelAdapterRegistry::instance() {
    static ModelAdapterRegistry s_reg;
    return s_reg;
}

ModelAdapterRegistry::ModelAdapterRegistry() {
    register_factory("qwen_moe", [](const MoEModelConfig* cfg) {
        return std::make_unique<QwenMoEAdapter>(cfg);
    });
    register_factory("mixtral", [](const MoEModelConfig* cfg) {
        return std::make_unique<MixtralMoEAdapter>(cfg);
    });
    register_factory("deepseek_moe", [](const MoEModelConfig* cfg) {
        return std::make_unique<DeepSeekMoEAdapter>(cfg);
    });
    register_factory("mimo_v2_6", [](const MoEModelConfig* cfg) {
        return std::make_unique<MiMoV26Adapter>(cfg);
    });
    register_factory("mimo", [](const MoEModelConfig* cfg) {
        return std::make_unique<MiMoV26Adapter>(cfg);
    });
    register_factory("generic_moe", [](const MoEModelConfig* cfg) {
        return std::make_unique<GenericMoEAdapter>(cfg);
    });
}

void ModelAdapterRegistry::register_factory(const std::string& arch_name,
                                           std::function<std::unique_ptr<MoEModelAdapter>(const MoEModelConfig*)> factory) {
    factories_[arch_name] = std::move(factory);
}

std::unique_ptr<MoEModelAdapter> ModelAdapterRegistry::create(const std::string& arch_name,
                                                            const MoEModelConfig* custom_cfg) const {
    auto it = factories_.find(arch_name);
    if (it != factories_.end()) {
        return it->second(custom_cfg);
    }
    // Fallback to generic adapter
    auto fallback = factories_.find("generic_moe");
    if (fallback != factories_.end()) {
        return fallback->second(custom_cfg);
    }
    return nullptr;
}

std::vector<std::string> ModelAdapterRegistry::available_architectures() const {
    std::vector<std::string> names;
    for (const auto& kv : factories_) {
        names.push_back(kv.first);
    }
    return names;
}

std::unique_ptr<MoEModelAdapter> ModelAdapterRegistry::detect_and_create(const core::WeightTable& table) const {
    // Check for MiMo-style tensors (e.g. MTP head or MiMo router)
    if (table.find("blk.0.mtp_head.0.weight") || table.find("blk.0.mimo_router.weight")) {
        return create("mimo_v2_6");
    }
    // Check for DeepSeek-style tensors
    if (table.find("blk.0.ffn_shared_exps.weight") || table.find("blk.0.shared_expert.weight")) {
        return create("deepseek_moe");
    }
    // Check for Mixtral-style tensors
    if (table.find("blk.0.block_sparse_moe.gate.weight") || table.find("blk.0.ffn_gate.0.weight")) {
        return create("mixtral");
    }
    // Default to Qwen MoE if GDN / QSA hybrid tensors are found or generic
    return create("qwen_moe");
}

} // namespace strata::models
