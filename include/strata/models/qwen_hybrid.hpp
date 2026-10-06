// include/strata/models/qwen_hybrid.hpp - High-Performance Qwen Architecture Optimizations
//
// Specialized optimization engine for Qwen3.8-Flash-Next and its variants (Swift 1.5, Coder, Unsloth).
// Implements:
// 1. Monotonic Top-10 routing with 51x transcendental FLOP reduction.
// 2. High-performance Gated Delta Net (GDN) linear recurrent state update.
// 3. Fused Gated Residual (GR) 4-stream low-rank accumulation (hc=4, hc_lr=320).
// 4. Hybrid memory footprint calculation for 75% GDN + 25% SWA.
#pragma once

#include "strata/models/model_adapter.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace strata::models::qwen {

/// Configuration constants for Qwen3.8-Flash-Next
struct QwenOptimConfig {
    static constexpr int64_t N_LAYERS = 48;
    static constexpr int64_t N_EMBD = 2560;
    static constexpr int64_t N_EXPERTS = 512;
    static constexpr int64_t ACTIVE_EXPERTS = 10;
    static constexpr int64_t HYBRID_INTERVAL = 4; // 3 GDN layers, 1 QSA layer
    static constexpr int64_t SSM_STATE_SIZE = 128;
    static constexpr int64_t SSM_K_HEADS = 16;
    static constexpr int64_t SSM_V_HEADS = 48;
    static constexpr int64_t HC_STREAMS = 4;
    static constexpr int64_t HC_LR_DIM = 320;
    static constexpr int64_t SLIDING_WINDOW = 4096;
};

/// High-performance Qwen Routing Optimizer
class QwenRoutingOptimizer {
public:
    /// Selects top-10 experts from 512 routing logits with zero heap allocation and 51x fewer expf calls.
    /// Exploits monotonicity of sigmoid: sigma(a) > sigma(b) <=> a > b.
    static void route_top10(const float* routing_logits, int32_t* selected_experts_out, float* weights_out);
};

/// Gated Delta Net (GDN) State for linear attention layers
struct QwenGDNState {
    int64_t k_heads = QwenOptimConfig::SSM_K_HEADS;
    int64_t v_heads = QwenOptimConfig::SSM_V_HEADS;
    int64_t state_dim = QwenOptimConfig::SSM_STATE_SIZE;
    std::vector<float> state_tensor; // shape: [v_heads, state_dim, state_dim]

    QwenGDNState();
    void reset();
    size_t memory_bytes() const;
};

/// High-Capacity Gated Residual State (hc = 4)
struct QwenGatedResidualState {
    int64_t n_streams = QwenOptimConfig::HC_STREAMS;
    int64_t n_embd = QwenOptimConfig::N_EMBD;
    std::vector<float> streams; // shape: [4, 2560]

    QwenGatedResidualState();
    void reset();
    void accumulate(const float* layer_delta, const float* gate_weights);
};

/// Specialized Execution Engine for Qwen Models
class QwenExecutionEngine {
public:
    /// Executes one recurrent GDN step on CPU / Host
    static void gdn_step(QwenGDNState& gdn_state,
                         const float* q,
                         const float* k,
                         const float* v,
                         const float* gate,
                         const float* beta,
                         float* output,
                         float scale = 1.0f / 11.3137f);

    /// Fused 4-stream Gated Residual accumulation and normalization
    static void fused_gated_residual(QwenGatedResidualState& gr_state,
                                     const float* layer_out,
                                     const float* residual_gate,
                                     float* merged_out);

    /// Calculates exact KV + Recurrence cache memory requirements across full sequence length
    struct MemoryBreakdown {
        size_t gdn_fixed_bytes = 0;       // Constant across all sequence lengths
        size_t qsa_unbounded_kv_bytes = 0;// Unbounded traditional KV cache
        size_t qsa_swa_bounded_kv_bytes = 0; // Bounded SWA ring buffer KV cache
        double memory_saving_ratio = 1.0;
    };

    static MemoryBreakdown calculate_memory_footprint(int64_t sequence_length, int64_t window_size = 4096);
};

} // namespace strata::models::qwen
