// src/models/qwen_hybrid.cpp - High-Performance Qwen Architecture Optimizations
#include "strata/models/qwen_hybrid.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "strata/kernels/fast_activations.hpp"

namespace strata::models::qwen {

namespace {
inline float fast_sigmoid(float x) {
    return strata::kernels::fast_math::fast_sigmoid(x);
}
} // anonymous namespace

// ------------------- Qwen Routing Optimizer -------------------

void QwenRoutingOptimizer::route_top10(const float* routing_logits,
                                       int32_t* selected_experts_out,
                                       float* weights_out) {
    // Zero-heap L1 stack array for 512 experts (4 KB total)
    std::array<std::pair<float, int32_t>, QwenOptimConfig::N_EXPERTS> candidates;

    for (int32_t i = 0; i < static_cast<int32_t>(QwenOptimConfig::N_EXPERTS); ++i) {
        candidates[i] = {routing_logits[i], i};
    }

    // Exploit strict monotonicity of sigmoid: top-10 by logit == top-10 by sigmoid(logit)
    // Eliminates 502 expf() calls per token per MoE layer!
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + QwenOptimConfig::ACTIVE_EXPERTS,
        candidates.end(),
        [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second; // Deterministic index tie-breaking
        }
    );

    float sum = 0.0f;
    for (int64_t k = 0; k < QwenOptimConfig::ACTIVE_EXPERTS; ++k) {
        selected_experts_out[k] = candidates[k].second;
        // Compute sigmoid only for the chosen top-10 experts
        float sig_weight = fast_sigmoid(candidates[k].first);
        weights_out[k] = sig_weight;
        sum += sig_weight;
    }

    if (sum > 0.0f) {
        const float inv_sum = 1.0f / sum;
        for (int64_t k = 0; k < QwenOptimConfig::ACTIVE_EXPERTS; ++k) {
            weights_out[k] *= inv_sum;
        }
    }
}

// ------------------- GDN State -------------------

QwenGDNState::QwenGDNState() {
    state_tensor.resize(v_heads * state_dim * state_dim, 0.0f);
}

void QwenGDNState::reset() {
    std::fill(state_tensor.begin(), state_tensor.end(), 0.0f);
}

size_t QwenGDNState::memory_bytes() const {
    return state_tensor.size() * sizeof(float);
}

// ------------------- Gated Residual State -------------------

QwenGatedResidualState::QwenGatedResidualState() {
    streams.resize(n_streams * n_embd, 0.0f);
}

void QwenGatedResidualState::reset() {
    std::fill(streams.begin(), streams.end(), 0.0f);
}

void QwenGatedResidualState::accumulate(const float* layer_delta, const float* gate_weights) {
    for (int64_t s = 0; s < n_streams; ++s) {
        float g = gate_weights ? gate_weights[s] : 1.0f;
        float* stream_ptr = streams.data() + s * n_embd;
        for (int64_t i = 0; i < n_embd; ++i) {
            stream_ptr[i] += g * layer_delta[i];
        }
    }
}

// ------------------- Qwen Execution Engine -------------------

void QwenExecutionEngine::gdn_step(QwenGDNState& gdn_state,
                                  const float* q,
                                  const float* k,
                                  const float* v,
                                  const float* gate,
                                  const float* beta,
                                  float* output,
                                  float scale) {
    const int64_t S = gdn_state.state_dim;
    const int64_t h_v = gdn_state.v_heads;
    const int64_t h_k = gdn_state.k_heads;
    const int64_t group_ratio = h_v / h_k;

    for (int64_t v_head = 0; v_head < h_v; ++v_head) {
        const int64_t k_head = v_head / group_ratio;
        const float g_val = std::exp(gate[v_head]);
        const float beta_val = beta[v_head];

        const float* q_head_ptr = q + k_head * S;
        const float* k_head_ptr = k + k_head * S;
        const float* v_head_ptr = v + v_head * S;
        float* out_head_ptr = output + v_head * S;
        float* s_matrix = gdn_state.state_tensor.data() + v_head * S * S;

        // Step 1: Compute kv_col = S_{t-1} * k
        for (int64_t col = 0; col < S; ++col) {
            float kv_col = 0.0f;
            for (int64_t row = 0; row < S; ++row) {
                kv_col += s_matrix[row * S + col] * k_head_ptr[row];
            }

            // Step 2: Compute delta_col = (v - g * (S * k)) * beta
            float delta_col = (v_head_ptr[col] - g_val * kv_col) * beta_val;

            // Step 3: Update state matrix S_t = g * S_{t-1} + k (x) delta
            float attn_col = 0.0f;
            for (int64_t row = 0; row < S; ++row) {
                s_matrix[row * S + col] = g_val * s_matrix[row * S + col] + k_head_ptr[row] * delta_col;
                attn_col += s_matrix[row * S + col] * q_head_ptr[row];
            }

            // Step 4: Attention output
            out_head_ptr[col] = attn_col * scale;
        }
    }
}

void QwenExecutionEngine::fused_gated_residual(QwenGatedResidualState& gr_state,
                                               const float* layer_out,
                                               const float* residual_gate,
                                               float* merged_out) {
    const int64_t streams = gr_state.n_streams;
    const int64_t dim = gr_state.n_embd;

    // Accumulate into all 4 residual streams
    gr_state.accumulate(layer_out, residual_gate);

    // Fuse 4 streams into output projection
    std::fill(merged_out, merged_out + dim, 0.0f);
    for (int64_t s = 0; s < streams; ++s) {
        const float* stream_ptr = gr_state.streams.data() + s * dim;
        for (int64_t i = 0; i < dim; ++i) {
            merged_out[i] += stream_ptr[i];
        }
    }
}

QwenExecutionEngine::MemoryBreakdown
QwenExecutionEngine::calculate_memory_footprint(int64_t sequence_length, int64_t window_size) {
    MemoryBreakdown mb;

    // Qwen3.8-Flash-Next has 48 layers: 36 GDN layers, 12 QSA layers
    constexpr int64_t N_GDN_LAYERS = 36;
    constexpr int64_t N_QSA_LAYERS = 12;

    // 1. GDN Constant Recurrence State (zero growth with context length)
    // 36 layers * 48 v_heads * 128 * 128 * sizeof(float)
    mb.gdn_fixed_bytes = N_GDN_LAYERS * QwenOptimConfig::SSM_V_HEADS *
                         QwenOptimConfig::SSM_STATE_SIZE * QwenOptimConfig::SSM_STATE_SIZE * sizeof(float);

    // 2. QSA Attention Layers:
    // 12 layers * 2 (K+V) * 2 KV heads * 256 head_dim * sizeof(fp16) = 24,576 bytes/token
    const size_t bytes_per_token_qsa = N_QSA_LAYERS * 2 * 2 * 256 * sizeof(uint16_t);

    mb.qsa_unbounded_kv_bytes = bytes_per_token_qsa * sequence_length;

    int64_t effective_qsa_tokens = std::min(sequence_length, window_size);
    mb.qsa_swa_bounded_kv_bytes = bytes_per_token_qsa * effective_qsa_tokens;

    size_t total_traditional = mb.gdn_fixed_bytes + mb.qsa_unbounded_kv_bytes;
    size_t total_optimized = mb.gdn_fixed_bytes + mb.qsa_swa_bounded_kv_bytes;

    mb.memory_saving_ratio = (total_optimized > 0) ? (static_cast<double>(total_traditional) / total_optimized) : 1.0;
    return mb;
}

} // namespace strata::models::qwen
