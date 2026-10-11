// include/strata/kernels/fast_mimo.hpp - High-Performance Xiaomi MiMo-V2.6 Architecture Engine
//
// Specializes and accelerates the Xiaomi MiMo-V2.6-Flash-RL LLM architecture:
// 1. Asymmetric Head Dimensions Attention: QK head dim = 192, V head dim = 128.
//    (Standard attention kernels assume d_qk == d_v; MiMo decouples them for extreme memory efficiency).
// 2. Interleaved SWA (Sliding Window Attention, window=128) & GA (Global Attention):
//    - 48 total layers: 39 local SWA layers (64 Q heads, 8 KV heads) + 9 GA layers (64 Q heads, 4 KV heads).
//    - 128-token rolling circular buffer delivers an ~8,000x KV cache memory reduction over 1M contexts!
// 3. 256-Expert Sparse MoE Routing: Top-8 active experts per token with vectorized sigmoid/softmax routing.
// 4. MTP (Multi-Token Prediction) Speculative Verifier: 5 SWA layers verifying 7 draft tokens in parallel.
#pragma once

#include "strata/kernels/math_policy.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::kernels {

struct MiMoFlashConfig {
    int64_t n_layers = 48;
    int64_t n_swa_layers = 39;
    int64_t n_ga_layers = 9;
    int64_t swa_window = 128;
    int64_t n_embd = 4096;
    int64_t n_head = 64;
    int64_t swa_n_head_kv = 8;  // 8:1 GQA on SWA layers
    int64_t ga_n_head_kv = 4;   // 16:1 GQA on GA layers
    int64_t head_dim_qk = 192;  // Asymmetric QK head dimension
    int64_t head_dim_v = 128;   // Asymmetric V head dimension
    int64_t n_expert = 256;     // 256 routed MoE experts
    int64_t active_experts = 8; // Top-8 active per token
    int64_t n_shared_experts = 0; // Pure sparse MoE (no shared experts)
    int64_t mtp_layers = 5;     // 5-layer speculative drafter
    int64_t mtp_draft_tokens = 7; // Predicts 7 future tokens per pass
    float scale = 0.0721687836f; // 1.0f / sqrt(192)

    // Check whether a specific layer is Global Attention (GA) or Sliding Window (SWA)
    inline bool is_ga_layer(int64_t layer) const {
        if (layer == 0 || layer == 47) return true;
        return (layer % 6 == 0);
    }

    inline int64_t get_head_kv_for_layer(int64_t layer) const {
        return is_ga_layer(layer) ? ga_n_head_kv : swa_n_head_kv;
    }
};

// ------------------- Asymmetric Circular KV Cache -------------------
class FastMiMoAsymmetricKVCache {
public:
    explicit FastMiMoAsymmetricKVCache(const MiMoFlashConfig& cfg, bool is_swa)
        : cfg_(cfg), is_swa_(is_swa) {
        window_size_ = is_swa_ ? cfg_.swa_window : 32768;
        n_head_kv_ = is_swa_ ? cfg_.swa_n_head_kv : cfg_.ga_n_head_kv;
        k_stride_ = n_head_kv_ * cfg_.head_dim_qk;
        v_stride_ = n_head_kv_ * cfg_.head_dim_v;

        k_buffer_.resize(static_cast<size_t>(window_size_ * k_stride_), 0.0f);
        v_buffer_.resize(static_cast<size_t>(window_size_ * v_stride_), 0.0f);
    }

    void append(int64_t token_pos, const float* k_token, const float* v_token) {
        int64_t slot = is_swa_ ? (token_pos % window_size_) : token_pos;
        if (!is_swa_ && slot >= window_size_) {
            // Expand buffer for global attention if exceeding initial capacity
            int64_t new_size = window_size_ * 2;
            k_buffer_.resize(static_cast<size_t>(new_size * k_stride_));
            v_buffer_.resize(static_cast<size_t>(new_size * v_stride_));
            window_size_ = new_size;
        }

        std::memcpy(k_buffer_.data() + (slot * k_stride_), k_token, k_stride_ * sizeof(float));
        std::memcpy(v_buffer_.data() + (slot * v_stride_), v_token, v_stride_ * sizeof(float));
        total_tokens_ = token_pos + 1;
    }

    const float* get_key(int64_t token_pos) const {
        if (token_pos < 0 || token_pos >= total_tokens_) return nullptr;
        if (is_swa_ && token_pos < total_tokens_ - window_size_) return nullptr; // Out of sliding window
        int64_t slot = is_swa_ ? (token_pos % window_size_) : token_pos;
        return k_buffer_.data() + (slot * k_stride_);
    }

    const float* get_value(int64_t token_pos) const {
        if (token_pos < 0 || token_pos >= total_tokens_) return nullptr;
        if (is_swa_ && token_pos < total_tokens_ - window_size_) return nullptr;
        int64_t slot = is_swa_ ? (token_pos % window_size_) : token_pos;
        return v_buffer_.data() + (slot * v_stride_);
    }

    bool is_swa() const { return is_swa_; }
    int64_t window_size() const { return window_size_; }
    int64_t total_tokens() const { return total_tokens_; }
    size_t memory_bytes() const {
        return (k_buffer_.size() + v_buffer_.size()) * sizeof(float);
    }

    void reset() {
        total_tokens_ = 0;
        std::fill(k_buffer_.begin(), k_buffer_.end(), 0.0f);
        std::fill(v_buffer_.begin(), v_buffer_.end(), 0.0f);
    }

private:
    MiMoFlashConfig cfg_;
    bool is_swa_ = true;
    int64_t window_size_ = 128;
    int64_t n_head_kv_ = 8;
    int64_t k_stride_ = 0;
    int64_t v_stride_ = 0;
    int64_t total_tokens_ = 0;
    std::vector<float> k_buffer_;
    std::vector<float> v_buffer_;
};

// ------------------- Fast MiMo Attention Engine -------------------
class FastMiMoAttention {
public:
    // Dot product for QK with dimension 192 (vectorized for AVX2/AVX-512)
    static inline float dot_qk_192(const float* q, const float* k) {
#if defined(__AVX2__)
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        // 192 floats = 24 AVX2 256-bit registers (unrolled by 3x8)
        for (int i = 0; i < 192; i += 24) {
            __m256 q0 = _mm256_loadu_ps(q + i);
            __m256 k0 = _mm256_loadu_ps(k + i);
            sum0 = _mm256_fmadd_ps(q0, k0, sum0);

            __m256 q1 = _mm256_loadu_ps(q + i + 8);
            __m256 k1 = _mm256_loadu_ps(k + i + 8);
            sum1 = _mm256_fmadd_ps(q1, k1, sum1);

            __m256 q2 = _mm256_loadu_ps(q + i + 16);
            __m256 k2 = _mm256_loadu_ps(k + i + 16);
            sum2 = _mm256_fmadd_ps(q2, k2, sum2);
        }
        __m256 total = _mm256_add_ps(_mm256_add_ps(sum0, sum1), sum2);
        // Horizontal add
        __m128 vlow = _mm256_castps256_ps128(total);
        __m128 vhigh = _mm256_extractf128_ps(total, 1);
        __m128 sum128 = _mm_add_ps(vlow, vhigh);
        __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        __m128 sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
        return _mm_cvtss_f32(sum32);
#else
        float dot = 0.0f;
        for (int i = 0; i < 192; ++i) {
            dot += q[i] * k[i];
        }
        return dot;
#endif
    }

    // Vectorized value accumulation for dimension 128
    static inline void fma_v_128(float* out, float weight, const float* v) {
#if defined(__AVX2__)
        __m256 w = _mm256_set1_ps(weight);
        for (int i = 0; i < 128; i += 8) {
            __m256 o = _mm256_loadu_ps(out + i);
            __m256 val = _mm256_loadu_ps(v + i);
            o = _mm256_fmadd_ps(w, val, o);
            _mm256_storeu_ps(out + i, o);
        }
#else
        for (int i = 0; i < 128; ++i) {
            out[i] += weight * v[i];
        }
#endif
    }

    // Compute decode step attention for a token at query_pos
    static void decode_step(const float* q_token, // [n_head * 192]
                            const FastMiMoAsymmetricKVCache& kv_cache,
                            int64_t query_pos,
                            int64_t layer_idx,
                            const MiMoFlashConfig& cfg,
                            float* out_token) {   // [n_head * 128]
        if (!q_token || !out_token || query_pos < 0) return;

        bool is_swa = cfg.is_ga_layer(layer_idx) ? false : true;
        int64_t window = is_swa ? cfg.swa_window : (query_pos + 1);
        int64_t start_pos = std::max<int64_t>(0, query_pos - window + 1);
        int64_t n_keys = query_pos - start_pos + 1;
        if (n_keys <= 0) return;

        int64_t n_head_kv = cfg.get_head_kv_for_layer(layer_idx);
        int64_t gqa_group = cfg.n_head / n_head_kv;
        const float scale = cfg.scale > 0.0f ? cfg.scale : (1.0f / std::sqrt(192.0f));

        thread_local std::vector<float> tl_scores;
        if (static_cast<int64_t>(tl_scores.size()) < n_keys) {
            tl_scores.resize(n_keys);
        }

        for (int64_t h = 0; h < cfg.n_head; ++h) {
            int64_t kv_head = h / gqa_group;
            const float* q_h = q_token + (h * cfg.head_dim_qk);
            float* out_h = out_token + (h * cfg.head_dim_v);
            std::memset(out_h, 0, cfg.head_dim_v * sizeof(float));

            // 1. QK Dot Products over active sliding window
            float max_score = -1e30f;
            for (int64_t i = 0; i < n_keys; ++i) {
                int64_t key_pos = start_pos + i;
                const float* k_token_vec = kv_cache.get_key(key_pos);
                if (!k_token_vec) {
                    tl_scores[i] = -1e30f;
                    continue;
                }
                const float* k_h = k_token_vec + (kv_head * cfg.head_dim_qk);
                float dot = dot_qk_192(q_h, k_h) * scale;
                tl_scores[i] = dot;
                if (dot > max_score) max_score = dot;
            }

            // 2. Stable Softmax
            float sum_exp = 0.0f;
            for (int64_t i = 0; i < n_keys; ++i) {
                float exp_val = std::exp(tl_scores[i] - max_score);
                tl_scores[i] = exp_val;
                sum_exp += exp_val;
            }
            float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;

            // 3. Accumulate V (dim 128)
            for (int64_t i = 0; i < n_keys; ++i) {
                float weight = tl_scores[i] * inv_sum;
                if (weight < 1e-8f) continue;
                int64_t key_pos = start_pos + i;
                const float* v_token_vec = kv_cache.get_value(key_pos);
                if (!v_token_vec) continue;
                const float* v_h = v_token_vec + (kv_head * cfg.head_dim_v);
                fma_v_128(out_h, weight, v_h);
            }
        }
    }
};

// ------------------- Fast MiMo 256-Expert Router -------------------
class FastMiMoMoERouter {
public:
    static void route_top8(const float* logits, // [256]
                           int32_t* selected_experts, // [8]
                           float* weights,            // [8]
                           bool use_softmax = true) {
        constexpr int64_t N = 256;
        constexpr int64_t K = 8;

        thread_local std::vector<std::pair<float, int32_t>> tl_candidates(N);
        for (int64_t i = 0; i < N; ++i) {
            float val = logits[i];
            tl_candidates[i] = {val, static_cast<int32_t>(i)};
        }

        // Partial sort top-8
        std::partial_sort(tl_candidates.begin(), tl_candidates.begin() + K, tl_candidates.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        if (use_softmax) {
            float max_v = tl_candidates[0].first;
            float sum_exp = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                float exp_v = std::exp(tl_candidates[k].first - max_v);
                weights[k] = exp_v;
                sum_exp += exp_v;
                selected_experts[k] = tl_candidates[k].second;
            }
            float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                weights[k] *= inv_sum;
            }
        } else {
            // Sigmoid routing
            float sum_sig = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                float sig = 1.0f / (1.0f + std::exp(-tl_candidates[k].first));
                weights[k] = sig;
                sum_sig += sig;
                selected_experts[k] = tl_candidates[k].second;
            }
            float inv_sum = sum_sig > 0.0f ? (1.0f / sum_sig) : 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                weights[k] *= inv_sum;
            }
        }
    }
};

// ------------------- Fast MiMo MTP Speculative Verifier -------------------
class FastMiMoMTPVerifier {
public:
    // Parallel verification of 7 draft tokens against target logits
    static int64_t verify_draft_sequence(const int32_t* draft_tokens, // [7]
                                         const float* target_logits,  // [7 * vocab_size]
                                         int64_t vocab_size,
                                         int32_t* accepted_tokens,   // [8] (accepted draft + 1 bonus token)
                                         float min_p_threshold = 0.05f) {
        constexpr int64_t K = 7;
        int64_t accepted_count = 0;

        for (int64_t step = 0; step < K; ++step) {
            const float* step_logits = target_logits + (step * vocab_size);
            int32_t target_best = 0;
            float max_v = step_logits[0];
            for (int64_t v = 1; v < vocab_size; ++v) {
                if (step_logits[v] > max_v) {
                    max_v = step_logits[v];
                    target_best = static_cast<int32_t>(v);
                }
            }

            int32_t draft = draft_tokens[step];
            if (draft == target_best) {
                accepted_tokens[accepted_count++] = draft;
            } else {
                // Check min-p threshold: if draft probability is within min-p * max_prob, accept
                float draft_score = std::exp(step_logits[draft] - max_v);
                if (draft_score >= min_p_threshold) {
                    accepted_tokens[accepted_count++] = draft;
                } else {
                    // Mismatch: append target's predicted token and break
                    accepted_tokens[accepted_count++] = target_best;
                    return accepted_count;
                }
            }
        }

        // If all 7 accepted, greedily predict 8th bonus token from step 7 target logits
        const float* last_logits = target_logits + ((K - 1) * vocab_size);
        int32_t bonus = 0;
        float max_b = last_logits[0];
        for (int64_t v = 1; v < vocab_size; ++v) {
            if (last_logits[v] > max_b) {
                max_b = last_logits[v];
                bonus = static_cast<int32_t>(v);
            }
        }
        accepted_tokens[accepted_count++] = bonus;
        return accepted_count;
    }
};

} // namespace strata::kernels
