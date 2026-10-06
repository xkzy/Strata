// include/strata/kernels/fast_attention.hpp
#pragma once

#include "strata/kernels/math_policy.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::kernels {

class FastAttention {
private:
#if defined(__AVX2__) || defined(__AVX__) || defined(_MSC_VER)
    static inline float hsum256_ps(__m256 v) {
        __m128 vlow = _mm256_castps256_ps128(v);
        __m128 vhigh = _mm256_extractf128_ps(v, 1);
        __m128 sum128 = _mm_add_ps(vlow, vhigh);
        __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        __m128 sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
        return _mm_cvtss_f32(sum32);
    }
#endif

#if defined(__AVX512F__)
    static inline float hsum512_ps(__m512 v) {
        return _mm512_reduce_add_ps(v);
    }
#endif

    static inline float dot_product(const float* a, const float* b, size_t dim) {
        size_t d = 0;
#if defined(__AVX512F__)
        __m512 acc0 = _mm512_setzero_ps();
        for (; d + 15 < dim; d += 16) {
            __m512 va = _mm512_loadu_ps(a + d);
            __m512 vb = _mm512_loadu_ps(b + d);
            acc0 = _mm512_fmadd_ps(va, vb, acc0);
        }
        float dot = hsum512_ps(acc0);
#elif defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
        __m256 acc0 = _mm256_setzero_ps();
        for (; d + 7 < dim; d += 8) {
            __m256 va = _mm256_loadu_ps(a + d);
            __m256 vb = _mm256_loadu_ps(b + d);
            acc0 = _mm256_fmadd_ps(va, vb, acc0);
        }
        float dot = hsum256_ps(acc0);
#else
        float dot = 0.0f;
#endif
        for (; d < dim; ++d) {
            dot += a[d] * b[d];
        }
        return dot;
    }

    static inline void accumulate_weighted(float* out, const float* v, float weight, size_t dim) {
        if (weight == 0.0f) return;
        size_t d = 0;
#if defined(__AVX512F__)
        __m512 w_v = _mm512_set1_ps(weight);
        for (; d + 15 < dim; d += 16) {
            __m512 vo = _mm512_loadu_ps(out + d);
            __m512 vv = _mm512_loadu_ps(v + d);
            _mm512_storeu_ps(out + d, _mm512_fmadd_ps(w_v, vv, vo));
        }
#elif defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
        __m256 w_v = _mm256_set1_ps(weight);
        for (; d + 7 < dim; d += 8) {
            __m256 vo = _mm256_loadu_ps(out + d);
            __m256 vv = _mm256_loadu_ps(v + d);
            _mm256_storeu_ps(out + d, _mm256_fmadd_ps(w_v, vv, vo));
        }
#endif
        for (; d < dim; ++d) {
            out[d] += weight * v[d];
        }
    }

public:
    // ------------------------------------------------------------------------
    // Single-Head Incremental Attention Decode:
    // Computes dot-product attention of single query vector q [head_dim]
    // against key cache k_cache [seq_len, head_dim], applies causal scaling,
    // bounds sanitization, optional additive masking, softmax, and aggregates
    // v_cache [seq_len, head_dim] into out [head_dim].
    // ------------------------------------------------------------------------
    static void scaled_dot_product_decode(
        const float* __restrict__ q,
        const float* __restrict__ k_cache,
        const float* __restrict__ v_cache,
        float* __restrict__ out,
        size_t seq_len,
        size_t head_dim,
        float scale,
        const MathContext& ctx = MathContext()) {
        scaled_dot_product_decode(q, k_cache, v_cache, out, seq_len, head_dim, scale, nullptr, ctx);
    }

    static void scaled_dot_product_decode(
        const float* __restrict__ q,
        const float* __restrict__ k_cache,
        const float* __restrict__ v_cache,
        float* __restrict__ out,
        size_t seq_len,
        size_t head_dim,
        float scale,
        const float* __restrict__ mask,
        const MathContext& ctx = MathContext()) {
        if (!q || !k_cache || !v_cache || !out || seq_len == 0 || head_dim == 0) return;

        std::vector<float> scores(seq_len);
        float max_score = -1e30f;

        for (size_t s = 0; s < seq_len; ++s) {
            const float* k_row = k_cache + s * head_dim;
            float dot = dot_product(q, k_row, head_dim);
            float score = sanitize_logit(dot * scale, ctx);
            if (mask) {
                score += mask[s];
            }
            scores[s] = score;
            if (score > max_score) max_score = score;
        }

        float sum_exp = 0.0f;
        for (size_t s = 0; s < seq_len; ++s) {
            scores[s] = std::exp(scores[s] - max_score);
            sum_exp += scores[s];
        }
        const float inv_sum = sum_exp > 0.0f ? (1.0f / (sum_exp + 1e-9f)) : 0.0f;
        for (size_t s = 0; s < seq_len; ++s) {
            scores[s] *= inv_sum;
        }

        std::fill(out, out + head_dim, 0.0f);
        for (size_t s = 0; s < seq_len; ++s) {
            const float weight = scores[s];
            const float* v_row = v_cache + s * head_dim;
            accumulate_weighted(out, v_row, weight, head_dim);
        }
    }

    // Alias for design-specification naming parity
    static inline void scaled_dot_product_attention_decode(
        const float* __restrict__ q,
        const float* __restrict__ k_cache,
        const float* __restrict__ v_cache,
        float* __restrict__ out,
        size_t seq_len,
        size_t head_dim,
        float scale,
        const MathContext& ctx = MathContext()) {
        scaled_dot_product_decode(q, k_cache, v_cache, out, seq_len, head_dim, scale, nullptr, ctx);
    }

    // ------------------------------------------------------------------------
    // Multi-Head & Grouped Query Attention (GQA / MQA / MHA) Incremental Decode
    // ------------------------------------------------------------------------
    static void multi_head_scaled_dot_product_decode(
        const float* __restrict__ q,
        const float* __restrict__ k_cache,
        const float* __restrict__ v_cache,
        float* __restrict__ out,
        size_t num_heads,
        size_t num_kv_heads,
        size_t seq_len,
        size_t head_dim,
        float scale,
        const MathContext& ctx = MathContext()) {
        multi_head_scaled_dot_product_decode(q, k_cache, v_cache, out, num_heads, num_kv_heads, seq_len, head_dim, scale, nullptr, ctx);
    }

    static void multi_head_scaled_dot_product_decode(
        const float* __restrict__ q,
        const float* __restrict__ k_cache,
        const float* __restrict__ v_cache,
        float* __restrict__ out,
        size_t num_heads,
        size_t num_kv_heads,
        size_t seq_len,
        size_t head_dim,
        float scale,
        const float* __restrict__ mask,
        const MathContext& ctx = MathContext()) {
        if (!q || !k_cache || !v_cache || !out || num_heads == 0 || num_kv_heads == 0 || seq_len == 0 || head_dim == 0) return;

        const size_t gqa_group_size = num_heads / num_kv_heads;
        std::vector<float> scores(seq_len);

        for (size_t h = 0; h < num_heads; ++h) {
            const size_t kv_head = (gqa_group_size > 0) ? std::min(h / gqa_group_size, num_kv_heads - 1) : 0;
            const float* q_h = q + h * head_dim;
            float* out_h = out + h * head_dim;

            float max_score = -1e30f;

            for (size_t s = 0; s < seq_len; ++s) {
                const float* k_row = k_cache + (s * num_kv_heads + kv_head) * head_dim;
                float dot = dot_product(q_h, k_row, head_dim);
                float score = sanitize_logit(dot * scale, ctx);
                if (mask) {
                    score += mask[s];
                }
                scores[s] = score;
                if (score > max_score) max_score = score;
            }

            float sum_exp = 0.0f;
            for (size_t s = 0; s < seq_len; ++s) {
                scores[s] = std::exp(scores[s] - max_score);
                sum_exp += scores[s];
            }
            const float inv_sum = sum_exp > 0.0f ? (1.0f / (sum_exp + 1e-9f)) : 0.0f;
            for (size_t s = 0; s < seq_len; ++s) {
                scores[s] *= inv_sum;
            }

            std::fill(out_h, out_h + head_dim, 0.0f);
            for (size_t s = 0; s < seq_len; ++s) {
                const float weight = scores[s];
                const float* v_row = v_cache + (s * num_kv_heads + kv_head) * head_dim;
                accumulate_weighted(out_h, v_row, weight, head_dim);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Prefill Attention Reference Pipeline
    // ------------------------------------------------------------------------
    static void flash_attention_prefill(
        const float* __restrict__ q,
        const float* __restrict__ k,
        const float* __restrict__ v,
        float* __restrict__ out,
        size_t seq_len,
        size_t head_dim,
        float scale,
        bool causal = true,
        const MathContext& ctx = MathContext()) {
        if (!q || !k || !v || !out || seq_len == 0 || head_dim == 0) return;

        for (size_t i = 0; i < seq_len; ++i) {
            const float* q_i = q + i * head_dim;
            float* out_i = out + i * head_dim;
            size_t active_keys = causal ? (i + 1) : seq_len;

            scaled_dot_product_decode(
                q_i, k, v, out_i,
                active_keys, head_dim, scale, nullptr, ctx
            );
        }
    }
};

} // namespace strata::kernels
