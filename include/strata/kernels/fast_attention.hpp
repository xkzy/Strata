// include/strata/kernels/fast_attention.hpp
#pragma once

#include "strata/kernels/math_policy.hpp"
#include "strata/kernels/fast_parallel.hpp"
#include "strata/kernels/fast_paged_kv.hpp"
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

    static inline void scale_vector(float* v, float factor, size_t dim) {
        if (!v || dim == 0) return;
        size_t d = 0;
#if defined(__AVX512F__)
        __m512 f = _mm512_set1_ps(factor);
        for (; d + 15 < dim; d += 16) {
            __m512 vo = _mm512_loadu_ps(v + d);
            _mm512_storeu_ps(v + d, _mm512_mul_ps(vo, f));
        }
#elif defined(__AVX2__)
        __m256 f = _mm256_set1_ps(factor);
        for (; d + 7 < dim; d += 8) {
            __m256 vo = _mm256_loadu_ps(v + d);
            _mm256_storeu_ps(v + d, _mm256_mul_ps(vo, f));
        }
#endif
        for (; d < dim; ++d) v[d] *= factor;
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

        float local_scores[1024];
        std::vector<float> heap_scores;
        float* scores = (seq_len <= 1024) ? local_scores : (heap_scores.resize(seq_len), heap_scores.data());
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
    // Tiled Online Flash Attention Decoding (1-Pass Zero-Alloc):
    // Computes single query vector q [head_dim] against KV cache [seq_len, head_dim]
    // using running log-sum-exp in a single pass over KV memory blocks without
    // allocating or storing intermediate score buffers.
    // ------------------------------------------------------------------------
    static void flash_decode_scaled_dot_product(
        const float* __restrict__ q,
        const float* __restrict__ k_cache,
        const float* __restrict__ v_cache,
        float* __restrict__ out,
        size_t seq_len,
        size_t head_dim,
        float scale,
        const float* __restrict__ mask = nullptr,
        const MathContext& ctx = MathContext()) {
        if (!q || !k_cache || !v_cache || !out || seq_len == 0 || head_dim == 0) return;

        std::fill(out, out + head_dim, 0.0f);
        float m_i = -1e30f;
        float l_i = 0.0f;

        constexpr size_t TILE_SIZE = 32;
        alignas(32) float tile_scores[TILE_SIZE];

        for (size_t t = 0; t < seq_len; t += TILE_SIZE) {
            const size_t cur_tile = std::min(TILE_SIZE, seq_len - t);
            float tile_max = -1e30f;

            for (size_t i = 0; i < cur_tile; ++i) {
                const size_t s = t + i;
                const float* k_row = k_cache + s * head_dim;
                float dot = dot_product(q, k_row, head_dim);
                float sc = sanitize_logit(dot * scale, ctx);
                if (mask) sc += mask[s];
                tile_scores[i] = sc;
                if (sc > tile_max) tile_max = sc;
            }

            const float m_new = std::max(m_i, tile_max);
            const float alpha = (m_i > -1e20f) ? std::exp(m_i - m_new) : 0.0f;

            // Rescale accumulator
            if (alpha != 1.0f && l_i > 0.0f) {
                scale_vector(out, alpha, head_dim);
                l_i *= alpha;
            }

            // Accumulate tile weights and values
            for (size_t i = 0; i < cur_tile; ++i) {
                const size_t s = t + i;
                const float weight = std::exp(tile_scores[i] - m_new);
                l_i += weight;
                const float* v_row = v_cache + s * head_dim;
                accumulate_weighted(out, v_row, weight, head_dim);
            }

            m_i = m_new;
        }

        if (l_i > 0.0f) {
            scale_vector(out, 1.0f / l_i, head_dim);
        }
    }

    // ------------------------------------------------------------------------
    // Grouped-Query Flash Attention Decoding (GQA / MQA / MHA):
    // Multi-head single-pass tiled online Flash Decoding across all query heads.
    // ------------------------------------------------------------------------
    static void flash_decode_gqa(
        const float* __restrict__ q,
        const float* __restrict__ k_cache,
        const float* __restrict__ v_cache,
        float* __restrict__ out,
        size_t num_heads,
        size_t num_kv_heads,
        size_t seq_len,
        size_t head_dim,
        float scale,
        const float* __restrict__ mask = nullptr,
        const MathContext& ctx = MathContext()) {
        if (!q || !k_cache || !v_cache || !out || num_heads == 0 || num_kv_heads == 0 || seq_len == 0 || head_dim == 0) return;

        const size_t gqa_group_size = num_heads / num_kv_heads;

        #pragma omp parallel for schedule(static) if (num_heads >= 4 && seq_len >= 32)
        for (size_t h = 0; h < num_heads; ++h) {
            const size_t kv_head = (gqa_group_size > 0) ? std::min(h / gqa_group_size, num_kv_heads - 1) : 0;
            const float* q_h = q + h * head_dim;
            float* out_h = out + h * head_dim;
            std::fill(out_h, out_h + head_dim, 0.0f);

            float m_i = -1e30f;
            float l_i = 0.0f;

            constexpr size_t TILE_SIZE = 32;
            alignas(32) float tile_scores[TILE_SIZE];

            for (size_t t = 0; t < seq_len; t += TILE_SIZE) {
                const size_t cur_tile = std::min(TILE_SIZE, seq_len - t);
                float tile_max = -1e30f;

                for (size_t i = 0; i < cur_tile; ++i) {
                    const size_t s = t + i;
                    const float* k_row = k_cache + (s * num_kv_heads + kv_head) * head_dim;
                    float dot = dot_product(q_h, k_row, head_dim);
                    float sc = sanitize_logit(dot * scale, ctx);
                    if (mask) sc += mask[s];
                    tile_scores[i] = sc;
                    if (sc > tile_max) tile_max = sc;
                }

                const float m_new = std::max(m_i, tile_max);
                const float alpha = (m_i > -1e20f) ? std::exp(m_i - m_new) : 0.0f;

                if (alpha != 1.0f && l_i > 0.0f) {
                    scale_vector(out_h, alpha, head_dim);
                    l_i *= alpha;
                }

                for (size_t i = 0; i < cur_tile; ++i) {
                    const size_t s = t + i;
                    const float weight = std::exp(tile_scores[i] - m_new);
                    l_i += weight;
                    const float* v_row = v_cache + (s * num_kv_heads + kv_head) * head_dim;
                    accumulate_weighted(out_h, v_row, weight, head_dim);
                }

                m_i = m_new;
            }

            if (l_i > 0.0f) {
                scale_vector(out_h, 1.0f / l_i, head_dim);
            }
        }
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

        #pragma omp parallel for schedule(static) if (num_heads >= 4 && seq_len >= 32)
        for (size_t h = 0; h < num_heads; ++h) {
            const size_t kv_head = (gqa_group_size > 0) ? std::min(h / gqa_group_size, num_kv_heads - 1) : 0;
            const float* q_h = q + h * head_dim;
            float* out_h = out + h * head_dim;

            float local_scores[1024];
            std::vector<float> heap_scores;
            float* scores = (seq_len <= 1024) ? local_scores : (heap_scores.resize(seq_len), heap_scores.data());

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
    // Prefill Attention: Tiled Online-Softmax FlashAttention-2 Engine
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

        // Block tiling configuration (tuning for L1/L2 cache locality)
        constexpr size_t Br = 64; // Query block size
        constexpr size_t Bc = 64; // Key/Value block size

        const size_t num_q_blocks = (seq_len + Br - 1) / Br;
        const size_t num_kv_blocks = (seq_len + Bc - 1) / Bc;

        #pragma omp parallel for schedule(dynamic) if (num_q_blocks >= fast_parallel::kFlashPrefillMinQBlocks)
        for (size_t q_blk = 0; q_blk < num_q_blocks; ++q_blk) {
            const size_t r_start = q_blk * Br;
            const size_t r_end = std::min(seq_len, r_start + Br);
            const size_t current_br = r_end - r_start;

            // Zero-allocation stack buffers for block state
            float stack_m[Br];
            float stack_l[Br];
            float stack_acc[Br * 256];
            float stack_tile_scores[Bc];

            std::vector<float> heap_acc;
            float* acc = (head_dim <= 256)
                ? stack_acc
                : (heap_acc.resize(current_br * head_dim, 0.0f), heap_acc.data());

            for (size_t i = 0; i < current_br; ++i) {
                stack_m[i] = -1e30f;
                stack_l[i] = 0.0f;
            }
            if (head_dim <= 256) {
                std::fill(stack_acc, stack_acc + current_br * head_dim, 0.0f);
            }

            for (size_t kv_blk = 0; kv_blk < num_kv_blocks; ++kv_blk) {
                const size_t c_start = kv_blk * Bc;
                const size_t c_end = std::min(seq_len, c_start + Bc);

                // Causal pruning: if lowest key pos in block > highest query pos in block, skip entirely
                if (causal && c_start >= r_end) {
                    break;
                }

                for (size_t r = r_start; r < r_end; ++r) {
                    const size_t local_r = r - r_start;
                    const float* q_row = q + r * head_dim;
                    float* acc_row = acc + local_r * head_dim;

                    const size_t active_c_end = causal ? std::min(c_end, r + 1) : c_end;
                    if (c_start >= active_c_end) continue;

                    const size_t active_len = active_c_end - c_start;

                    // 1. Compute QK dot products for this row and tile
                    float tile_max = -1e30f;
                    for (size_t c_idx = 0; c_idx < active_len; ++c_idx) {
                        const size_t c = c_start + c_idx;
                        const float* k_row = k + c * head_dim;
                        float dot = dot_product(q_row, k_row, head_dim);
                        float score = sanitize_logit(dot * scale, ctx);
                        stack_tile_scores[c_idx] = score;
                        if (score > tile_max) tile_max = score;
                    }

                    // 2. Online softmax update
                    float m_prev = stack_m[local_r];
                    float m_new = std::max(m_prev, tile_max);
                    float alpha = std::exp(m_prev - m_new);

                    // Rescale previous accumulator and sum-of-exp
                    for (size_t d = 0; d < head_dim; ++d) {
                        acc_row[d] *= alpha;
                    }
                    float l_new = stack_l[local_r] * alpha;

                    // 3. Accumulate new tile weights into output and sum
                    for (size_t c_idx = 0; c_idx < active_len; ++c_idx) {
                        const size_t c = c_start + c_idx;
                        float p = std::exp(stack_tile_scores[c_idx] - m_new);
                        l_new += p;
                        const float* v_row = v + c * head_dim;
                        accumulate_weighted(acc_row, v_row, p, head_dim);
                    }

                    stack_m[local_r] = m_new;
                    stack_l[local_r] = l_new;
                }
            }

            // Final normalization and writeout to destination
            for (size_t r = r_start; r < r_end; ++r) {
                const size_t local_r = r - r_start;
                const float* acc_row = acc + local_r * head_dim;
                float* out_row = out + r * head_dim;
                const float inv_l = stack_l[local_r] > 0.0f ? (1.0f / (stack_l[local_r] + 1e-9f)) : 0.0f;

                for (size_t d = 0; d < head_dim; ++d) {
                    out_row[d] = acc_row[d] * inv_l;
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // Multi-Head / Grouped-Query Attention (GQA) Prefill Attention
    // ------------------------------------------------------------------------
    static void flash_attention_prefill_mha(
        const float* __restrict__ q,       // [seq_len, num_q_heads, head_dim]
        const float* __restrict__ k,       // [seq_len, num_kv_heads, head_dim]
        const float* __restrict__ v,       // [seq_len, num_kv_heads, head_dim]
        float* __restrict__ out,           // [seq_len, num_q_heads, head_dim]
        size_t seq_len,
        size_t num_q_heads,
        size_t num_kv_heads,
        size_t head_dim,
        float scale,
        bool causal = true,
        const MathContext& ctx = MathContext()) {
        if (!q || !k || !v || !out || seq_len == 0 || head_dim == 0) return;

        const size_t gqa_ratio = num_q_heads / (num_kv_heads > 0 ? num_kv_heads : 1);

        #pragma omp parallel for collapse(2) schedule(dynamic) if (seq_len * seq_len * num_q_heads * head_dim >= fast_parallel::kPrefillMhaMinWork)
        for (size_t h = 0; h < num_q_heads; ++h) {
            for (size_t r = 0; r < seq_len; ++r) {
                const size_t kv_h = (gqa_ratio > 0) ? (h / gqa_ratio) : 0;
                const float* q_row = q + (r * num_q_heads + h) * head_dim;
                float* out_row = out + (r * num_q_heads + h) * head_dim;

                const size_t active_keys = causal ? (r + 1) : seq_len;

                // Fast decode for current row with zero-allocation stack buffer for active_keys <= 1024
                float stack_scores[1024];
                std::vector<float> heap_scores;
                float* scores = (active_keys <= 1024)
                    ? stack_scores
                    : (heap_scores.resize(active_keys), heap_scores.data());
                float max_score = -1e30f;

                for (size_t s = 0; s < active_keys; ++s) {
                    const float* k_row = k + (s * num_kv_heads + kv_h) * head_dim;
                    float dot = dot_product(q_row, k_row, head_dim);
                    float score = sanitize_logit(dot * scale, ctx);
                    scores[s] = score;
                    if (score > max_score) max_score = score;
                }

                float sum_exp = 0.0f;
                for (size_t s = 0; s < active_keys; ++s) {
                    scores[s] = std::exp(scores[s] - max_score);
                    sum_exp += scores[s];
                }
                const float inv_sum = sum_exp > 0.0f ? (1.0f / (sum_exp + 1e-9f)) : 0.0f;

                std::fill(out_row, out_row + head_dim, 0.0f);
                for (size_t s = 0; s < active_keys; ++s) {
                    const float weight = scores[s] * inv_sum;
                    const float* v_row = v + (s * num_kv_heads + kv_h) * head_dim;
                    accumulate_weighted(out_row, v_row, weight, head_dim);
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // Copy-on-Write (COW) Split-KV Incremental Attention Decode (Xenolith Architecture)
    // Seamlessly evaluates attention across an immutable prefix buffer [0, prefix_len)
    // and a mutable tail buffer [prefix_len, prefix_len + tail_len) without memory copies.
    // ------------------------------------------------------------------------
    static void cow_scaled_dot_product_decode(
        const float* __restrict__ q,
        const float* __restrict__ prefix_k,
        const float* __restrict__ prefix_v,
        size_t prefix_len,
        const float* __restrict__ tail_k,
        const float* __restrict__ tail_v,
        size_t tail_len,
        float* __restrict__ out,
        size_t head_dim,
        float scale,
        const MathContext& ctx = MathContext()) {
        if (!q || !out || head_dim == 0) return;
        const size_t total_len = prefix_len + tail_len;
        if (total_len == 0) return;

        float local_scores[1024];
        std::vector<float> heap_scores;
        float* scores = (total_len <= 1024) ? local_scores : (heap_scores.resize(total_len), heap_scores.data());
        float max_score = -1e30f;

        // 1. Process immutable prefix cache
        for (size_t s = 0; s < prefix_len; ++s) {
            const float* k_row = prefix_k + s * head_dim;
            float dot = dot_product(q, k_row, head_dim);
            float score = sanitize_logit(dot * scale, ctx);
            scores[s] = score;
            if (score > max_score) max_score = score;
        }

        // 2. Process mutable tail cache
        for (size_t s = 0; s < tail_len; ++s) {
            const float* k_row = tail_k + s * head_dim;
            float dot = dot_product(q, k_row, head_dim);
            float score = sanitize_logit(dot * scale, ctx);
            scores[prefix_len + s] = score;
            if (score > max_score) max_score = score;
        }

        // 3. Softmax
        float sum_exp = 0.0f;
        for (size_t s = 0; s < total_len; ++s) {
            scores[s] = std::exp(scores[s] - max_score);
            sum_exp += scores[s];
        }
        const float inv_sum = sum_exp > 0.0f ? (1.0f / (sum_exp + 1e-9f)) : 0.0f;
        for (size_t s = 0; s < total_len; ++s) {
            scores[s] *= inv_sum;
        }

        // 4. Output aggregation across split cache
        std::fill(out, out + head_dim, 0.0f);
        for (size_t s = 0; s < prefix_len; ++s) {
            accumulate_weighted(out, prefix_v + s * head_dim, scores[s], head_dim);
        }
        for (size_t s = 0; s < tail_len; ++s) {
            accumulate_weighted(out, tail_v + s * head_dim, scores[prefix_len + s], head_dim);
        }
    }

    // ------------------------------------------------------------------------
    // Copy-on-Write (COW) Multi-Head & GQA Incremental Attention Decode
    // ------------------------------------------------------------------------
    static void cow_multi_head_scaled_dot_product_decode(
        const float* __restrict__ q,
        const float* __restrict__ prefix_k,
        const float* __restrict__ prefix_v,
        size_t prefix_len,
        const float* __restrict__ tail_k,
        const float* __restrict__ tail_v,
        size_t tail_len,
        float* __restrict__ out,
        size_t num_heads,
        size_t num_kv_heads,
        size_t head_dim,
        float scale,
        const MathContext& ctx = MathContext()) {
        if (!q || !out || num_heads == 0 || num_kv_heads == 0 || head_dim == 0) return;
        const size_t total_len = prefix_len + tail_len;
        if (total_len == 0) return;

        const size_t gqa_group_size = num_heads / num_kv_heads;

        #pragma omp parallel for schedule(static) if (num_heads >= 4 && total_len >= 32)
        for (size_t h = 0; h < num_heads; ++h) {
            const size_t kv_head = (gqa_group_size > 0) ? std::min(h / gqa_group_size, num_kv_heads - 1) : 0;
            const float* q_h = q + h * head_dim;
            float* out_h = out + h * head_dim;

            float local_scores[1024];
            std::vector<float> heap_scores;
            float* scores = (total_len <= 1024) ? local_scores : (heap_scores.resize(total_len), heap_scores.data());
            float max_score = -1e30f;

            for (size_t s = 0; s < prefix_len; ++s) {
                const float* k_row = prefix_k + (s * num_kv_heads + kv_head) * head_dim;
                float dot = dot_product(q_h, k_row, head_dim);
                float score = sanitize_logit(dot * scale, ctx);
                scores[s] = score;
                if (score > max_score) max_score = score;
            }

            for (size_t s = 0; s < tail_len; ++s) {
                const float* k_row = tail_k + (s * num_kv_heads + kv_head) * head_dim;
                float dot = dot_product(q_h, k_row, head_dim);
                float score = sanitize_logit(dot * scale, ctx);
                scores[prefix_len + s] = score;
                if (score > max_score) max_score = score;
            }

            float sum_exp = 0.0f;
            for (size_t s = 0; s < total_len; ++s) {
                scores[s] = std::exp(scores[s] - max_score);
                sum_exp += scores[s];
            }
            const float inv_sum = sum_exp > 0.0f ? (1.0f / (sum_exp + 1e-9f)) : 0.0f;
            for (size_t s = 0; s < total_len; ++s) {
                scores[s] *= inv_sum;
            }

            std::fill(out_h, out_h + head_dim, 0.0f);
            for (size_t s = 0; s < prefix_len; ++s) {
                const float* v_row = prefix_v + (s * num_kv_heads + kv_head) * head_dim;
                accumulate_weighted(out_h, v_row, scores[s], head_dim);
            }
            for (size_t s = 0; s < tail_len; ++s) {
                const float* v_row = tail_v + (s * num_kv_heads + kv_head) * head_dim;
                accumulate_weighted(out_h, v_row, scores[prefix_len + s], head_dim);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Split-K Flash Attention Decoding for Long Contexts
    // Partitions the KV sequence into parallel chunks of size chunk_size (default 256),
    // computing local log-sum-exp and partial accumulators, then reducing across splits.
    // ------------------------------------------------------------------------
    static void split_k_scaled_dot_product_decode(
        const float* __restrict__ q,
        const float* __restrict__ k,
        const float* __restrict__ v,
        float* __restrict__ out,
        size_t seq_len,
        size_t head_dim,
        float scale,
        size_t chunk_size = 256,
        const MathContext& ctx = MathContext()) {
        if (!q || !k || !v || !out || seq_len == 0 || head_dim == 0) return;
        if (seq_len <= 1024 || chunk_size == 0) {
            scaled_dot_product_decode(q, k, v, out, seq_len, head_dim, scale, ctx);
            return;
        }

        const size_t num_splits = (seq_len + chunk_size - 1) / chunk_size;
        constexpr size_t kMaxSplits = 64;
        const size_t actual_splits = std::min(num_splits, kMaxSplits);

        float split_m[kMaxSplits];
        float split_l[kMaxSplits];
        float split_acc[kMaxSplits * 256];
        std::vector<float> heap_split_acc;
        float* acc_buf = (head_dim <= 256) ? split_acc : (heap_split_acc.resize(actual_splits * head_dim), heap_split_acc.data());

        #pragma omp parallel for schedule(dynamic) if (actual_splits >= 2)
        for (size_t sp = 0; sp < actual_splits; ++sp) {
            const size_t s_begin = sp * chunk_size;
            const size_t s_end = std::min(s_begin + chunk_size, seq_len);
            const size_t split_len = s_end - s_begin;
            float* acc_sp = acc_buf + sp * head_dim;

            float local_scores[256];
            std::vector<float> heap_scores;
            float* scores = (split_len <= 256) ? local_scores : (heap_scores.resize(split_len), heap_scores.data());
            float max_score = -1e30f;

            for (size_t s = 0; s < split_len; ++s) {
                const float* k_row = k + (s_begin + s) * head_dim;
                float dot = dot_product(q, k_row, head_dim);
                float score = sanitize_logit(dot * scale, ctx);
                scores[s] = score;
                if (score > max_score) max_score = score;
            }

            float sum_exp = 0.0f;
            for (size_t s = 0; s < split_len; ++s) {
                scores[s] = std::exp(scores[s] - max_score);
                sum_exp += scores[s];
            }

            std::fill(acc_sp, acc_sp + head_dim, 0.0f);
            for (size_t s = 0; s < split_len; ++s) {
                const float* v_row = v + (s_begin + s) * head_dim;
                accumulate_weighted(acc_sp, v_row, scores[s], head_dim);
            }

            split_m[sp] = max_score;
            split_l[sp] = sum_exp;
        }

        // Global split reduction using stabilized log-sum-exp
        float global_m = -1e30f;
        for (size_t sp = 0; sp < actual_splits; ++sp) {
            if (split_m[sp] > global_m) global_m = split_m[sp];
        }

        float global_l = 0.0f;
        for (size_t sp = 0; sp < actual_splits; ++sp) {
            float factor = std::exp(split_m[sp] - global_m);
            global_l += split_l[sp] * factor;
        }

        const float inv_global_l = global_l > 0.0f ? (1.0f / (global_l + 1e-9f)) : 0.0f;
        std::fill(out, out + head_dim, 0.0f);

        for (size_t sp = 0; sp < actual_splits; ++sp) {
            float factor = std::exp(split_m[sp] - global_m) * inv_global_l;
            const float* acc_sp = acc_buf + sp * head_dim;
            accumulate_weighted(out, acc_sp, factor, head_dim);
        }
    }

    // ------------------------------------------------------------------------
    // PagedAttention Decoding: Scales across non-contiguous physical KV blocks
    // eliminating internal and external fragmentation (NVIDIA / vLLM style)
    // ------------------------------------------------------------------------
    static void paged_scaled_dot_product_decode(
        const float* __restrict__ q,
        const uint32_t* block_table,
        size_t num_blocks,
        const FastPagedKVCache& cache,
        size_t seq_len,
        size_t block_size,
        size_t num_kv_heads,
        size_t kv_head,
        size_t head_dim,
        float scale,
        float* __restrict__ out,
        const MathContext& ctx = MathContext()) {
        if (!q || !block_table || !out || seq_len == 0 || head_dim == 0 || block_size == 0) return;

        float local_scores[1024];
        std::vector<float> heap_scores;
        float* scores = (seq_len <= 1024) ? local_scores : (heap_scores.resize(seq_len), heap_scores.data());
        float max_score = -1e30f;

        size_t tokens_remaining = seq_len;
        size_t global_t = 0;

        for (size_t b = 0; b < num_blocks && tokens_remaining > 0; ++b) {
            uint32_t block_id = block_table[b];
            const PagedBlock& block = cache.get_block(block_id);
            const size_t tokens_in_block = std::min(block_size, tokens_remaining);

            for (size_t t = 0; t < tokens_in_block; ++t) {
                const float* k_row = block.k_token_ptr(t, kv_head, num_kv_heads, head_dim);
                float dot = dot_product(q, k_row, head_dim);
                float score = sanitize_logit(dot * scale, ctx);
                scores[global_t++] = score;
                if (score > max_score) max_score = score;
            }
            tokens_remaining -= tokens_in_block;
        }

        // Softmax
        float sum_exp = 0.0f;
        for (size_t s = 0; s < seq_len; ++s) {
            scores[s] = std::exp(scores[s] - max_score);
            sum_exp += scores[s];
        }
        const float inv_sum = sum_exp > 0.0f ? (1.0f / (sum_exp + 1e-9f)) : 0.0f;
        for (size_t s = 0; s < seq_len; ++s) {
            scores[s] *= inv_sum;
        }

        // Weighted value accumulation
        std::fill(out, out + head_dim, 0.0f);
        tokens_remaining = seq_len;
        global_t = 0;

        for (size_t b = 0; b < num_blocks && tokens_remaining > 0; ++b) {
            uint32_t block_id = block_table[b];
            const PagedBlock& block = cache.get_block(block_id);
            const size_t tokens_in_block = std::min(block_size, tokens_remaining);

            for (size_t t = 0; t < tokens_in_block; ++t) {
                const float* v_row = block.v_token_ptr(t, kv_head, num_kv_heads, head_dim);
                accumulate_weighted(out, v_row, scores[global_t++], head_dim);
            }
            tokens_remaining -= tokens_in_block;
        }
    }

    /// Convenience wrapper for paged decoding of a sequence from FastPagedKVCache
    static void paged_scaled_dot_product_decode(
        const float* __restrict__ q,
        const FastPagedKVCache& cache,
        uint64_t seq_id,
        size_t kv_head,
        float* __restrict__ out,
        float scale,
        const MathContext& ctx = MathContext()) {
        std::vector<uint32_t> block_table;
        size_t seq_len = 0;
        if (!cache.get_block_table(seq_id, block_table, seq_len) || seq_len == 0) return;

        const auto& cfg = cache.config();
        paged_scaled_dot_product_decode(
            q, block_table.data(), block_table.size(),
            cache, seq_len, cfg.block_size, cfg.num_kv_heads, kv_head,
            cfg.head_dim, scale, out, ctx);
    }

    /// Paged Grouped-Query Attention (GQA) & Multi-Head Attention Decoding
    static void paged_gqa_decode(
        const float* __restrict__ q,
        const FastPagedKVCache& cache,
        uint64_t seq_id,
        float* __restrict__ out,
        size_t num_heads,
        size_t num_kv_heads,
        size_t head_dim,
        float scale,
        const MathContext& ctx = MathContext()) {
        if (!q || !out || num_heads == 0 || num_kv_heads == 0 || head_dim == 0) return;

        std::vector<uint32_t> block_table;
        size_t seq_len = 0;
        if (!cache.get_block_table(seq_id, block_table, seq_len) || seq_len == 0) return;

        const size_t block_size = cache.config().block_size;
        const size_t gqa_group_size = num_heads / num_kv_heads;

        #pragma omp parallel for schedule(static) if (num_heads >= 4 && seq_len >= 32)
        for (size_t h = 0; h < num_heads; ++h) {
            const size_t kv_head = (gqa_group_size > 0) ? std::min(h / gqa_group_size, num_kv_heads - 1) : 0;
            const float* q_h = q + h * head_dim;
            float* out_h = out + h * head_dim;

            paged_scaled_dot_product_decode(
                q_h, block_table.data(), block_table.size(),
                cache, seq_len, block_size, num_kv_heads, kv_head,
                head_dim, scale, out_h, ctx);
        }
    }
};

} // namespace strata::kernels

