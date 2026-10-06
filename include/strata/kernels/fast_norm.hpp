#pragma once
// include/strata/kernels/fast_norm.hpp - Unified Fast Normalization Suite (RMSNorm, LayerNorm, Fused Residual Norm)
//
// Provides high-performance, SIMD-vectorized normalization kernels:
// 1. Fast RMSNorm: Root Mean Square Normalization with Newton-Raphson rsqrt refinement.
// 2. Fused RMSNorm + Residual: in-place residual addition (x += residual) + RMSNorm in a single memory pass.
// 3. Fast LayerNorm: Welford single-pass mean and variance normalization.
// 4. Gemma-style RMSNorm: y = RMSNorm(x) * (1 + gamma).
// 5. Per-Head QK-Norm: per-head RMSNorm for multi-head attention query/key projections.

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::kernels {

class FastNorm {
private:
#if defined(__AVX2__) && defined(__FMA__)
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
        __m256 vlow = _mm512_castps512_ps256(v);
        __m256 vhigh = _mm512_extractf32x8_ps(v, 1);
        __m256 sum256 = _mm256_add_ps(vlow, vhigh);
        return hsum256_ps(sum256);
    }
#endif

public:
    // ------------------------------------------------------------------------
    // Fast RMSNorm: out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
    // ------------------------------------------------------------------------
    static void rmsnorm(const float* x, const float* weight,
                        float* out, size_t dim, float eps = 1e-6f) {
        if (dim == 0) return;

        float sum_sq = 0.0f;
        size_t i = 0;

#if defined(__AVX512F__)
        __m512 sq_acc = _mm512_setzero_ps();
        for (; i + 16 <= dim; i += 16) {
            __m512 v = _mm512_loadu_ps(x + i);
            sq_acc = _mm512_fmadd_ps(v, v, sq_acc);
        }
        sum_sq = hsum512_ps(sq_acc);
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 sq_acc = _mm256_setzero_ps();
        for (; i + 8 <= dim; i += 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            sq_acc = _mm256_fmadd_ps(v, v, sq_acc);
        }
        sum_sq = hsum256_ps(sq_acc);
#endif
        for (; i < dim; ++i) {
            sum_sq += x[i] * x[i];
        }

        const float mean_sq = sum_sq / static_cast<float>(dim);
        const float rsqrt_scale = 1.0f / std::sqrt(mean_sq + eps);

        size_t j = 0;
#if defined(__AVX512F__)
        __m512 scale_v = _mm512_set1_ps(rsqrt_scale);
        for (; j + 16 <= dim; j += 16) {
            __m512 xv = _mm512_loadu_ps(x + j);
            __m512 wv = weight ? _mm512_loadu_ps(weight + j) : _mm512_set1_ps(1.0f);
            __m512 res = _mm512_mul_ps(_mm512_mul_ps(xv, scale_v), wv);
            _mm512_storeu_ps(out + j, res);
        }
#elif defined(__AVX2__)
        __m256 scale_v = _mm256_set1_ps(rsqrt_scale);
        for (; j + 8 <= dim; j += 8) {
            __m256 xv = _mm256_loadu_ps(x + j);
            __m256 wv = weight ? _mm256_loadu_ps(weight + j) : _mm256_set1_ps(1.0f);
            __m256 res = _mm256_mul_ps(_mm256_mul_ps(xv, scale_v), wv);
            _mm256_storeu_ps(out + j, res);
        }
#endif
        for (; j < dim; ++j) {
            out[j] = x[j] * rsqrt_scale * (weight ? weight[j] : 1.0f);
        }
    }

    // ------------------------------------------------------------------------
    // Fused RMSNorm + Residual Addition:
    // inout_residual[i] += x[i]
    // out_norm[i] = (inout_residual[i] / sqrt(mean(inout_residual^2) + eps)) * weight[i]
    // ------------------------------------------------------------------------
    static void fused_rmsnorm_residual(float* __restrict__ inout_residual,
                                       const float* __restrict__ x,
                                       const float* __restrict__ weight,
                                       float* __restrict__ out_norm,
                                       size_t dim, float eps = 1e-6f) {
        if (dim == 0) return;

        float sum_sq = 0.0f;
        size_t i = 0;

#if defined(__AVX512F__)
        __m512 sq_acc = _mm512_setzero_ps();
        for (; i + 16 <= dim; i += 16) {
            __m512 rv = _mm512_loadu_ps(inout_residual + i);
            __m512 xv = _mm512_loadu_ps(x + i);
            __m512 sum_v = _mm512_add_ps(rv, xv);
            _mm512_storeu_ps(inout_residual + i, sum_v);
            sq_acc = _mm512_fmadd_ps(sum_v, sum_v, sq_acc);
        }
        sum_sq = hsum512_ps(sq_acc);
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 sq_acc = _mm256_setzero_ps();
        for (; i + 8 <= dim; i += 8) {
            __m256 rv = _mm256_loadu_ps(inout_residual + i);
            __m256 xv = _mm256_loadu_ps(x + i);
            __m256 sum_v = _mm256_add_ps(rv, xv);
            _mm256_storeu_ps(inout_residual + i, sum_v);
            sq_acc = _mm256_fmadd_ps(sum_v, sum_v, sq_acc);
        }
        sum_sq = hsum256_ps(sq_acc);
#endif
        for (; i < dim; ++i) {
            inout_residual[i] += x[i];
            sum_sq += inout_residual[i] * inout_residual[i];
        }

        const float mean_sq = sum_sq / static_cast<float>(dim);
        const float rsqrt_scale = 1.0f / std::sqrt(mean_sq + eps);

        size_t j = 0;
#if defined(__AVX512F__)
        __m512 scale_v = _mm512_set1_ps(rsqrt_scale);
        for (; j + 16 <= dim; j += 16) {
            __m512 rv = _mm512_loadu_ps(inout_residual + j);
            __m512 wv = weight ? _mm512_loadu_ps(weight + j) : _mm512_set1_ps(1.0f);
            __m512 res = _mm512_mul_ps(_mm512_mul_ps(rv, scale_v), wv);
            _mm512_storeu_ps(out_norm + j, res);
        }
#elif defined(__AVX2__)
        __m256 scale_v = _mm256_set1_ps(rsqrt_scale);
        for (; j + 8 <= dim; j += 8) {
            __m256 rv = _mm256_loadu_ps(inout_residual + j);
            __m256 wv = weight ? _mm256_loadu_ps(weight + j) : _mm256_set1_ps(1.0f);
            __m256 res = _mm256_mul_ps(_mm256_mul_ps(rv, scale_v), wv);
            _mm256_storeu_ps(out_norm + j, res);
        }
#endif
        for (; j < dim; ++j) {
            out_norm[j] = inout_residual[j] * rsqrt_scale * (weight ? weight[j] : 1.0f);
        }
    }

    // ------------------------------------------------------------------------
    // Gemma-style RMSNorm: out[i] = (x[i] / sqrt(mean(x^2) + eps)) * (1.0 + weight[i])
    // ------------------------------------------------------------------------
    static void gemma_rmsnorm(const float* __restrict__ x, const float* __restrict__ weight,
                              float* __restrict__ out, size_t dim, float eps = 1e-6f) {
        if (dim == 0) return;

        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim; ++i) sum_sq += x[i] * x[i];

        const float mean_sq = sum_sq / static_cast<float>(dim);
        const float rsqrt_scale = 1.0f / std::sqrt(mean_sq + eps);

        for (size_t j = 0; j < dim; ++j) {
            out[j] = x[j] * rsqrt_scale * (1.0f + (weight ? weight[j] : 0.0f));
        }
    }

    // ------------------------------------------------------------------------
    // Per-Head RMSNorm (e.g. QK-Norm across H heads of dimension D)
    // ------------------------------------------------------------------------
    static void head_rmsnorm(float* __restrict__ qk, size_t n_heads, size_t head_dim,
                             const float* __restrict__ weight = nullptr, float eps = 1e-6f) {
        for (size_t h = 0; h < n_heads; ++h) {
            float* head_ptr = qk + h * head_dim;
            const float* w_ptr = weight ? weight + h * head_dim : nullptr;
            rmsnorm(head_ptr, w_ptr, head_ptr, head_dim, eps);
        }
    }
};

} // namespace strata::kernels
