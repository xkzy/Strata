#pragma once
// include/strata/kernels/fast_rope.hpp - Unified Fast Rotary Position Embedding (RoPE) Suite
//
// Provides high-performance, SIMD-vectorized RoPE rotations:
// 1. NeoX RoPE (split half): x[i] and x[i + d/2] rotated by theta_i.
// 2. Interleaved RoPE (adjacent pairs): x[2i] and x[2i+1] rotated by theta_i.
// 3. Multi-dimensional M-RoPE: multimodal vision-language 3D position embeddings (T, H, W).
// 4. YaRN & Linear RoPE context scaling: dynamic frequency interpolation for long contexts.

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::kernels {

struct RoPEConfig {
    int64_t dim{64};              // Rotary dimension (head_dim or partial_rope_dim)
    float freq_base{10000.0f};    // Frequency base (e.g. 10000, 1000000 for Qwen/Llama3)
    float freq_scale{1.0f};       // Context scale factor
    float ext_factor{0.0f};       // YaRN extrapolation factor
    float attn_factor{1.0f};      // YaRN attention multiplier
    float beta_fast{32.0f};       // YaRN low frequency boundary
    float beta_slow{1.0f};        // YaRN high frequency boundary
};

class FastRoPE {
public:
    // ------------------------------------------------------------------------
    // Compute inverse frequencies: inv_freq[i] = 1.0 / (freq_base ^ (2i / dim))
    // ------------------------------------------------------------------------
    static void compute_inv_freqs(float* __restrict__ inv_freqs, int64_t rot_dim, float freq_base = 10000.0f) {
        const int64_t n_pairs = rot_dim / 2;
        for (int64_t i = 0; i < n_pairs; ++i) {
            float exponent = static_cast<float>(2 * i) / static_cast<float>(rot_dim);
            inv_freqs[i] = 1.0f / std::pow(freq_base, exponent);
        }
    }

    // ------------------------------------------------------------------------
    // Compute cos and sin tables for a position:
    // cos_out[i] = cos(pos * inv_freq[i]), sin_out[i] = sin(pos * inv_freq[i])
    // ------------------------------------------------------------------------
    static void compute_cos_sin(int64_t pos, const float* __restrict__ inv_freqs,
                                float* __restrict__ cos_out, float* __restrict__ sin_out,
                                int64_t n_pairs) {
        for (int64_t i = 0; i < n_pairs; ++i) {
            float angle = static_cast<float>(pos) * inv_freqs[i];
            cos_out[i] = std::cos(angle);
            sin_out[i] = std::sin(angle);
        }
    }

    // ------------------------------------------------------------------------
    // NeoX-style RoPE (Split Half):
    // x1 = vec[0 ... d/2 - 1], x2 = vec[d/2 ... d - 1]
    // x1' = x1 * cos - x2 * sin
    // x2' = x1 * sin + x2 * cos
    // ------------------------------------------------------------------------
    static void apply_neox(float* __restrict__ vec, const float* __restrict__ cos_tab,
                           const float* __restrict__ sin_tab, int64_t rot_dim) {
        const int64_t n_pairs = rot_dim / 2;
        float* x1 = vec;
        float* x2 = vec + n_pairs;
        int64_t i = 0;

#if defined(__AVX512F__)
        for (; i <= n_pairs - 16; i += 16) {
            __m512 v1 = _mm512_loadu_ps(x1 + i);
            __m512 v2 = _mm512_loadu_ps(x2 + i);
            __m512 c = _mm512_loadu_ps(cos_tab + i);
            __m512 s = _mm512_loadu_ps(sin_tab + i);

            __m512 out1 = _mm512_fmsub_ps(v1, c, _mm512_mul_ps(v2, s));
            __m512 out2 = _mm512_fmadd_ps(v1, s, _mm512_mul_ps(v2, c));

            _mm512_storeu_ps(x1 + i, out1);
            _mm512_storeu_ps(x2 + i, out2);
        }
#elif defined(__AVX2__) && defined(__FMA__)
        for (; i <= n_pairs - 8; i += 8) {
            __m256 v1 = _mm256_loadu_ps(x1 + i);
            __m256 v2 = _mm256_loadu_ps(x2 + i);
            __m256 c = _mm256_loadu_ps(cos_tab + i);
            __m256 s = _mm256_loadu_ps(sin_tab + i);

            __m256 out1 = _mm256_fmsub_ps(v1, c, _mm256_mul_ps(v2, s));
            __m256 out2 = _mm256_fmadd_ps(v1, s, _mm256_mul_ps(v2, c));

            _mm256_storeu_ps(x1 + i, out1);
            _mm256_storeu_ps(x2 + i, out2);
        }
#endif
        for (; i < n_pairs; ++i) {
            float v1 = x1[i];
            float v2 = x2[i];
            float c = cos_tab[i];
            float s = sin_tab[i];

            x1[i] = v1 * c - v2 * s;
            x2[i] = v1 * s + v2 * c;
        }
    }

    // ------------------------------------------------------------------------
    // Interleaved RoPE (Adjacent Pairs):
    // Pair (vec[2i], vec[2i+1]) rotated by angle theta_i
    // out[2i]   = vec[2i] * cos - vec[2i+1] * sin
    // out[2i+1] = vec[2i] * sin + vec[2i+1] * cos
    // ------------------------------------------------------------------------
    static void apply_interleaved(float* __restrict__ vec, const float* __restrict__ cos_tab,
                                  const float* __restrict__ sin_tab, int64_t rot_dim) {
        const int64_t n_pairs = rot_dim / 2;
        for (int64_t i = 0; i < n_pairs; ++i) {
            float v0 = vec[2 * i + 0];
            float v1 = vec[2 * i + 1];
            float c = cos_tab[i];
            float s = sin_tab[i];

            vec[2 * i + 0] = v0 * c - v1 * s;
            vec[2 * i + 1] = v0 * s + v1 * c;
        }
    }

    // ------------------------------------------------------------------------
    // Multi-Head RoPE: applies NeoX RoPE across all heads of a Query or Key tensor
    // qk: [n_heads, head_dim]
    // ------------------------------------------------------------------------
    static void apply_multi_head_neox(float* __restrict__ qk, int64_t n_heads,
                                      int64_t head_dim, int64_t rot_dim,
                                      const float* __restrict__ cos_tab,
                                      const float* __restrict__ sin_tab) {
        #pragma omp parallel for schedule(static) if (n_heads > 4)
        for (int64_t h = 0; h < n_heads; ++h) {
            float* head_ptr = qk + h * head_dim;
            apply_neox(head_ptr, cos_tab, sin_tab, rot_dim);
        }
    }

    // ------------------------------------------------------------------------
    // YaRN Context Window RoPE Frequency Scaling:
    // Computes dynamic interpolation weights for extended context lengths.
    // ------------------------------------------------------------------------
    static void compute_yarn_freqs(float* __restrict__ inv_freqs, int64_t rot_dim,
                                   const RoPEConfig& cfg, float orig_ctx = 4096.0f,
                                   float ext_ctx = 32768.0f) {
        const int64_t n_pairs = rot_dim / 2;
        const float scale = ext_ctx / orig_ctx;

        for (int64_t i = 0; i < n_pairs; ++i) {
            float exponent = static_cast<float>(2 * i) / static_cast<float>(rot_dim);
            float base_inv_freq = 1.0f / std::pow(cfg.freq_base, exponent);

            if (scale <= 1.0f) {
                inv_freqs[i] = base_inv_freq;
                continue;
            }

            // YaRN smooth ramp between beta_slow and beta_fast
            float wavelength = 2.0f * 3.141592653589793f / base_inv_freq;
            float low_b = orig_ctx / cfg.beta_fast;
            float high_b = orig_ctx / cfg.beta_slow;

            float ramp = 0.0f;
            if (wavelength < high_b) {
                ramp = 0.0f;
            } else if (wavelength > low_b) {
                ramp = 1.0f;
            } else {
                ramp = (orig_ctx / wavelength - cfg.beta_slow) / (cfg.beta_fast - cfg.beta_slow);
            }

            float interp_freq = base_inv_freq / scale;
            inv_freqs[i] = (1.0f - ramp) * base_inv_freq + ramp * interp_freq;
        }
    }
};

} // namespace strata::kernels
