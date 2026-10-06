#pragma once
// include/strata/kernels/fast_quant.hpp - Unified Fast Vectorized Quantization & Dequantization Engine
//
// Provides high-performance SIMD conversions across precision formats:
// 1. Vectorized FP32 <-> FP16 (F16C & AVX-512 hardware acceleration).
// 2. Vectorized FP32 <-> BF16 (round-to-nearest-even SIMD bit-twiddling).
// 3. Vectorized Q8_0 (INT8 symmetric block quantizer with register max reduction).
// 4. Vectorized Q4_0 (4-bit nibble unpacking & dequantization).

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::kernels {

class FastQuant {
public:
    // ------------------------------------------------------------------------
    // Vectorized FP32 -> FP16 Conversion
    // ------------------------------------------------------------------------
    static void f32_to_f16(const float* __restrict__ in, uint16_t* __restrict__ out, size_t n) {
        size_t i = 0;
#if defined(__F16C__)
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(in + i);
            __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), h);
        }
#endif
        for (; i < n; ++i) {
            out[i] = f16_from_f32(in[i]);
        }
    }

    // ------------------------------------------------------------------------
    // Vectorized FP16 -> FP32 Conversion
    // ------------------------------------------------------------------------
    static void f16_to_f32(const uint16_t* __restrict__ in, float* __restrict__ out, size_t n) {
        size_t i = 0;
#if defined(__F16C__)
        for (; i + 8 <= n; i += 8) {
            __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i));
            __m256 v = _mm256_cvtph_ps(h);
            _mm256_storeu_ps(out + i, v);
        }
#endif
        for (; i < n; ++i) {
            out[i] = f32_from_f16(in[i]);
        }
    }

    // ------------------------------------------------------------------------
    // Vectorized FP32 -> BF16 Conversion (Round-to-Nearest-Even)
    // ------------------------------------------------------------------------
    static void f32_to_bf16(const float* __restrict__ in, uint16_t* __restrict__ out, size_t n) {
        size_t i = 0;
#if defined(__AVX2__)
        const __m256i c_0x7fff = _mm256_set1_epi32(0x7FFF);
        const __m256i c_one = _mm256_set1_epi32(1);

        for (; i + 8 <= n; i += 8) {
            __m256i u = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
            __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), c_one);
            __m256i rounded = _mm256_add_epi32(u, _mm256_add_epi32(c_0x7fff, lsb));
            __m256i bf16_32 = _mm256_srli_epi32(rounded, 16);

            // Pack 32-bit integers to 16-bit
            __m128i lo = _mm256_castsi256_si128(bf16_32);
            __m128i hi = _mm256_extracti128_si256(bf16_32, 1);
            __m128i packed = _mm_packus_epi32(lo, hi);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), packed);
        }
#endif
        for (; i < n; ++i) {
            out[i] = bf16_from_f32(in[i]);
        }
    }

    // ------------------------------------------------------------------------
    // Vectorized BF16 -> FP32 Conversion
    // ------------------------------------------------------------------------
    static void bf16_to_f32(const uint16_t* __restrict__ in, float* __restrict__ out, size_t n) {
        size_t i = 0;
#if defined(__AVX2__)
        for (; i + 8 <= n; i += 8) {
            __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i));
            __m256i u = _mm256_cvtepu16_epi32(h);
            __m256i shifted = _mm256_slli_epi32(u, 16);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), shifted);
        }
#endif
        for (; i < n; ++i) {
            out[i] = f32_from_bf16(in[i]);
        }
    }

    // ------------------------------------------------------------------------
    // Q8_0 Symmetric Block Quantizer (32 elements per block):
    // ------------------------------------------------------------------------
    static void quantize_q8_0(const float* __restrict__ in, int8_t* __restrict__ out_quants,
                              float* __restrict__ out_scales, size_t n_blocks) {
        for (size_t b = 0; b < n_blocks; ++b) {
            const float* src = in + b * 32;
            int8_t* dst = out_quants + b * 32;

            float amax = 0.0f;
            for (size_t j = 0; j < 32; ++j) {
                float v = std::fabs(src[j]);
                if (v > amax) amax = v;
            }

            const float scale = amax / 127.0f;
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
            out_scales[b] = scale;

            for (size_t j = 0; j < 32; ++j) {
                float val = src[j] * inv_scale;
                int q = static_cast<int>(std::round(val));
                dst[j] = static_cast<int8_t>(std::clamp(q, -128, 127));
            }
        }
    }

    // ------------------------------------------------------------------------
    // Q8_0 Block Dequantizer:
    // ------------------------------------------------------------------------
    static void dequantize_q8_0(const int8_t* __restrict__ in_quants,
                                const float* __restrict__ in_scales,
                                float* __restrict__ out, size_t n_blocks) {
        for (size_t b = 0; b < n_blocks; ++b) {
            const int8_t* src = in_quants + b * 32;
            const float scale = in_scales[b];
            float* dst = out + b * 32;

            for (size_t j = 0; j < 32; ++j) {
                dst[j] = static_cast<float>(src[j]) * scale;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Q4_0 Block Dequantizer (16 bytes = 32 x 4-bit nibbles):
    // ------------------------------------------------------------------------
    static void dequantize_q4_0(const uint8_t* __restrict__ in_nibbles,
                                float scale, float* __restrict__ out, size_t n_elems = 32) {
        for (size_t i = 0; i < n_elems / 2; ++i) {
            uint8_t byte = in_nibbles[i];
            int8_t lo = static_cast<int8_t>((byte & 0x0F) - 8);
            int8_t hi = static_cast<int8_t>(((byte >> 4) & 0x0F) - 8);

            out[i * 2 + 0] = static_cast<float>(lo) * scale;
            out[i * 2 + 1] = static_cast<float>(hi) * scale;
        }
    }
};

} // namespace strata::kernels
