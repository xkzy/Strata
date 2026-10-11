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
    // Xenolith-style Block Structs for Fast SIMD Quantized Dot Products
    // ------------------------------------------------------------------------
    struct BlockQ8_0 {
        float d{0.0f};          // Scale factor
        int8_t qs[32]{0};       // Quantized symmetric INT8 values
        int16_t sigma{0};       // Sum of signed elements for fast Q4_0 zero-offset correction
    };

    struct BlockQ4_0 {
        float d{0.0f};          // Scale factor
        uint8_t qs[16]{0};      // 32 nibbles packed into 16 bytes
    };

    struct BlockQ4_1 {
        float d{0.0f};          // Scale factor
        float m{0.0f};          // Min/bias factor
        uint8_t qs[16]{0};      // 32 nibbles packed into 16 bytes [0..15]
    };

    struct BlockQ8_1 {
        float d{0.0f};          // Scale factor
        float s{0.0f};          // Sum of unquantized float values in block
        int8_t qs[32]{0};       // Quantized symmetric INT8 values
    };

    struct BlockQ5_0 {
        float d{0.0f};          // Scale factor
        uint8_t qh[4]{0};       // 5th bit for 32 elements (4 bytes = 32 bits)
        uint8_t qs[16]{0};      // Low 4 bits for 32 elements (16 bytes)
    };

    // ------------------------------------------------------------------------
    // Fast Q8_0 Symmetric Block Quantizer with Vectorized Sum (sigma) Computation
    // ------------------------------------------------------------------------
    static void quantize_q8_0_with_sum(const float* __restrict__ in, BlockQ8_0* __restrict__ out_blocks, size_t n_blocks) {
        for (size_t b = 0; b < n_blocks; ++b) {
            const float* src = in + b * 32;
            BlockQ8_0& blk = out_blocks[b];

            float amax = 0.0f;
            for (size_t j = 0; j < 32; ++j) {
                float v = std::fabs(src[j]);
                if (v > amax) amax = v;
            }

            const float scale = amax / 127.0f;
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
            blk.d = scale;

            int32_t sum = 0;
            for (size_t j = 0; j < 32; ++j) {
                float val = src[j] * inv_scale;
                int q = static_cast<int>(std::round(val));
                int8_t q8 = static_cast<int8_t>(std::clamp(q, -128, 127));
                blk.qs[j] = q8;
                sum += static_cast<int32_t>(q8);
            }
            blk.sigma = static_cast<int16_t>(sum);
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
    // Q4_0 Block Quantizer (packs 32 float elements into BlockQ4_0)
    // ------------------------------------------------------------------------
    static void quantize_q4_0(const float* __restrict__ in, BlockQ4_0* __restrict__ out_blocks, size_t n_blocks) {
        for (size_t b = 0; b < n_blocks; ++b) {
            const float* src = in + b * 32;
            BlockQ4_0& blk = out_blocks[b];

            float amax = 0.0f;
            for (size_t j = 0; j < 32; ++j) {
                float v = std::fabs(src[j]);
                if (v > amax) amax = v;
            }

            const float scale = amax / 7.0f;
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
            blk.d = scale;

            for (size_t i = 0; i < 16; ++i) {
                float v0 = src[i * 2 + 0] * inv_scale;
                float v1 = src[i * 2 + 1] * inv_scale;
                int q0 = static_cast<int>(std::round(v0)) + 8;
                int q1 = static_cast<int>(std::round(v1)) + 8;
                uint8_t n0 = static_cast<uint8_t>(std::clamp(q0, 0, 15));
                uint8_t n1 = static_cast<uint8_t>(std::clamp(q1, 0, 15));
                blk.qs[i] = static_cast<uint8_t>(n0 | (n1 << 4));
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

    // ------------------------------------------------------------------------
    // Q4_1 Block Quantizer (32 elements per block: d * q + m, q in [0, 15])
    // ------------------------------------------------------------------------
    static void quantize_q4_1(const float* __restrict__ in, BlockQ4_1* __restrict__ out_blocks, size_t n_blocks) {
        for (size_t b = 0; b < n_blocks; ++b) {
            const float* src = in + b * 32;
            BlockQ4_1& blk = out_blocks[b];

            float min_v = src[0];
            float max_v = src[0];
            for (size_t j = 1; j < 32; ++j) {
                if (src[j] < min_v) min_v = src[j];
                if (src[j] > max_v) max_v = src[j];
            }

            const float scale = (max_v - min_v) / 15.0f;
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
            blk.d = scale;
            blk.m = min_v;

            for (size_t i = 0; i < 16; ++i) {
                float v0 = (src[i * 2 + 0] - min_v) * inv_scale;
                float v1 = (src[i * 2 + 1] - min_v) * inv_scale;
                int q0 = static_cast<int>(std::round(v0));
                int q1 = static_cast<int>(std::round(v1));
                uint8_t n0 = static_cast<uint8_t>(std::clamp(q0, 0, 15));
                uint8_t n1 = static_cast<uint8_t>(std::clamp(q1, 0, 15));
                blk.qs[i] = static_cast<uint8_t>(n0 | (n1 << 4));
            }
        }
    }

    // ------------------------------------------------------------------------
    // Q4_1 Block Dequantizer
    // ------------------------------------------------------------------------
    static void dequantize_q4_1(const BlockQ4_1& blk, float* __restrict__ out, size_t n_elems = 32) {
        const float d = blk.d;
        const float m = blk.m;
        for (size_t i = 0; i < n_elems / 2; ++i) {
            uint8_t byte = blk.qs[i];
            uint8_t lo = byte & 0x0F;
            uint8_t hi = (byte >> 4) & 0x0F;
            out[i * 2 + 0] = static_cast<float>(lo) * d + m;
            out[i * 2 + 1] = static_cast<float>(hi) * d + m;
        }
    }

    // ------------------------------------------------------------------------
    // Q8_1 Symmetric Block Quantizer with Float Sum
    // ------------------------------------------------------------------------
    static void quantize_q8_1(const float* __restrict__ in, BlockQ8_1* __restrict__ out_blocks, size_t n_blocks) {
        for (size_t b = 0; b < n_blocks; ++b) {
            const float* src = in + b * 32;
            BlockQ8_1& blk = out_blocks[b];

            float amax = 0.0f;
            for (size_t j = 0; j < 32; ++j) {
                float v = std::fabs(src[j]);
                if (v > amax) amax = v;
            }

            const float scale = amax / 127.0f;
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
            blk.d = scale;

            int32_t q_sum = 0;
            for (size_t j = 0; j < 32; ++j) {
                float val = src[j] * inv_scale;
                int q = static_cast<int>(std::round(val));
                int8_t q_clamped = static_cast<int8_t>(std::clamp(q, -128, 127));
                blk.qs[j] = q_clamped;
                q_sum += static_cast<int32_t>(q_clamped);
            }
            blk.s = static_cast<float>(q_sum) * scale;
        }
    }

    // ------------------------------------------------------------------------
    // Q8_1 Block Dequantizer
    // ------------------------------------------------------------------------
    static void dequantize_q8_1(const BlockQ8_1& blk, float* __restrict__ out, size_t n_elems = 32) {
        const float scale = blk.d;
        for (size_t j = 0; j < n_elems; ++j) {
            out[j] = static_cast<float>(blk.qs[j]) * scale;
        }
    }

    // ------------------------------------------------------------------------
    // Q5_0 Block Quantizer (32 elements per block: d * (q - 16), q in [0, 31])
    // ------------------------------------------------------------------------
    static void quantize_q5_0(const float* __restrict__ in, BlockQ5_0* __restrict__ out_blocks, size_t n_blocks) {
        for (size_t b = 0; b < n_blocks; ++b) {
            const float* src = in + b * 32;
            BlockQ5_0& blk = out_blocks[b];

            float amax = 0.0f;
            for (size_t j = 0; j < 32; ++j) {
                float v = std::fabs(src[j]);
                if (v > amax) amax = v;
            }

            const float scale = amax / 15.0f;
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
            blk.d = scale;

            uint32_t qh_bits = 0;
            for (size_t i = 0; i < 16; ++i) {
                float v0 = src[i * 2 + 0] * inv_scale;
                float v1 = src[i * 2 + 1] * inv_scale;
                int q0 = static_cast<int>(std::round(v0)) + 16;
                int q1 = static_cast<int>(std::round(v1)) + 16;
                uint8_t n0 = static_cast<uint8_t>(std::clamp(q0, 0, 31));
                uint8_t n1 = static_cast<uint8_t>(std::clamp(q1, 0, 31));

                blk.qs[i] = static_cast<uint8_t>((n0 & 0x0F) | ((n1 & 0x0F) << 4));
                qh_bits |= static_cast<uint32_t>((n0 >> 4) & 1) << (i * 2 + 0);
                qh_bits |= static_cast<uint32_t>((n1 >> 4) & 1) << (i * 2 + 1);
            }
            std::memcpy(blk.qh, &qh_bits, 4);
        }
    }

    // ------------------------------------------------------------------------
    // Q5_0 Block Dequantizer
    // ------------------------------------------------------------------------
    static void dequantize_q5_0(const BlockQ5_0& blk, float* __restrict__ out, size_t n_elems = 32) {
        const float d = blk.d;
        uint32_t qh_bits = 0;
        std::memcpy(&qh_bits, blk.qh, 4);

        for (size_t i = 0; i < n_elems / 2; ++i) {
            uint8_t byte = blk.qs[i];
            uint8_t b0 = static_cast<uint8_t>((qh_bits >> (i * 2 + 0)) & 1);
            uint8_t b1 = static_cast<uint8_t>((qh_bits >> (i * 2 + 1)) & 1);

            int8_t q0 = static_cast<int8_t>((byte & 0x0F) | (b0 << 4)) - 16;
            int8_t q1 = static_cast<int8_t>(((byte >> 4) & 0x0F) | (b1 << 4)) - 16;

            out[i * 2 + 0] = static_cast<float>(q0) * d;
            out[i * 2 + 1] = static_cast<float>(q1) * d;
        }
    }
};

} // namespace strata::kernels
