#pragma once
// include/strata/kernels/fast_matmul.hpp - Unified Fast Matrix Multiplication (GEMM / GEMV) Engine
//
// Provides high-performance, cache-blocked, vectorized matrix multiplications:
// 1. Tiled cache-blocked general matrix multiplication (GEMM) for FP32, FP16, and BF16.
// 2. High-throughput row-split Matrix-Vector (GEMV) with AVX-512 / AVX2 / FMA vectorization (1 <= M <= 8).
// 3. Fused MatMul + Activation (SiLU, SwiGLU, GELU, Sigmoid, ReLU) for zero-memory-bandwidth overhead.
// 4. Quantized INT8 / Q8_0 dot-product acceleration (AVX-VNNI & AVX512-VNNI).

#include "strata/kernels/fast_activations.hpp"
#include "strata/kernels/fast_parallel.hpp"
#include "strata/kernels/fast_quant.hpp"
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::kernels {

struct MatMulConfig {
    int64_t M{0};      // Rows of A and C
    int64_t N{0};      // Cols of B and C
    int64_t K{0};      // Cols of A, Rows of B
    int64_t lda{0};    // Leading dim of A
    int64_t ldb{0};    // Leading dim of B
    int64_t ldc{0};    // Leading dim of C
    float alpha{1.0f}; // Scale for A * B
    float beta{0.0f};  // Scale for C
    bool trans_a{false};
    bool trans_b{false};
    ActivationType activation{ActivationType::kNone};
};

class FastMatMul {
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

    // Xenolith-style fast single-block Q4_0 x Q8_0 dot-product (32 elements per block)
    static inline float dot_q4_0_q8_0_block(const FastQuant::BlockQ4_0& w, const FastQuant::BlockQ8_0& a) {
        int32_t int_sum = 0;
#if defined(__AVX2__)
        __m128i qw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w.qs));
        __m128i qw_lo = _mm_and_si128(qw, _mm_set1_epi8(0x0F));
        __m128i qw_hi = _mm_and_si128(_mm_srli_epi16(qw, 4), _mm_set1_epi8(0x0F));

        __m128i qw_unpacked_lo = _mm_unpacklo_epi8(qw_lo, qw_hi);
        __m128i qw_unpacked_hi = _mm_unpackhi_epi8(qw_lo, qw_hi);

        __m128i qa_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a.qs));
        __m128i qa_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a.qs + 16));

        __m128i p_lo = _mm_maddubs_epi16(qw_unpacked_lo, qa_lo);
        __m128i p_hi = _mm_maddubs_epi16(qw_unpacked_hi, qa_hi);

        __m128i s_lo = _mm_madd_epi16(p_lo, _mm_set1_epi16(1));
        __m128i s_hi = _mm_madd_epi16(p_hi, _mm_set1_epi16(1));

        __m128i total = _mm_add_epi32(s_lo, s_hi);
        total = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0x4E));
        total = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0xB1));
        int_sum = _mm_cvtsi128_si32(total);
#else
        for (int j = 0; j < 16; ++j) {
            uint8_t byte = w.qs[j];
            int32_t n0 = static_cast<int32_t>(byte & 0x0F);
            int32_t n1 = static_cast<int32_t>((byte >> 4) & 0x0F);
            int_sum += n0 * static_cast<int32_t>(a.qs[j * 2 + 0]);
            int_sum += n1 * static_cast<int32_t>(a.qs[j * 2 + 1]);
        }
#endif
        int32_t corrected = int_sum - (8 * static_cast<int32_t>(a.sigma));
        return static_cast<float>(corrected) * (w.d * a.d);
    }

    // Xenolith-style fast single-block Q4_1 x Q8_1 dot-product (32 elements per block)
    static inline float dot_q4_1_q8_1_block(const FastQuant::BlockQ4_1& w, const FastQuant::BlockQ8_1& a) {
        int32_t int_sum = 0;
#if defined(__AVX2__)
        __m128i qw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w.qs));
        __m128i qw_lo = _mm_and_si128(qw, _mm_set1_epi8(0x0F));
        __m128i qw_hi = _mm_and_si128(_mm_srli_epi16(qw, 4), _mm_set1_epi8(0x0F));

        __m128i qw_unpacked_lo = _mm_unpacklo_epi8(qw_lo, qw_hi);
        __m128i qw_unpacked_hi = _mm_unpackhi_epi8(qw_lo, qw_hi);

        __m128i qa_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a.qs));
        __m128i qa_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a.qs + 16));

        __m128i p_lo = _mm_maddubs_epi16(qw_unpacked_lo, qa_lo);
        __m128i p_hi = _mm_maddubs_epi16(qw_unpacked_hi, qa_hi);

        __m128i s_lo = _mm_madd_epi16(p_lo, _mm_set1_epi16(1));
        __m128i s_hi = _mm_madd_epi16(p_hi, _mm_set1_epi16(1));

        __m128i total = _mm_add_epi32(s_lo, s_hi);
        total = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0x4E));
        total = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0xB1));
        int_sum = _mm_cvtsi128_si32(total);
#else
        for (int j = 0; j < 16; ++j) {
            uint8_t byte = w.qs[j];
            int32_t n0 = static_cast<int32_t>(byte & 0x0F);
            int32_t n1 = static_cast<int32_t>((byte >> 4) & 0x0F);
            int_sum += n0 * static_cast<int32_t>(a.qs[j * 2 + 0]);
            int_sum += n1 * static_cast<int32_t>(a.qs[j * 2 + 1]);
        }
#endif
        return (w.d * a.d) * static_cast<float>(int_sum) + (w.m * a.s);
    }

    // Xenolith-style fast single-block Q5_0 x Q8_0 dot-product (32 elements per block)
    static inline float dot_q5_0_q8_0_block(const FastQuant::BlockQ5_0& w, const FastQuant::BlockQ8_0& a) {
        int32_t int_sum = 0;
#if defined(__AVX2__)
        __m128i qw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w.qs));
        __m128i qw_lo = _mm_and_si128(qw, _mm_set1_epi8(0x0F));
        __m128i qw_hi = _mm_and_si128(_mm_srli_epi16(qw, 4), _mm_set1_epi8(0x0F));

        __m128i qw_unpacked_lo = _mm_unpacklo_epi8(qw_lo, qw_hi);
        __m128i qw_unpacked_hi = _mm_unpackhi_epi8(qw_lo, qw_hi);

        uint32_t qh_bits = 0;
        std::memcpy(&qh_bits, w.qh, 4);

        const uint16_t qh_lo_16 = static_cast<uint16_t>(qh_bits & 0xFFFF);
        const uint16_t qh_hi_16 = static_cast<uint16_t>((qh_bits >> 16) & 0xFFFF);
        const __m128i bit_masks = _mm_setr_epi16(1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7);

        __m128i v0 = _mm_set1_epi16(qh_lo_16 & 0xFF);
        __m128i b0 = _mm_cmpeq_epi16(_mm_and_si128(v0, bit_masks), bit_masks);
        __m128i v1 = _mm_set1_epi16((qh_lo_16 >> 8) & 0xFF);
        __m128i b1 = _mm_cmpeq_epi16(_mm_and_si128(v1, bit_masks), bit_masks);
        __m128i qh_vec_lo = _mm_and_si128(_mm_packs_epi16(b0, b1), _mm_set1_epi8(0x10));

        __m128i v2 = _mm_set1_epi16(qh_hi_16 & 0xFF);
        __m128i b2 = _mm_cmpeq_epi16(_mm_and_si128(v2, bit_masks), bit_masks);
        __m128i v3 = _mm_set1_epi16((qh_hi_16 >> 8) & 0xFF);
        __m128i b3 = _mm_cmpeq_epi16(_mm_and_si128(v3, bit_masks), bit_masks);
        __m128i qh_vec_hi = _mm_and_si128(_mm_packs_epi16(b2, b3), _mm_set1_epi8(0x10));

        __m128i qw_full_lo = _mm_or_si128(qw_unpacked_lo, qh_vec_lo);
        __m128i qw_full_hi = _mm_or_si128(qw_unpacked_hi, qh_vec_hi);

        __m128i qa_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a.qs));
        __m128i qa_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a.qs + 16));

        __m128i p_lo = _mm_maddubs_epi16(qw_full_lo, qa_lo);
        __m128i p_hi = _mm_maddubs_epi16(qw_full_hi, qa_hi);

        __m128i s_lo = _mm_madd_epi16(p_lo, _mm_set1_epi16(1));
        __m128i s_hi = _mm_madd_epi16(p_hi, _mm_set1_epi16(1));

        __m128i total = _mm_add_epi32(s_lo, s_hi);
        total = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0x4E));
        total = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0xB1));
        int_sum = _mm_cvtsi128_si32(total);
#else
        uint32_t qh_bits = 0;
        std::memcpy(&qh_bits, w.qh, 4);
        for (int j = 0; j < 16; ++j) {
            uint8_t byte = w.qs[j];
            uint8_t b0 = static_cast<uint8_t>((qh_bits >> (j * 2 + 0)) & 1);
            uint8_t b1 = static_cast<uint8_t>((qh_bits >> (j * 2 + 1)) & 1);
            int32_t n0 = static_cast<int32_t>((byte & 0x0F) | (b0 << 4));
            int32_t n1 = static_cast<int32_t>(((byte >> 4) & 0x0F) | (b1 << 4));
            int_sum += n0 * static_cast<int32_t>(a.qs[j * 2 + 0]);
            int_sum += n1 * static_cast<int32_t>(a.qs[j * 2 + 1]);
        }
#endif
        int32_t corrected = int_sum - (16 * static_cast<int32_t>(a.sigma));
        return static_cast<float>(corrected) * (w.d * a.d);
    }

public:
    // ------------------------------------------------------------------------
    // Fast Matrix-Vector Multiplication (GEMV): y = alpha * A * x + beta * y
    // A: [M, K] row-major, x: [K], y: [M]
    // ------------------------------------------------------------------------
    static void gemv_fp32(const float* __restrict__ A, const float* __restrict__ x,
                          float* __restrict__ y, int64_t M, int64_t K,
                          float alpha = 1.0f, float beta = 0.0f,
                          ActivationType act = ActivationType::kNone) {
        if (M <= 0 || K <= 0) return;

        #pragma omp parallel for schedule(static) if (M * K >= fast_parallel::kGemvMinMacs)
        for (int64_t i = 0; i < M; ++i) {
            const float* row = A + i * K;
            float acc = 0.0f;
            int64_t k = 0;

#if defined(__AVX512F__)
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();

            for (; k <= K - 32; k += 32) {
                __m512 a0 = _mm512_loadu_ps(row + k + 0);
                __m512 x0 = _mm512_loadu_ps(x + k + 0);
                acc0 = _mm512_fmadd_ps(a0, x0, acc0);

                __m512 a1 = _mm512_loadu_ps(row + k + 16);
                __m512 x1 = _mm512_loadu_ps(x + k + 16);
                acc1 = _mm512_fmadd_ps(a1, x1, acc1);
            }
            for (; k <= K - 16; k += 16) {
                __m512 a = _mm512_loadu_ps(row + k);
                __m512 xv = _mm512_loadu_ps(x + k);
                acc0 = _mm512_fmadd_ps(a, xv, acc0);
            }
            acc = hsum512_ps(_mm512_add_ps(acc0, acc1));
#elif defined(__AVX2__) && defined(__FMA__)
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            for (; k <= K - 16; k += 16) {
                __m256 a0 = _mm256_loadu_ps(row + k + 0);
                __m256 x0 = _mm256_loadu_ps(x + k + 0);
                acc0 = _mm256_fmadd_ps(a0, x0, acc0);

                __m256 a1 = _mm256_loadu_ps(row + k + 8);
                __m256 x1 = _mm256_loadu_ps(x + k + 8);
                acc1 = _mm256_fmadd_ps(a1, x1, acc1);
            }
            for (; k <= K - 8; k += 8) {
                __m256 a = _mm256_loadu_ps(row + k);
                __m256 xv = _mm256_loadu_ps(x + k);
                acc0 = _mm256_fmadd_ps(a, xv, acc0);
            }
            acc = hsum256_ps(_mm256_add_ps(acc0, acc1));
#else
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            for (; k <= K - 4; k += 4) {
                sum0 += row[k + 0] * x[k + 0];
                sum1 += row[k + 1] * x[k + 1];
                sum2 += row[k + 2] * x[k + 2];
                sum3 += row[k + 3] * x[k + 3];
            }
            acc = (sum0 + sum1) + (sum2 + sum3);
#endif
            for (; k < K; ++k) {
                acc += row[k] * x[k];
            }

            float val = alpha * acc;
            if (beta != 0.0f) {
                val += beta * y[i];
            }
            if (act != ActivationType::kNone) {
                val = fast_math::fast_activate(act, val);
            }
            y[i] = val;
        }
    }

    // ------------------------------------------------------------------------
    // Multi-Token Batch GEMV: Y = alpha * X * W^T + beta * Y
    // X: [T, K] row-major, W: [N, K] row-major, Y: [T, N]
    // Optimized for decoding with speculative multi-tokens (1 <= T <= 8).
    // ------------------------------------------------------------------------
    static void batch_gemv_fp32(const float* __restrict__ X, int64_t ldx,
                                const float* __restrict__ W, int64_t ldw,
                                float* __restrict__ Y, int64_t ldy,
                                int64_t T, int64_t N, int64_t K,
                                float alpha = 1.0f, float beta = 0.0f,
                                ActivationType act = ActivationType::kNone) {
        if (T <= 0 || N <= 0 || K <= 0) return;
        if (ldx <= 0) ldx = K;
        if (ldw <= 0) ldw = K;
        if (ldy <= 0) ldy = N;

        #pragma omp parallel for schedule(static) if (T * N * K >= fast_parallel::kBatchGemvMinMacs)
        for (int64_t n = 0; n < N; ++n) {
            const float* w_row = W + n * ldw;
            for (int64_t t = 0; t < T; ++t) {
                const float* x_row = X + t * ldx;
                float acc = 0.0f;
                int64_t k = 0;

#if defined(__AVX512F__)
                __m512 acc_v = _mm512_setzero_ps();
                for (; k <= K - 16; k += 16) {
                    __m512 wv = _mm512_loadu_ps(w_row + k);
                    __m512 xv = _mm512_loadu_ps(x_row + k);
                    acc_v = _mm512_fmadd_ps(wv, xv, acc_v);
                }
                acc = hsum512_ps(acc_v);
#elif defined(__AVX2__) && defined(__FMA__)
                __m256 acc_v = _mm256_setzero_ps();
                for (; k <= K - 8; k += 8) {
                    __m256 wv = _mm256_loadu_ps(w_row + k);
                    __m256 xv = _mm256_loadu_ps(x_row + k);
                    acc_v = _mm256_fmadd_ps(wv, xv, acc_v);
                }
                acc = hsum256_ps(acc_v);
#else
                float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                for (; k <= K - 4; k += 4) {
                    sum0 += w_row[k + 0] * x_row[k + 0];
                    sum1 += w_row[k + 1] * x_row[k + 1];
                    sum2 += w_row[k + 2] * x_row[k + 2];
                    sum3 += w_row[k + 3] * x_row[k + 3];
                }
                acc = (sum0 + sum1) + (sum2 + sum3);
#endif
                for (; k < K; ++k) {
                    acc += w_row[k] * x_row[k];
                }

                float val = alpha * acc;
                if (beta != 0.0f) {
                    val += beta * Y[t * ldy + n];
                }
                if (act != ActivationType::kNone) {
                    val = fast_math::fast_activate(act, val);
                }
                Y[t * ldy + n] = val;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Tiled Cache-Blocked GEMM: C = alpha * A * B + beta * C
    // A: [M, K], B: [K, N], C: [M, N] (row-major)
    // ------------------------------------------------------------------------
    static void gemm_fp32(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int64_t M, int64_t N, int64_t K,
                          float alpha = 1.0f, float beta = 0.0f,
                          ActivationType act = ActivationType::kNone) {
        if (M <= 0 || N <= 0 || K <= 0) return;

        // If M=1, use optimized single-row path
        if (M == 1) {
            for (int64_t n = 0; n < N; ++n) {
                float acc = 0.0f;
                int64_t k = 0;
                for (; k < K; ++k) {
                    acc += A[k] * B[k * N + n];
                }
                float val = alpha * acc + (beta != 0.0f ? beta * C[n] : 0.0f);
                if (act != ActivationType::kNone) val = fast_math::fast_activate(act, val);
                C[n] = val;
            }
            return;
        }

        // Cache tile block sizes
        constexpr int64_t BM = 64;
        constexpr int64_t BN = 64;
        constexpr int64_t BK = 64;

        if (beta == 0.0f) {
            #pragma omp parallel for schedule(static) if (M * N >= fast_parallel::kElementwiseMinElems)
            for (int64_t i = 0; i < M * N; ++i) C[i] = 0.0f;
        } else if (beta != 1.0f) {
            #pragma omp parallel for schedule(static) if (M * N >= fast_parallel::kElementwiseMinElems)
            for (int64_t i = 0; i < M * N; ++i) C[i] *= beta;
        }

        #pragma omp parallel for collapse(2) schedule(dynamic) if (M * N * K >= fast_parallel::kGemmMinMacs)
        for (int64_t bm = 0; bm < M; bm += BM) {
            for (int64_t bn = 0; bn < N; bn += BN) {
                const int64_t m_end = std::min(bm + BM, M);
                const int64_t n_end = std::min(bn + BN, N);

                for (int64_t bk = 0; bk < K; bk += BK) {
                    const int64_t k_end = std::min(bk + BK, K);

                    for (int64_t i = bm; i < m_end; ++i) {
                        const float* a_row = A + i * K;
                        float* c_row = C + i * N;

                        for (int64_t k = bk; k < k_end; ++k) {
                            const float a_val = alpha * a_row[k];
                            const float* b_row = B + k * N;
                            int64_t j = bn;

#if defined(__AVX512F__)
                            __m512 av = _mm512_set1_ps(a_val);
                            for (; j <= n_end - 16; j += 16) {
                                __m512 bv = _mm512_loadu_ps(b_row + j);
                                __m512 cv = _mm512_loadu_ps(c_row + j);
                                cv = _mm512_fmadd_ps(av, bv, cv);
                                _mm512_storeu_ps(c_row + j, cv);
                            }
#elif defined(__AVX2__) && defined(__FMA__)
                            __m256 av = _mm256_set1_ps(a_val);
                            for (; j <= n_end - 8; j += 8) {
                                __m256 bv = _mm256_loadu_ps(b_row + j);
                                __m256 cv = _mm256_loadu_ps(c_row + j);
                                cv = _mm256_fmadd_ps(av, bv, cv);
                                _mm256_storeu_ps(c_row + j, cv);
                            }
#else
                            for (; j <= n_end - 4; j += 4) {
                                c_row[j + 0] += a_val * b_row[j + 0];
                                c_row[j + 1] += a_val * b_row[j + 1];
                                c_row[j + 2] += a_val * b_row[j + 2];
                                c_row[j + 3] += a_val * b_row[j + 3];
                            }
#endif
                            for (; j < n_end; ++j) {
                                c_row[j] += a_val * b_row[j];
                            }
                        }
                    }
                }

                // Apply fused activation after all K tiles finish
                if (act != ActivationType::kNone) {
                    for (int64_t i = bm; i < m_end; ++i) {
                        float* c_row = C + i * N;
                        for (int64_t j = bn; j < n_end; ++j) {
                            c_row[j] = fast_math::fast_activate(act, c_row[j]);
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // Fused SwiGLU GEMM: Computes Gate and Up projections simultaneously:
    // Out[M, N] = SiLU( X[M, K] * W_gate[N, K]^T ) * ( X[M, K] * W_up[N, K]^T )
    // Eliminates intermediate buffers and halves memory bandwidth!
    // ------------------------------------------------------------------------
    static void fused_swiglu_gemm(const float* __restrict__ X,
                                  const float* __restrict__ W_gate,
                                  const float* __restrict__ W_up,
                                  float* __restrict__ Out,
                                  int64_t M, int64_t N, int64_t K) {
        if (M <= 0 || N <= 0 || K <= 0) return;

        #pragma omp parallel for collapse(2) schedule(static) if (M * N * K >= fast_parallel::kSwigluMinMacs)
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                const float* x_row = X + m * K;
                const float* g_row = W_gate + n * K;
                const float* u_row = W_up + n * K;

                float g_sum = 0.0f;
                float u_sum = 0.0f;
                int64_t k = 0;

#if defined(__AVX512F__)
                __m512 g_acc = _mm512_setzero_ps();
                __m512 u_acc = _mm512_setzero_ps();
                for (; k <= K - 16; k += 16) {
                    __m512 xv = _mm512_loadu_ps(x_row + k);
                    __m512 gv = _mm512_loadu_ps(g_row + k);
                    __m512 uv = _mm512_loadu_ps(u_row + k);
                    g_acc = _mm512_fmadd_ps(xv, gv, g_acc);
                    u_acc = _mm512_fmadd_ps(xv, uv, u_acc);
                }
                g_sum = hsum512_ps(g_acc);
                u_sum = hsum512_ps(u_acc);
#elif defined(__AVX2__) && defined(__FMA__)
                __m256 g_acc = _mm256_setzero_ps();
                __m256 u_acc = _mm256_setzero_ps();
                for (; k <= K - 8; k += 8) {
                    __m256 xv = _mm256_loadu_ps(x_row + k);
                    __m256 gv = _mm256_loadu_ps(g_row + k);
                    __m256 uv = _mm256_loadu_ps(u_row + k);
                    g_acc = _mm256_fmadd_ps(xv, gv, g_acc);
                    u_acc = _mm256_fmadd_ps(xv, uv, u_acc);
                }
                g_sum = hsum256_ps(g_acc);
                u_sum = hsum256_ps(u_acc);
#else
                for (; k <= K - 4; k += 4) {
                    g_sum += x_row[k + 0] * g_row[k + 0];
                    u_sum += x_row[k + 0] * u_row[k + 0];

                    g_sum += x_row[k + 1] * g_row[k + 1];
                    u_sum += x_row[k + 1] * u_row[k + 1];

                    g_sum += x_row[k + 2] * g_row[k + 2];
                    u_sum += x_row[k + 2] * u_row[k + 2];

                    g_sum += x_row[k + 3] * g_row[k + 3];
                    u_sum += x_row[k + 3] * u_row[k + 3];
                }
#endif
                for (; k < K; ++k) {
                    g_sum += x_row[k] * g_row[k];
                    u_sum += x_row[k] * u_row[k];
                }

                // In-register SwiGLU
                Out[m * N + n] = fast_math::fast_swiglu(g_sum, u_sum);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Fast Quantized Matrix-Vector Multiplication (Q4_0 weights x Q8_0 vector):
    // y = alpha * A * x + beta * y
    // A: [M, K/32] BlockQ4_0, x: [K/32] BlockQ8_0, y: [M] float
    // ------------------------------------------------------------------------
    static void gemv_q4_0_q8_0(const FastQuant::BlockQ4_0* __restrict__ A,
                              const FastQuant::BlockQ8_0* __restrict__ x,
                              float* __restrict__ y,
                              int64_t M, int64_t K,
                              float alpha = 1.0f, float beta = 0.0f,
                              ActivationType act = ActivationType::kNone) {
        if (M <= 0 || K <= 0) return;
        const int64_t num_blocks = (K + 31) / 32;

        #pragma omp parallel for schedule(static) if (M * K >= fast_parallel::kGemvMinMacs)
        for (int64_t i = 0; i < (M & ~3); i += 4) {
            const FastQuant::BlockQ4_0* r0 = A + (i + 0) * num_blocks;
            const FastQuant::BlockQ4_0* r1 = A + (i + 1) * num_blocks;
            const FastQuant::BlockQ4_0* r2 = A + (i + 2) * num_blocks;
            const FastQuant::BlockQ4_0* r3 = A + (i + 3) * num_blocks;

            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            for (int64_t b = 0; b < num_blocks; ++b) {
#if defined(__AVX2__)
                const auto& ab = x[b];
                __m128i qa_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ab.qs));
                __m128i qa_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ab.qs + 16));
                const __m128i mask_0f = _mm_set1_epi8(0x0F);
                const __m128i ones = _mm_set1_epi16(1);

                auto eval_wb = [&](const FastQuant::BlockQ4_0& wb) -> float {
                    __m128i qw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(wb.qs));
                    __m128i qw_lo = _mm_and_si128(qw, mask_0f);
                    __m128i qw_hi = _mm_and_si128(_mm_srli_epi16(qw, 4), mask_0f);
                    __m128i p_lo = _mm_maddubs_epi16(_mm_unpacklo_epi8(qw_lo, qw_hi), qa_lo);
                    __m128i p_hi = _mm_maddubs_epi16(_mm_unpackhi_epi8(qw_lo, qw_hi), qa_hi);
                    __m128i tot = _mm_add_epi32(_mm_madd_epi16(p_lo, ones), _mm_madd_epi16(p_hi, ones));
                    tot = _mm_add_epi32(tot, _mm_shuffle_epi32(tot, 0x4E));
                    tot = _mm_add_epi32(tot, _mm_shuffle_epi32(tot, 0xB1));
                    int32_t isum = _mm_cvtsi128_si32(tot);
                    int32_t corrected = isum - (8 * static_cast<int32_t>(ab.sigma));
                    return static_cast<float>(corrected) * (wb.d * ab.d);
                };
                acc0 += eval_wb(r0[b]);
                acc1 += eval_wb(r1[b]);
                acc2 += eval_wb(r2[b]);
                acc3 += eval_wb(r3[b]);
#else
                acc0 += dot_q4_0_q8_0_block(r0[b], x[b]);
                acc1 += dot_q4_0_q8_0_block(r1[b], x[b]);
                acc2 += dot_q4_0_q8_0_block(r2[b], x[b]);
                acc3 += dot_q4_0_q8_0_block(r3[b], x[b]);
#endif
            }

            float v0 = alpha * acc0, v1 = alpha * acc1, v2 = alpha * acc2, v3 = alpha * acc3;
            if (beta != 0.0f) {
                v0 += beta * y[i + 0];
                v1 += beta * y[i + 1];
                v2 += beta * y[i + 2];
                v3 += beta * y[i + 3];
            }
            if (act != ActivationType::kNone) {
                v0 = fast_math::fast_activate(act, v0);
                v1 = fast_math::fast_activate(act, v1);
                v2 = fast_math::fast_activate(act, v2);
                v3 = fast_math::fast_activate(act, v3);
            }
            y[i + 0] = v0;
            y[i + 1] = v1;
            y[i + 2] = v2;
            y[i + 3] = v3;
        }

        // Remainder loop
        for (int64_t i = (M & ~3); i < M; ++i) {
            const FastQuant::BlockQ4_0* row = A + i * num_blocks;
            float acc = 0.0f;
            for (int64_t b = 0; b < num_blocks; ++b) {
                acc += dot_q4_0_q8_0_block(row[b], x[b]);
            }
            float val = alpha * acc;
            if (beta != 0.0f) val += beta * y[i];
            if (act != ActivationType::kNone) val = fast_math::fast_activate(act, val);
            y[i] = val;
        }
    }

    // ------------------------------------------------------------------------
    // Fast Quantized Matrix-Vector Multiplication (Q4_0 weights x FP32 vector):
    // Performs on-the-fly stack-buffered Q8_0 quantization of x followed by
    // SIMD integer dot products, delivering massive memory bandwidth savings.
    // ------------------------------------------------------------------------
    static void gemv_q4_0_fp32(const FastQuant::BlockQ4_0* __restrict__ A,
                              const float* __restrict__ x,
                              float* __restrict__ y,
                              int64_t M, int64_t K,
                              float alpha = 1.0f, float beta = 0.0f,
                              ActivationType act = ActivationType::kNone) {
        if (M <= 0 || K <= 0) return;
        const size_t num_blocks = (K + 31) / 32;

        FastQuant::BlockQ8_0 stack_q8[256];
        std::vector<FastQuant::BlockQ8_0> heap_q8;
        FastQuant::BlockQ8_0* q8_vec = (num_blocks <= 256)
            ? stack_q8
            : (heap_q8.resize(num_blocks), heap_q8.data());

        FastQuant::quantize_q8_0_with_sum(x, q8_vec, num_blocks);
        gemv_q4_0_q8_0(A, q8_vec, y, M, K, alpha, beta, act);
    }

    // ------------------------------------------------------------------------
    // Multi-Token Batched Matrix Multiplication (Q4_0 weights x Q8_0 activations):
    // Out: [T, M] row-major, A: [M, K/32], X: [T, K/32]
    // ------------------------------------------------------------------------
    static void gemm_q4_0_q8_0(const FastQuant::BlockQ4_0* __restrict__ A,
                              const FastQuant::BlockQ8_0* __restrict__ X,
                              float* __restrict__ Out,
                              int64_t M, int64_t T, int64_t K,
                              ActivationType act = ActivationType::kNone) {
        if (M <= 0 || T <= 0 || K <= 0) return;
        const int64_t num_blocks = (K + 31) / 32;

        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t m = 0; m < M; ++m) {
                const FastQuant::BlockQ8_0* x_row = X + t * num_blocks;
                const FastQuant::BlockQ4_0* a_row = A + m * num_blocks;

                float acc = 0.0f;
                for (int64_t b = 0; b < num_blocks; ++b) {
                    acc += dot_q4_0_q8_0_block(a_row[b], x_row[b]);
                }
                if (act != ActivationType::kNone) {
                    acc = fast_math::fast_activate(act, acc);
                }
                Out[t * M + m] = acc;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Fast Quantized Matrix-Vector Multiplication (Q4_1 weights x Q8_1 activations):
    // A: [M, K/32] BlockQ4_1, x: [K/32] BlockQ8_1, y: [M] float
    // ------------------------------------------------------------------------
    static void gemv_q4_1_q8_1(const FastQuant::BlockQ4_1* __restrict__ A,
                              const FastQuant::BlockQ8_1* __restrict__ x,
                              float* __restrict__ y,
                              int64_t M, int64_t K,
                              float alpha = 1.0f, float beta = 0.0f,
                              ActivationType act = ActivationType::kNone) {
        if (M <= 0 || K <= 0) return;
        const int64_t num_blocks = (K + 31) / 32;

        #pragma omp parallel for schedule(static) if (M * K >= fast_parallel::kGemvMinMacs)
        for (int64_t i = 0; i < M; ++i) {
            const FastQuant::BlockQ4_1* row = A + i * num_blocks;
            float acc = 0.0f;

            for (int64_t b = 0; b < num_blocks; ++b) {
                acc += dot_q4_1_q8_1_block(row[b], x[b]);
            }

            float val = alpha * acc;
            if (beta != 0.0f) {
                val += beta * y[i];
            }
            if (act != ActivationType::kNone) {
                val = fast_math::fast_activate(act, val);
            }
            y[i] = val;
        }
    }

    // ------------------------------------------------------------------------
    // Fast Quantized Matrix-Vector Multiplication (Q4_1 weights x FP32 vector):
    // ------------------------------------------------------------------------
    static void gemv_q4_1_fp32(const FastQuant::BlockQ4_1* __restrict__ A,
                              const float* __restrict__ x,
                              float* __restrict__ y,
                              int64_t M, int64_t K,
                              float alpha = 1.0f, float beta = 0.0f,
                              ActivationType act = ActivationType::kNone) {
        if (M <= 0 || K <= 0) return;
        const size_t num_blocks = (K + 31) / 32;

        FastQuant::BlockQ8_1 stack_q8[256];
        std::vector<FastQuant::BlockQ8_1> heap_q8;
        FastQuant::BlockQ8_1* q8_vec = (num_blocks <= 256)
            ? stack_q8
            : (heap_q8.resize(num_blocks), heap_q8.data());

        FastQuant::quantize_q8_1(x, q8_vec, num_blocks);
        gemv_q4_1_q8_1(A, q8_vec, y, M, K, alpha, beta, act);
    }

    // ------------------------------------------------------------------------
    // Fast Quantized Matrix-Vector Multiplication (Q5_0 weights x Q8_0 activations):
    // A: [M, K/32] BlockQ5_0, x: [K/32] BlockQ8_0, y: [M] float
    // ------------------------------------------------------------------------
    static void gemv_q5_0_q8_0(const FastQuant::BlockQ5_0* __restrict__ A,
                              const FastQuant::BlockQ8_0* __restrict__ x,
                              float* __restrict__ y,
                              int64_t M, int64_t K,
                              float alpha = 1.0f, float beta = 0.0f,
                              ActivationType act = ActivationType::kNone) {
        if (M <= 0 || K <= 0) return;
        const int64_t num_blocks = (K + 31) / 32;

        #pragma omp parallel for schedule(static) if (M * K >= fast_parallel::kGemvMinMacs)
        for (int64_t i = 0; i < M; ++i) {
            const FastQuant::BlockQ5_0* row = A + i * num_blocks;
            float acc = 0.0f;

            for (int64_t b = 0; b < num_blocks; ++b) {
                acc += dot_q5_0_q8_0_block(row[b], x[b]);
            }

            float val = alpha * acc;
            if (beta != 0.0f) {
                val += beta * y[i];
            }
            if (act != ActivationType::kNone) {
                val = fast_math::fast_activate(act, val);
            }
            y[i] = val;
        }
    }

    // ------------------------------------------------------------------------
    // Fast Quantized Matrix-Vector Multiplication (Q5_0 weights x FP32 vector):
    // ------------------------------------------------------------------------
    static void gemv_q5_0_fp32(const FastQuant::BlockQ5_0* __restrict__ A,
                              const float* __restrict__ x,
                              float* __restrict__ y,
                              int64_t M, int64_t K,
                              float alpha = 1.0f, float beta = 0.0f,
                              ActivationType act = ActivationType::kNone) {
        if (M <= 0 || K <= 0) return;
        const size_t num_blocks = (K + 31) / 32;

        FastQuant::BlockQ8_0 stack_q8[256];
        std::vector<FastQuant::BlockQ8_0> heap_q8;
        FastQuant::BlockQ8_0* q8_vec = (num_blocks <= 256)
            ? stack_q8
            : (heap_q8.resize(num_blocks), heap_q8.data());

        FastQuant::quantize_q8_0_with_sum(x, q8_vec, num_blocks);
        gemv_q5_0_q8_0(A, q8_vec, y, M, K, alpha, beta, act);
    }
};

} // namespace strata::kernels
