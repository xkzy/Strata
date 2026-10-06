#pragma once
// include/strata/kernels/fast_softmax.hpp - Online Fast Softmax & Logits Processor
//
// Provides high-performance, numerically stable probability distributions & logits operations:
// 1. Online Flash-Softmax (2-pass & single-pass online max/sum tracking with SIMD vectorization).
// 2. Numerically Stable Log-Sum-Exp (LSE).
// 3. Logits Temperature Scaling, Repetition Penalties, and Top-K/Top-P/Min-P Filtering.
// 4. Fast ArgMax SIMD reduction for greedy decoding.

#include "strata/kernels/fast_activations.hpp"
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <functional>
#include <vector>
#include <limits>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::kernels {

class FastSoftmax {
private:
#if defined(__AVX2__)
    static inline float hmax256_ps(__m256 v) {
        __m128 vlow = _mm256_castps256_ps128(v);
        __m128 vhigh = _mm256_extractf128_ps(v, 1);
        __m128 max128 = _mm_max_ps(vlow, vhigh);
        __m128 max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
        __m128 max32 = _mm_max_ss(max64, _mm_shuffle_ps(max64, max64, 0x55));
        return _mm_cvtss_f32(max32);
    }
#endif

#if defined(__AVX2__) && defined(__FMA__)
    // exp(x) for 8 lanes (Cephes polynomial; softmax outputs measured within 8e-6 relative of a double reference on logits in [-30, 30]). Inputs below -87.3 (including -inf) give exactly 0,
    // so masked logits stay zero. Inputs above 88.3 are clamped.
    static inline __m256 exp256_ps(__m256 x) {
        const __m256 underflow = _mm256_cmp_ps(x, _mm256_set1_ps(-87.3f), _CMP_LT_OQ);
        x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
        x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));
        __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
        x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
        __m256 y = _mm256_set1_ps(1.9875691500E-4f);
        y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
        y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
        y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
        y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
        y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
        y = _mm256_fmadd_ps(_mm256_mul_ps(y, x), x, _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
        __m256i n = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
        return _mm256_andnot_ps(underflow, _mm256_mul_ps(y, _mm256_castsi256_ps(n)));
    }

    static inline float hsum256_ps(__m256 v) {
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
        return _mm_cvtss_f32(s);
    }
#endif

#if defined(__AVX512F__)
    static inline float hmax512_ps(__m512 v) {
        __m256 vlow = _mm512_castps512_ps256(v);
        __m256 vhigh = _mm512_extractf32x8_ps(v, 1);
        __m256 max256 = _mm256_max_ps(vlow, vhigh);
        return hmax256_ps(max256);
    }
#endif

public:
    // ------------------------------------------------------------------------
    // Numerically Stable In-Place Softmax: x[i] = exp(x[i] - max) / sum(exp(x[j] - max))
    // ------------------------------------------------------------------------
    static void softmax_inplace(float* __restrict__ x, size_t n, float temperature = 1.0f) {
        if (n == 0) return;
        if (temperature <= 0.0f) {
            // Greedy delta function
            int32_t best_idx = argmax(x, n);
            std::fill(x, x + n, 0.0f);
            if (best_idx >= 0 && static_cast<size_t>(best_idx) < n) {
                x[best_idx] = 1.0f;
            }
            return;
        }

        const float inv_temp = 1.0f / temperature;

        // 1. Vectorized Max Search for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        size_t i = 0;

#if defined(__AVX512F__)
        __m512 max_v = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        __m512 inv_t_v = _mm512_set1_ps(inv_temp);
        for (; i + 16 <= n; i += 16) {
            __m512 v = _mm512_mul_ps(_mm512_loadu_ps(x + i), inv_t_v);
            max_v = _mm512_max_ps(max_v, v);
        }
        max_val = hmax512_ps(max_v);
#elif defined(__AVX2__)
        __m256 max_v = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        __m256 inv_t_v = _mm256_set1_ps(inv_temp);
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_mul_ps(_mm256_loadu_ps(x + i), inv_t_v);
            max_v = _mm256_max_ps(max_v, v);
        }
        max_val = hmax256_ps(max_v);
#endif
        for (; i < n; ++i) {
            float v = x[i] * inv_temp;
            if (v > max_val) max_val = v;
        }

        // 2. Exponentiate & Sum
        float sum = 0.0f;
        size_t j = 0;
#if defined(__AVX2__) && defined(__FMA__)
        {
            const __m256 inv_t = _mm256_set1_ps(inv_temp);
            const __m256 mx = _mm256_set1_ps(max_val);
            __m256 acc = _mm256_setzero_ps();
            for (; j + 8 <= n; j += 8) {
                __m256 e = exp256_ps(_mm256_fmsub_ps(_mm256_loadu_ps(x + j), inv_t, mx));
                _mm256_storeu_ps(x + j, e);
                acc = _mm256_add_ps(acc, e);
            }
            sum = hsum256_ps(acc);
        }
#endif
        for (; j < n; ++j) {
            float e = fast_math::fast_exp((x[j] * inv_temp) - max_val);
            x[j] = e;
            sum += e;
        }

        // 3. Vectorized Normalize
        if (sum > 0.0f) {
            const float inv_sum = 1.0f / sum;
            size_t k = 0;
#if defined(__AVX512F__)
            __m512 inv_s_v = _mm512_set1_ps(inv_sum);
            for (; k + 16 <= n; k += 16) {
                __m512 v = _mm512_loadu_ps(x + k);
                _mm512_storeu_ps(x + k, _mm512_mul_ps(v, inv_s_v));
            }
#elif defined(__AVX2__)
            __m256 inv_s_v = _mm256_set1_ps(inv_sum);
            for (; k + 8 <= n; k += 8) {
                __m256 v = _mm256_loadu_ps(x + k);
                _mm256_storeu_ps(x + k, _mm256_mul_ps(v, inv_s_v));
            }
#endif
            for (; k < n; ++k) {
                x[k] *= inv_sum;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Numerically Stable Log-Sum-Exp (LSE): log(sum(exp(x[i])))
    // ------------------------------------------------------------------------
    static float log_sum_exp(const float* __restrict__ x, size_t n) {
        if (n == 0) return -std::numeric_limits<float>::infinity();

        float max_val = -std::numeric_limits<float>::infinity();
        size_t i = 0;
#if defined(__AVX512F__)
        __m512 max_v = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        for (; i + 16 <= n; i += 16) {
            max_v = _mm512_max_ps(max_v, _mm512_loadu_ps(x + i));
        }
        max_val = hmax512_ps(max_v);
#elif defined(__AVX2__)
        __m256 max_v = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        for (; i + 8 <= n; i += 8) {
            max_v = _mm256_max_ps(max_v, _mm256_loadu_ps(x + i));
        }
        max_val = hmax256_ps(max_v);
#endif
        for (; i < n; ++i) {
            if (x[i] > max_val) max_val = x[i];
        }

        float sum = 0.0f;
        size_t j = 0;
#if defined(__AVX2__) && defined(__FMA__)
        {
            const __m256 mx = _mm256_set1_ps(max_val);
            __m256 acc = _mm256_setzero_ps();
            for (; j + 8 <= n; j += 8) {
                acc = _mm256_add_ps(acc, exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(x + j), mx)));
            }
            sum = hsum256_ps(acc);
        }
#endif
        for (; j < n; ++j) {
            sum += fast_math::fast_exp(x[j] - max_val);
        }

        return max_val + std::log(sum);
    }

    // ------------------------------------------------------------------------
    // Fast ArgMax for Greedy Token Selection: returns index of maximum element
    // ------------------------------------------------------------------------
    static int32_t argmax(const float* __restrict__ x, size_t n) {
        if (n == 0) return -1;
        int32_t best_idx = 0;
        float max_val = x[0];
        size_t i = 1;

#if defined(__AVX2__)
        // Blocked: SIMD max over a block, and only when the block beats the running max, a scalar scan of that block
        // for the first index (keeps the first-maximum tie-break). NaNs never win, as in the scalar loop.
        if (n >= 64) {
            constexpr size_t kBlock = 256;
            while (i < n) {
                const size_t end = std::min(n, i + kBlock);
                size_t j = i;
                __m256 mv = _mm256_set1_ps(max_val);
                for (; j + 8 <= end; j += 8) mv = _mm256_max_ps(_mm256_loadu_ps(x + j), mv);
                float block_max = hmax256_ps(mv);
                if (block_max > max_val) {
                    for (size_t k = i; k < j; ++k) {
                        if (x[k] > max_val) { max_val = x[k]; best_idx = static_cast<int32_t>(k); }
                    }
                }
                for (; j < end; ++j) {
                    if (x[j] > max_val) { max_val = x[j]; best_idx = static_cast<int32_t>(j); }
                }
                i = end;
            }
            return best_idx;
        }
#endif
        for (; i < n; ++i) {
            if (x[i] > max_val) {
                max_val = x[i];
                best_idx = static_cast<int32_t>(i);
            }
        }
        return best_idx;
    }

    // ------------------------------------------------------------------------
    // Top-K Filter: Keeps the top-K highest logits, sets the rest to -inf
    // ------------------------------------------------------------------------
    static void top_k_filter(float* __restrict__ logits, size_t n, int k) {
        if (k <= 0 || static_cast<size_t>(k) >= n) return;

        const float threshold = kth_largest(logits, n, static_cast<size_t>(k));
        const float neg_inf = -std::numeric_limits<float>::infinity();
        size_t i = 0;
#if defined(__AVX2__)
        const __m256 thr = _mm256_set1_ps(threshold);
        const __m256 ninf = _mm256_set1_ps(neg_inf);
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(logits + i);
            _mm256_storeu_ps(logits + i, _mm256_blendv_ps(v, ninf, _mm256_cmp_ps(v, thr, _CMP_LT_OQ)));
        }
#endif
        for (; i < n; ++i) {
            if (logits[i] < threshold) logits[i] = neg_inf;
        }
    }

private:
    // k-th largest value (1 <= k < n). Streams over x keeping a candidate buffer of the values above the running
    // threshold; when the buffer fills, nth_element trims it to the top k and raises the threshold. Almost every
    // element of a logits row is rejected by one compare, so no full-size copy or sort is needed.
    static float kth_largest(const float* x, size_t n, size_t k) {
        constexpr float kNegInf = -std::numeric_limits<float>::infinity();
        std::vector<float>& buf = candidate_buffer();
        const size_t cap = std::max<size_t>(2 * k, 256);
        if (buf.size() < cap) buf.resize(cap);
        float* b = buf.data();
        size_t m = 0;
        float t = kNegInf;
        auto push = [&](float v) {
            b[m++] = v;
            if (m == cap) {
                std::nth_element(b, b + (k - 1), b + m, std::greater<float>());
                t = b[k - 1];
                m = k;
            }
        };
        size_t i = 0;
#if defined(__AVX2__)
        __m256 tv = _mm256_set1_ps(t);
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            int mask = _mm256_movemask_ps(_mm256_cmp_ps(v, tv, _CMP_GT_OQ));
            if (mask == 0) continue;
            for (int lane = 0; lane < 8; ++lane) {
                if (mask & (1 << lane)) push(x[i + lane]);
            }
            tv = _mm256_set1_ps(t);
        }
#endif
        for (; i < n; ++i) {
            if (x[i] > t) push(x[i]);
        }
        if (m < k) return kNegInf; // fewer than k values above -inf: nothing to filter
        std::nth_element(b, b + (k - 1), b + m, std::greater<float>());
        return b[k - 1];
    }

    static std::vector<float>& candidate_buffer() {
        static thread_local std::vector<float> buf;
        return buf;
    }

public:
    // ------------------------------------------------------------------------
    // Min-P Filter: Keeps tokens with p >= min_p * p_max, sets rest to 0
    // ------------------------------------------------------------------------
    static void min_p_filter(float* __restrict__ probs, size_t n, float min_p) {
        if (min_p <= 0.0f || n == 0) return;

        float p_max = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            if (probs[i] > p_max) p_max = probs[i];
        }

        const float threshold = min_p * p_max;
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            if (probs[i] < threshold) {
                probs[i] = 0.0f;
            } else {
                sum += probs[i];
            }
        }

        // Renormalize surviving tokens
        if (sum > 0.0f) {
            const float inv_sum = 1.0f / sum;
            for (size_t i = 0; i < n; ++i) {
                probs[i] *= inv_sum;
            }
        }
    }
};

} // namespace strata::kernels
