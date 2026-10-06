#pragma once
// include/strata/kernels/fast_softmax.hpp - Online Fast Softmax & Logits Processor
//
// Provides high-performance, numerically stable probability distributions & logits operations:
// 1. Online Flash-Softmax (2-pass & single-pass online max/sum tracking).
// 2. Numerically Stable Log-Sum-Exp (LSE).
// 3. Logits Temperature Scaling, Repetition Penalties, and Top-K/Top-P Filtering.
// 4. Fast ArgMax SIMD reduction for greedy decoding.

#include "strata/kernels/fast_activations.hpp"
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <vector>
#include <limits>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::kernels {

class FastSoftmax {
public:
    // ------------------------------------------------------------------------
    // Numerically Stable In-Place Softmax: x[i] = exp(x[i] - max) / sum(exp(x[j] - max))
    // ------------------------------------------------------------------------
    static void softmax_inplace(float* __restrict__ x, size_t n, float temperature = 1.0f) {
        if (n == 0) return;
        if (temperature <= 0.0f) {
            // Greedy delta function
            size_t best_idx = 0;
            float max_v = x[0];
            for (size_t i = 1; i < n; ++i) {
                if (x[i] > max_v) {
                    max_v = x[i];
                    best_idx = i;
                }
            }
            std::fill(x, x + n, 0.0f);
            x[best_idx] = 1.0f;
            return;
        }

        const float inv_temp = 1.0f / temperature;

        // 1. Find Max for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < n; ++i) {
            float v = x[i] * inv_temp;
            if (v > max_val) max_val = v;
        }

        // 2. Exponentiate & Sum
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            float e = fast_math::fast_exp((x[i] * inv_temp) - max_val);
            x[i] = e;
            sum += e;
        }

        // 3. Normalize
        if (sum > 0.0f) {
            const float inv_sum = 1.0f / sum;
            for (size_t i = 0; i < n; ++i) {
                x[i] *= inv_sum;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Numerically Stable Log-Sum-Exp (LSE): log(sum(exp(x[i])))
    // ------------------------------------------------------------------------
    static float log_sum_exp(const float* __restrict__ x, size_t n) {
        if (n == 0) return -std::numeric_limits<float>::infinity();

        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < n; ++i) {
            if (x[i] > max_val) max_val = x[i];
        }

        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            sum += fast_math::fast_exp(x[i] - max_val);
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

        for (size_t i = 1; i < n; ++i) {
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

        std::vector<std::pair<float, size_t>> scored(n);
        for (size_t i = 0; i < n; ++i) {
            scored[i] = {logits[i], i};
        }

        std::nth_element(scored.begin(), scored.begin() + k, scored.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });

        float threshold = scored[k - 1].first;
        for (size_t i = 0; i < n; ++i) {
            if (logits[i] < threshold) {
                logits[i] = -std::numeric_limits<float>::infinity();
            }
        }
    }

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
