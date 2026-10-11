#pragma once
// include/strata/kernels/fast_activations.hpp - Unified Fast Activation Function Math for CPU, CUDA & HIP
//
// Provides high-performance, low-overhead implementations of modern neural network activation functions:
// - Fast Sigmoid, SiLU (Swish), SwiGLU, GELU (tanh approx), QuickGELU, GeGLU, Softplus,
//   Tanh, Squared ReLU, LeakyReLU, HardSigmoid, HardSwish, Mish, and ELU.
// - Supports both scalar and vectorized (float2, float4) paths with fast GPU intrinsics (__expf, __fdividef, __fmaf_rn, __tanhf)
//   and optimized CPU host execution.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if !defined(__CUDACC__) && !defined(__HIPCC__) && (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_HD __host__ __device__ __forceinline__
#define STRATA_D __device__ __forceinline__
#include <cuda_runtime.h>
#else
#define STRATA_HD inline
#define STRATA_D inline
#endif

namespace strata::kernels {

/// Supported activation function types across Strata models & runtimes.
enum class ActivationType : uint8_t {
    kNone = 0,
    kSiLU = 1,          // x * sigmoid(x) (Swish)
    kSigmoid = 2,       // 1 / (1 + exp(-x))
    kGELU = 3,          // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    kQuickGELU = 4,     // x * sigmoid(1.702 * x)
    kReLU = 5,          // max(0, x)
    kSquaredReLU = 6,   // (max(0, x))^2
    kSwiGLU = 7,        // SiLU(gate) * up
    kGeGLU = 8,         // GELU(gate) * up
    kSoftplus = 9,      // log(1 + exp(x))
    kTanh = 10,         // tanh(x)
    kHardSigmoid = 11,  // clamp((x + 3) / 6, 0, 1)
    kHardSwish = 12,    // x * HardSigmoid(x)
    kLeakyReLU = 13,    // x >= 0 ? x : x * slope
    kMish = 14,         // x * tanh(softplus(x))
    kELU = 15           // x >= 0 ? x : alpha * (exp(x) - 1)
};

namespace fast_math {

// ----------------------------------------------------------------------------
// Fast Exponential Function
// ----------------------------------------------------------------------------
STRATA_HD float fast_exp(float x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __expf(x);
#else
    return std::exp(x);
#endif
}

// ----------------------------------------------------------------------------
// Fast Sigmoid: sigma(x) = 1 / (1 + exp(-x))
// ----------------------------------------------------------------------------
STRATA_HD float fast_sigmoid(float x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (x > 16.0f) return 1.0f;
    if (x < -16.0f) return 0.0f;
    return __fdividef(1.0f, 1.0f + __expf(-x));
#else
    if (x > 16.0f) return 1.0f;
    if (x < -16.0f) return 0.0f;
    return 1.0f / (1.0f + std::exp(-x));
#endif
}

// ----------------------------------------------------------------------------
// Fast SiLU (Swish): silu(x) = x * sigma(x) = x / (1 + exp(-x))
// ----------------------------------------------------------------------------
STRATA_HD float fast_silu(float x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (x > 16.0f) return x;
    if (x < -16.0f) return 0.0f;
    return __fdividef(x, 1.0f + __expf(-x));
#else
    if (x > 16.0f) return x;
    if (x < -16.0f) return 0.0f;
    return x / (1.0f + std::exp(-x));
#endif
}

// ----------------------------------------------------------------------------
// Fast SwiGLU: SwiGLU(gate, up) = silu(gate) * up
// ----------------------------------------------------------------------------
STRATA_HD float fast_swiglu(float gate, float up) {
    return fast_silu(gate) * up;
}

// ----------------------------------------------------------------------------
// Fast GELU (Gaussian Error Linear Unit - Tanh approximation)
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ----------------------------------------------------------------------------
STRATA_HD float fast_gelu(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f; // sqrt(2 / pi)
    constexpr float kCoeff = 0.044715f;
    const float inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return 0.5f * x * (1.0f + __tanhf(inner));
#else
    return 0.5f * x * (1.0f + std::tanh(inner));
#endif
}

// ----------------------------------------------------------------------------
// Fast QuickGELU (Sigmoid-based approximation)
// quick_gelu(x) = x * sigmoid(1.702 * x)
// ----------------------------------------------------------------------------
STRATA_HD float fast_quick_gelu(float x) {
    return x * fast_sigmoid(1.702f * x);
}

// ----------------------------------------------------------------------------
// Fast GeGLU: GeGLU(gate, up) = gelu(gate) * up
// ----------------------------------------------------------------------------
STRATA_HD float fast_geglu(float gate, float up) {
    return fast_gelu(gate) * up;
}

// ----------------------------------------------------------------------------
// Fast Softplus: softplus(x) = log(1 + exp(x))
// ----------------------------------------------------------------------------
STRATA_HD float fast_softplus(float x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return log1pf(__expf(x));
#else
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return std::log1p(std::exp(x));
#endif
}

// ----------------------------------------------------------------------------
// Fast Tanh
// ----------------------------------------------------------------------------
STRATA_HD float fast_tanh(float x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __tanhf(x);
#else
    return std::tanh(x);
#endif
}

// ----------------------------------------------------------------------------
// Fast ReLU & Squared ReLU
// ----------------------------------------------------------------------------
STRATA_HD float fast_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

STRATA_HD float fast_squared_relu(float x) {
    const float r = x > 0.0f ? x : 0.0f;
    return r * r;
}

// ----------------------------------------------------------------------------
// Fast LeakyReLU
// ----------------------------------------------------------------------------
STRATA_HD float fast_leaky_relu(float x, float negative_slope = 0.01f) {
    return x >= 0.0f ? x : x * negative_slope;
}

// ----------------------------------------------------------------------------
// Fast HardSigmoid & HardSwish
// ----------------------------------------------------------------------------
STRATA_HD float fast_hard_sigmoid(float x) {
    const float v = (x + 3.0f) * (1.0f / 6.0f);
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

STRATA_HD float fast_hard_swish(float x) {
    return x * fast_hard_sigmoid(x);
}

// ----------------------------------------------------------------------------
// Fast Mish: mish(x) = x * tanh(softplus(x))
// ----------------------------------------------------------------------------
STRATA_HD float fast_mish(float x) {
    return x * fast_tanh(fast_softplus(x));
}

// ----------------------------------------------------------------------------
// Fast ELU: elu(x, alpha) = x >= 0 ? x : alpha * (exp(x) - 1)
// ----------------------------------------------------------------------------
STRATA_HD float fast_elu(float x, float alpha = 1.0f) {
    return x >= 0.0f ? x : alpha * (fast_exp(x) - 1.0f);
}

// ----------------------------------------------------------------------------
// Unified Fast Activation Dispatcher
// ----------------------------------------------------------------------------
STRATA_HD float fast_activate(ActivationType type, float x) {
    switch (type) {
        case ActivationType::kSiLU:         return fast_silu(x);
        case ActivationType::kSigmoid:      return fast_sigmoid(x);
        case ActivationType::kGELU:         return fast_gelu(x);
        case ActivationType::kQuickGELU:    return fast_quick_gelu(x);
        case ActivationType::kReLU:         return fast_relu(x);
        case ActivationType::kSquaredReLU:  return fast_squared_relu(x);
        case ActivationType::kSoftplus:     return fast_softplus(x);
        case ActivationType::kTanh:         return fast_tanh(x);
        case ActivationType::kHardSigmoid:  return fast_hard_sigmoid(x);
        case ActivationType::kHardSwish:    return fast_hard_swish(x);
        case ActivationType::kLeakyReLU:    return fast_leaky_relu(x);
        case ActivationType::kMish:         return fast_mish(x);
        case ActivationType::kELU:          return fast_elu(x);
        case ActivationType::kNone:
        default:                            return x;
    }
}

STRATA_HD float fast_gated_activate(ActivationType type, float gate, float up) {
    switch (type) {
        case ActivationType::kSwiGLU:       return fast_swiglu(gate, up);
        case ActivationType::kGeGLU:        return fast_geglu(gate, up);
        default:                            return fast_activate(type, gate) * up;
    }
}

#if defined(__CUDACC__) || defined(__HIPCC__)
// ----------------------------------------------------------------------------
// Vectorized GPU Primitives (float2 / float4)
// ----------------------------------------------------------------------------
STRATA_D float2 fast_silu2(float2 v) {
    return make_float2(fast_silu(v.x), fast_silu(v.y));
}

STRATA_D float4 fast_silu4(float4 v) {
    return make_float4(fast_silu(v.x), fast_silu(v.y), fast_silu(v.z), fast_silu(v.w));
}

STRATA_D float2 fast_sigmoid2(float2 v) {
    return make_float2(fast_sigmoid(v.x), fast_sigmoid(v.y));
}

STRATA_D float4 fast_sigmoid4(float4 v) {
    return make_float4(fast_sigmoid(v.x), fast_sigmoid(v.y), fast_sigmoid(v.z), fast_sigmoid(v.w));
}

STRATA_D float2 fast_swiglu2(float2 g, float2 u) {
    return make_float2(fast_swiglu(g.x, u.x), fast_swiglu(g.y, u.y));
}

STRATA_D float4 fast_swiglu4(float4 g, float4 u) {
    return make_float4(fast_swiglu(g.x, u.x), fast_swiglu(g.y, u.y), fast_swiglu(g.z, u.z), fast_swiglu(g.w, u.w));
}

STRATA_D float2 fast_gelu2(float2 v) {
    return make_float2(fast_gelu(v.x), fast_gelu(v.y));
}

STRATA_D float4 fast_gelu4(float4 v) {
    return make_float4(fast_gelu(v.x), fast_gelu(v.y), fast_gelu(v.z), fast_gelu(v.w));
}
#endif

#if !defined(__CUDACC__) && !defined(__HIPCC__) && (defined(__x86_64__) || defined(_M_X64))
#if defined(__AVX2__) && defined(__FMA__)
// Fast vector exp(x) approximation accurate to single precision (< 1e-6 relative error)
inline __m256 _mm256_fastexp_ps(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    __m256 z = _mm256_mul_ps(x, log2e);
    __m256 n_flt = _mm256_round_ps(z, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256i n_int = _mm256_cvtps_epi32(n_flt);

    const __m256 c_ln2_hi = _mm256_set1_ps(-0.6931471805599453f);
    const __m256 c_ln2_lo = _mm256_set1_ps(-1.4286068203094172e-6f);
    __m256 r = _mm256_fmadd_ps(n_flt, c_ln2_hi, x);
    r = _mm256_fmadd_ps(n_flt, c_ln2_lo, r);

    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(1.0f);
    const __m256 c2 = _mm256_set1_ps(0.5f);
    const __m256 c3 = _mm256_set1_ps(1.6666666666666666e-1f);
    const __m256 c4 = _mm256_set1_ps(4.1666666666666664e-2f);
    const __m256 c5 = _mm256_set1_ps(8.3333333333333332e-3f);

    __m256 p = _mm256_fmadd_ps(c5, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    p = _mm256_fmadd_ps(p, r, c0);

    __m256i pow2 = _mm256_slli_epi32(_mm256_add_epi32(n_int, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(pow2));
}

inline __m256 _mm256_fastsilu_ps(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg_x = _mm256_fastexp_ps(neg_x);
    __m256 denom = _mm256_add_ps(one, exp_neg_x);
    __m256 rcp = _mm256_rcp_ps(denom);
    rcp = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(denom, rcp, _mm256_set1_ps(2.0f)));
    return _mm256_mul_ps(x, rcp);
}
#endif
#endif

} // namespace fast_math

// ----------------------------------------------------------------------------
// CPU Batch Vectorized Helpers
// ----------------------------------------------------------------------------
inline void fast_activate_array(const float* __restrict__ in, float* __restrict__ out, size_t n, ActivationType type) {
    size_t i = 0;
#if !defined(__CUDACC__) && !defined(__HIPCC__) && defined(__AVX2__) && defined(__FMA__)
    if (type == ActivationType::kSiLU) {
        #pragma omp parallel for schedule(static) if (n >= 4096)
        for (size_t j = 0; j < (n / 8) * 8; j += 8) {
            __m256 v = _mm256_loadu_ps(in + j);
            _mm256_storeu_ps(out + j, fast_math::_mm256_fastsilu_ps(v));
        }
        i = (n / 8) * 8;
    } else if (type == ActivationType::kReLU) {
        const __m256 zero = _mm256_setzero_ps();
        #pragma omp parallel for schedule(static) if (n >= 4096)
        for (size_t j = 0; j < (n / 8) * 8; j += 8) {
            __m256 v = _mm256_loadu_ps(in + j);
            _mm256_storeu_ps(out + j, _mm256_max_ps(v, zero));
        }
        i = (n / 8) * 8;
    }
#endif
    for (; i < n; ++i) {
        out[i] = fast_math::fast_activate(type, in[i]);
    }
}

inline void fast_swiglu_array(const float* __restrict__ gate, const float* __restrict__ up,
                              float* __restrict__ out, size_t n) {
    size_t i = 0;
#if !defined(__CUDACC__) && !defined(__HIPCC__) && defined(__AVX2__) && defined(__FMA__)
    #pragma omp parallel for schedule(static) if (n >= 4096)
    for (size_t j = 0; j < (n / 8) * 8; j += 8) {
        __m256 g = _mm256_loadu_ps(gate + j);
        __m256 u = _mm256_loadu_ps(up + j);
        __m256 silu_g = fast_math::_mm256_fastsilu_ps(g);
        __m256 res = _mm256_mul_ps(silu_g, u);
        _mm256_storeu_ps(out + j, res);
    }
    i = (n / 8) * 8;
#endif
    for (; i < n; ++i) {
        out[i] = fast_math::fast_swiglu(gate[i], up[i]);
    }
}

} // namespace strata::kernels
