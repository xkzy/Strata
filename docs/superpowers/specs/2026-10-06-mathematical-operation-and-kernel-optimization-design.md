# Mathematical Operation and Kernel Optimization (Section 19) Design Specification

**Date:** 2026-10-06  
**Status:** Approved  
**Scope:** Strata C++ Core Engine & Fast Kernel Architecture  

---

## 1. Objective & Design Goals

Optimize Strata's mathematical operations aggressively while preserving algorithmic correctness and configurable numerical semantics.

The core design goals are:
- **Minimize Overhead**: Reduce computation, memory traffic, synchronization, and unnecessary conversions while maximizing mathematical throughput.
- **Explicit Numerical Policies**: Support `STRICT` (exact standard IEEE-754 semantics), `BALANCED` (high-accuracy SIMD/FMA vectorization), and `FAST` (fast intrinsics, aggressive fusion, approximate activations where explicitly enabled).
- **Zero Unintended Precision Loss**: Never silently reduce precision; preserve FP32 accumulators during all reductions and matrix inner-products.
- **Operation Fusion**: Combine adjacent operations into single-pass kernels (e.g., Fused RMSNorm + Residual, Fused SwiGLU GEMM, Fused Attention QK scaling + Masking + Softmax, Fused MoE Routing + Top-K).
- **Cost-Model Kernel Dispatching & Auto-Tuning**: Dynamic selection of specialized kernels based on shape ($M, N, K$), batch size, data type, hardware capabilities, and numerical policy.
- **Deterministic Verification**: Verify numerical parity against reference scalar baselines across normal and edge cases (NaN, Inf, zeros, extreme magnitudes).

---

## 2. Architecture & Component Structure

### 2.1 Numerical Policies (`include/strata/kernels/math_policy.hpp`)

```cpp
namespace strata::kernels {

enum class MathPolicy : uint8_t {
    kStrict = 0,    // Exact IEEE-754, standard math, no approximations
    kBalanced = 1,  // High-accuracy SIMD vectorization (AVX2/AVX-512/FMA), Newton-Raphson refinement
    kFast = 2       // Fast hardware intrinsics (__expf, __fdividef), aggressive fusion, fast activations
};

struct MathContext {
    MathPolicy policy = MathPolicy::kBalanced;
    float rsqrt_epsilon = 1e-6f;
    float softmax_max_clamp = 88.0f;
    bool allow_fused_residual = true;
    bool allow_approximate_gelu = true;
    bool allow_fast_exp = false;
};

} // namespace strata::kernels
```

### 2.2 Kernel Dispatcher & Registry (`include/strata/kernels/kernel_dispatcher.hpp`)

```cpp
namespace strata::kernels {

enum class OpType : uint16_t {
    kMatMul = 0,
    kGEMV = 1,
    kRMSNorm = 2,
    kFusedNormResidual = 3,
    kAttention = 4,
    kRoPE = 5,
    kSwiGLU = 6,
    kMoERouting = 7,
    kSoftmax = 8,
    kQuantize = 9,
    kDequantize = 10
};

struct KernelKey {
    OpType op;
    int64_t M, N, K;
    int32_t dtype;
    MathPolicy policy;

    bool operator==(const KernelKey& o) const {
        return op == o.op && M == o.M && N == o.N && K == o.K &&
               dtype == o.dtype && policy == o.policy;
    }
};

class KernelDispatcher {
public:
    static KernelDispatcher& instance();
    void set_default_policy(MathPolicy policy);
    MathPolicy default_policy() const;
    void auto_tune_warmup();
};

} // namespace strata::kernels
```

### 2.3 Kernel Modules

1. **`fast_norm.hpp`**:
   - `rmsnorm(const float* x, const float* weight, float* out, size_t dim, float eps, MathPolicy policy)`
   - `fused_rmsnorm_residual(float* residual, const float* x, const float* weight, float* out, size_t dim, float eps, MathPolicy policy)`
   - `layernorm(const float* x, const float* gamma, const float* beta, float* out, size_t dim, float eps)`
   - `gemma_rmsnorm(const float* x, const float* weight, float* out, size_t dim, float eps)`
   - `per_head_qk_norm(float* qk_heads, size_t num_heads, size_t head_dim, float eps)`

2. **`fast_matmul.hpp`**:
   - `gemv_fp32`: Optimized row-split SIMD GEMV for $M \in [1, 8]$ (decode phase).
   - `gemm_fp32`: Tiled cache-blocked general matrix multiplication.
   - `fused_swiglu_gemm`: $\text{SiLU}(x \cdot W_{gate}) \odot (x \cdot W_{up})$ in single pass.
   - `gemv_int8_vnni` / `gemv_q8_0`: Quantized dot product acceleration.

3. **`fast_activations.hpp`**:
   - Vectorized activations: SiLU, SwiGLU, GELU (exact and fast tanh approximation), GeGLU, Sigmoid, QuickGELU, Softplus, Mish, HardSwish.

4. **`fast_rope.hpp`**:
   - `compute_inv_freqs`: Precomputed invariant frequency tables.
   - `apply_multi_head_neox`: Vectorized NeoX complex rotary layout.
   - `apply_multi_head_standard`: Vectorized standard pair-wise rotary layout.

5. **`fast_attention.hpp`**:
   - `scaled_dot_product_attention_decode`: Incremental single-token query over historical KV cache blocks with fused scaling and causal softmax.
   - `flash_attention_prefill`: Tiled block attention for prefill context.

6. **`fast_moe.hpp`**:
   - `route_topk`: Softmax logits + partial selection Top-K ($O(E + K \log K)$).
   - `dispatch_permutation`: Token permutation tables for expert dispatch without redundant tensor allocations.
   - `weighted_expert_aggregate`: Vectorized aggregation of expert output vectors.

7. **`fast_softmax.hpp`**:
   - `softmax_inplace`: Stable online single-pass softmax with temperature scaling.
   - `argmax`: Vectorized argmax index extraction.
   - `top_k_top_p_filter`: Combined threshold filtering and sampling.

8. **`fast_quant.hpp`**:
   - `quantize_q8_0` / `dequantize_q8_0`: Fast block quantization with SIMD scale application.
   - `quantize_fp8` / `dequantize_fp8`: FP8 conversions.

---

## 3. Data Flow & Numerical Invariants

- **Single Memory Pass Pipelines**: Eliminate intermediate tensor allocations between Norm, RoPE, Attention, Activation, and MoE routing.
- **FP32 Accumulators**: All inner products, reductions, variances, and softmax denominators accumulate in FP32 precision to prevent loss of significance.
- **Bounds Checking & Sanitization**: Softmax and exponentiation sanitize extreme values ($|x| > 88.0f$), preventing $\pm\infty$ and NaN propagation.
- **Isolated Workspaces**: Precomputed tables and intermediate buffers are strictly scoped to model/session instances to ensure multi-tenant safety.

---

## 4. Testing & Verification

1. **Parity Testing (`tests/kernels/test_fast_math_parity.cpp`)**:
   - Side-by-side comparison of optimized kernels against scalar reference implementations.
   - Maximum absolute error $\le 10^{-5}$ for `BALANCED`, exact bit equivalence where required in `STRICT`.
   - Edge cases: Zeros, negative numbers, extreme magnitudes, NaNs, and Infs.
2. **Unified Benchmarks (`tests/kernels/test_fast_unified_benchmark.cpp`)**:
   - 32-layer forward transformer simulation timing.
   - 152k-vocabulary softmax and sampling latency verification.
   - Online loop detector latency overhead bound ($< 5\,\mu\text{s}$ per token).
