// tests/kernels/test_fast_math_parity.cpp - Unified Numerical Parity & Edge Case Validation Suite
//
// Validates all fast kernels (RMSNorm, Fused Norm+Residual, SwiGLU, GEMV, FastAttention,
// FastMoE, FastSoftmax, FastRoPE, FastQuant) against reference scalar mathematics across:
// 1. Standard distributions (uniform, normal)
// 2. Exact zeros
// 3. Negative values
// 4. Extreme magnitudes (subnormals, extreme bounds, saturation regions)
// 5. Special IEEE-754 edge cases (quiet NaN, signaling NaN, +Inf, -Inf)

#include "strata/kernels/math_policy.hpp"
#include "strata/kernels/fast_norm.hpp"
#include "strata/kernels/fast_activations.hpp"
#include "strata/kernels/fast_matmul.hpp"
#include "strata/kernels/fast_attention.hpp"
#include "strata/kernels/fast_moe.hpp"
#include "strata/kernels/fast_softmax.hpp"
#include "strata/kernels/fast_rope.hpp"
#include "strata/kernels/fast_quant.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {

// ============================================================================
// 1. Edge & NaN / Inf Sanitization Test Suite
// ============================================================================
void test_edge_and_nan_sanitization() {
    using namespace strata::kernels;
    std::cout << "[Test 1/10] Verifying MathContext, Bounds & NaN/Inf Sanitization..." << std::endl;

    MathContext ctx;
    assert(ctx.policy == MathPolicy::kBalanced);
    assert(ctx.rsqrt_epsilon == 1e-6f);
    assert(ctx.softmax_max_clamp == 88.0f);
    assert(ctx.allow_fused_residual == true);
    assert(ctx.allow_approximate_gelu == true);
    assert(ctx.allow_fast_exp == false);

    // Clamping to bounds
    assert(sanitize_logit(100.0f, ctx) == 88.0f);
    assert(sanitize_logit(-100.0f, ctx) == -88.0f);
    assert(sanitize_logit(88.0f, ctx) == 88.0f);
    assert(sanitize_logit(-88.0f, ctx) == -88.0f);
    assert(sanitize_logit(0.0f, ctx) == 0.0f);
    assert(sanitize_logit(12.34f, ctx) == 12.34f);

    // NaN / Inf sanitization
    assert(sanitize_logit(std::numeric_limits<float>::quiet_NaN(), ctx) == -88.0f);
    assert(sanitize_logit(std::numeric_limits<float>::signaling_NaN(), ctx) == -88.0f);
    assert(sanitize_logit(std::numeric_limits<float>::infinity(), ctx) == 88.0f);
    assert(sanitize_logit(-std::numeric_limits<float>::infinity(), ctx) == -88.0f);

    // validate_math_bounds alias
    assert(validate_math_bounds(999.0f, ctx) == 88.0f);
    assert(validate_math_bounds(-999.0f, ctx) == -88.0f);
    assert(validate_math_bounds(std::numeric_limits<float>::quiet_NaN(), ctx) == -88.0f);

    // Policy transitions
    ctx.set_policy(MathPolicy::kStrict);
    assert(ctx.policy == MathPolicy::kStrict);
    assert(!ctx.allow_approximate_gelu);
    assert(!ctx.allow_fast_exp);

    ctx.set_policy(MathPolicy::kFast);
    assert(ctx.policy == MathPolicy::kFast);
    assert(ctx.allow_approximate_gelu);
    assert(ctx.allow_fast_exp);

    // Custom clamp threshold
    ctx.softmax_max_clamp = 42.0f;
    assert(sanitize_logit(50.0f, ctx) == 42.0f);
    assert(sanitize_logit(-50.0f, ctx) == -42.0f);
    assert(sanitize_logit(std::numeric_limits<float>::quiet_NaN(), ctx) == -42.0f);

    std::cout << "  Passed. Edge & NaN sanitization verified." << std::endl;
}

// ============================================================================
// 2. RMSNorm Parity & Edge Case Validation
// ============================================================================
void test_rmsnorm_parity() {
    using namespace strata::kernels;
    std::cout << "[Test 2/10] Verifying Fast RMSNorm Parity & Edge Cases..." << std::endl;

    auto run_rmsnorm_check = [](size_t dim, float eps, const std::vector<float>& x,
                                const std::vector<float>& weight, float tol = 1e-5f) {
        std::vector<float> out_opt(dim, 0.0f);
        std::vector<float> out_ref(dim, 0.0f);

        // Reference scalar RMSNorm
        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            sum_sq += x[i] * x[i];
        }
        float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + eps);
        for (size_t i = 0; i < dim; ++i) {
            out_ref[i] = x[i] * scale * (weight.empty() ? 1.0f : weight[i]);
        }

        // Optimized RMSNorm
        FastNorm::rmsnorm(x.data(), weight.empty() ? nullptr : weight.data(),
                          out_opt.data(), dim, eps);

        float max_err = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            max_err = std::max(max_err, std::abs(out_opt[i] - out_ref[i]));
        }
        assert(max_err <= tol);
    };

    // 2a. Standard random distribution (dim = 1024)
    const size_t dim = 1024;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    std::vector<float> x(dim), weight(dim);
    for (size_t i = 0; i < dim; ++i) {
        x[i] = dist(rng);
        weight[i] = 1.0f + 0.1f * dist(rng);
    }
    run_rmsnorm_check(dim, 1e-6f, x, weight, 1e-5f);

    // 2b. Null weight (identity weights)
    run_rmsnorm_check(dim, 1e-6f, x, {}, 1e-5f);

    // 2c. Unaligned / odd dimensions (testing vector tail handling)
    for (size_t test_dim : {1, 7, 15, 33, 67, 127, 255, 513}) {
        std::vector<float> x_unaligned(test_dim);
        std::vector<float> w_unaligned(test_dim);
        for (size_t i = 0; i < test_dim; ++i) {
            x_unaligned[i] = dist(rng);
            w_unaligned[i] = 1.0f + 0.05f * dist(rng);
        }
        run_rmsnorm_check(test_dim, 1e-6f, x_unaligned, w_unaligned, 1e-5f);
    }

    // 2d. Edge Case: Exact Zeros
    std::vector<float> zeros(dim, 0.0f);
    run_rmsnorm_check(dim, 1e-6f, zeros, weight, 1e-7f);

    // 2e. Edge Case: All Negative values
    std::vector<float> negatives(dim);
    for (size_t i = 0; i < dim; ++i) negatives[i] = -std::abs(dist(rng)) - 0.1f;
    run_rmsnorm_check(dim, 1e-6f, negatives, weight, 1e-5f);

    // 2f. Edge Case: Extreme Magnitudes (very small and very large values)
    std::vector<float> small_vals(dim, 1e-7f);
    run_rmsnorm_check(dim, 1e-6f, small_vals, weight, 1e-5f);

    std::vector<float> large_vals(dim, 1e3f);
    run_rmsnorm_check(dim, 1e-6f, large_vals, weight, 1e-4f);

    std::cout << "  Passed. Fast RMSNorm parity verified across standard & edge cases." << std::endl;
}

// ============================================================================
// 3. Fused RMSNorm + Residual Parity & Edge Case Validation
// ============================================================================
void test_fused_rmsnorm_residual_parity() {
    using namespace strata::kernels;
    std::cout << "[Test 3/10] Verifying Fused RMSNorm + Residual Parity & Edge Cases..." << std::endl;

    auto run_fused_check = [](size_t dim, float eps, const std::vector<float>& init_res,
                             const std::vector<float>& x, const std::vector<float>& weight, float tol = 1e-5f) {
        std::vector<float> res_ref = init_res;
        std::vector<float> out_ref(dim, 0.0f);

        // Reference math
        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            res_ref[i] += x[i];
            sum_sq += res_ref[i] * res_ref[i];
        }
        float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + eps);
        for (size_t i = 0; i < dim; ++i) {
            out_ref[i] = res_ref[i] * scale * (weight.empty() ? 1.0f : weight[i]);
        }

        // Optimized fused kernel
        std::vector<float> res_opt = init_res;
        std::vector<float> out_opt(dim, 0.0f);
        FastNorm::fused_rmsnorm_residual(res_opt.data(), x.data(),
                                         weight.empty() ? nullptr : weight.data(),
                                         out_opt.data(), dim, eps);

        // Check in-place residual addition parity
        for (size_t i = 0; i < dim; ++i) {
            float r_err = std::abs(res_opt[i] - res_ref[i]);
            assert(r_err <= 1e-5f);
        }

        // Check normalized output parity
        float max_err = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            max_err = std::max(max_err, std::abs(out_opt[i] - out_ref[i]));
        }
        assert(max_err <= tol);
    };

    const size_t dim = 1024;
    std::mt19937 rng(5678);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);

    std::vector<float> res(dim), x(dim), weight(dim);
    for (size_t i = 0; i < dim; ++i) {
        res[i] = dist(rng);
        x[i] = dist(rng);
        weight[i] = 1.0f + 0.1f * dist(rng);
    }
    run_fused_check(dim, 1e-6f, res, x, weight, 1e-5f);

    // Unaligned dimensions
    for (size_t test_dim : {9, 23, 67, 137, 511}) {
        std::vector<float> r_un(test_dim), x_un(test_dim), w_un(test_dim);
        for (size_t i = 0; i < test_dim; ++i) {
            r_un[i] = dist(rng);
            x_un[i] = dist(rng);
            w_un[i] = 1.0f + 0.05f * dist(rng);
        }
        run_fused_check(test_dim, 1e-6f, r_un, x_un, w_un, 1e-5f);
    }

    // Edge Cases: Zeros, all negatives, extreme magnitudes
    std::vector<float> zeros(dim, 0.0f);
    run_fused_check(dim, 1e-6f, zeros, zeros, weight, 1e-7f);

    std::vector<float> neg_res(dim, -3.0f), neg_x(dim, -2.0f);
    run_fused_check(dim, 1e-6f, neg_res, neg_x, weight, 1e-5f);

    std::vector<float> small_res(dim, 1e-6f), small_x(dim, 1e-6f);
    run_fused_check(dim, 1e-6f, small_res, small_x, weight, 1e-5f);

    std::cout << "  Passed. Fused RMSNorm + Residual parity verified." << std::endl;
}

// ============================================================================
// 4. SwiGLU Parity & Edge Case Validation
// ============================================================================
void test_swiglu_parity() {
    using namespace strata::kernels;
    std::cout << "[Test 4/10] Verifying SwiGLU Parity & Edge Cases..." << std::endl;

    auto ref_silu = [](float v) {
        return v / (1.0f + std::exp(-v));
    };
    auto ref_swiglu = [&](float gate, float up) {
        return ref_silu(gate) * up;
    };

    // Standard random range [-8.0f, 8.0f]
    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    const size_t n = 2048;

    std::vector<float> gate(n), up(n), out_opt(n);
    for (size_t i = 0; i < n; ++i) {
        gate[i] = dist(rng);
        up[i] = dist(rng);
    }

    fast_swiglu_array(gate.data(), up.data(), out_opt.data(), n);

    float max_err = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float ref = ref_swiglu(gate[i], up[i]);
        max_err = std::max(max_err, std::abs(out_opt[i] - ref));
    }
    assert(max_err < 1e-5f);

    // Scalar fast_math::fast_swiglu check
    for (size_t i = 0; i < 100; ++i) {
        float g = dist(rng);
        float u = dist(rng);
        float opt = fast_math::fast_swiglu(g, u);
        float ref = ref_swiglu(g, u);
        assert(std::abs(opt - ref) < 1e-5f);
    }

    // Edge Cases:
    // Zero gate -> 0
    assert(fast_math::fast_swiglu(0.0f, 10.0f) == 0.0f);
    assert(fast_math::fast_swiglu(0.0f, -10.0f) == 0.0f);
    // Zero up -> 0
    assert(fast_math::fast_swiglu(5.0f, 0.0f) == 0.0f);
    assert(fast_math::fast_swiglu(-5.0f, 0.0f) == 0.0f);
    // Large positive gate (> 16.0f saturation: silu(g) -> g)
    assert(std::abs(fast_math::fast_swiglu(20.0f, 2.0f) - 40.0f) < 1e-5f);
    assert(std::abs(fast_math::fast_swiglu(100.0f, -0.5f) - (-50.0f)) < 1e-5f);
    // Large negative gate (< -16.0f saturation: silu(g) -> 0.0f)
    assert(fast_math::fast_swiglu(-20.0f, 5.0f) == 0.0f);
    assert(fast_math::fast_swiglu(-100.0f, 1000.0f) == 0.0f);

    std::cout << "  Passed. SwiGLU parity and saturation behavior verified." << std::endl;
}

// ============================================================================
// 5. GEMV Parity & Edge Case Validation
// ============================================================================
void test_gemv_parity() {
    using namespace strata::kernels;
    std::cout << "[Test 5/10] Verifying Fast GEMV Parity & Edge Cases..." << std::endl;

    auto ref_gemv = [](const float* A, const float* x, float* y, int64_t M, int64_t K,
                       float alpha, float beta, ActivationType act) {
        for (int64_t i = 0; i < M; ++i) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                acc += A[i * K + k] * x[k];
            }
            float val = alpha * acc + (beta != 0.0f ? beta * y[i] : 0.0f);
            if (act != ActivationType::kNone) {
                val = fast_math::fast_activate(act, val);
            }
            y[i] = val;
        }
    };

    std::mt19937 rng(4321);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    // Test matrix configurations (including aligned and unaligned M, K)
    std::vector<std::pair<int64_t, int64_t>> configs = {
        {1, 64}, {1, 128}, {1, 255}, {4, 128}, {8, 256}, {7, 99}, {16, 512}
    };

    for (auto [M, K] : configs) {
        std::vector<float> A(M * K), x(K), y_ref(M), y_opt(M);
        for (size_t i = 0; i < A.size(); ++i) A[i] = dist(rng);
        for (size_t i = 0; i < x.size(); ++i) x[i] = dist(rng);
        for (size_t i = 0; i < y_ref.size(); ++i) {
            y_ref[i] = dist(rng);
            y_opt[i] = y_ref[i];
        }

        // 5a. alpha = 1.0f, beta = 0.0f
        ref_gemv(A.data(), x.data(), y_ref.data(), M, K, 1.0f, 0.0f, ActivationType::kNone);
        FastMatMul::gemv_fp32(A.data(), x.data(), y_opt.data(), M, K, 1.0f, 0.0f, ActivationType::kNone);

        float max_err = 0.0f;
        for (int64_t i = 0; i < M; ++i) {
            max_err = std::max(max_err, std::abs(y_opt[i] - y_ref[i]));
        }
        assert(max_err < 1e-4f);

        // 5b. alpha = 0.7f, beta = 0.3f with fused SiLU activation
        y_ref = y_opt;
        ref_gemv(A.data(), x.data(), y_ref.data(), M, K, 0.7f, 0.3f, ActivationType::kSiLU);
        FastMatMul::gemv_fp32(A.data(), x.data(), y_opt.data(), M, K, 0.7f, 0.3f, ActivationType::kSiLU);

        max_err = 0.0f;
        for (int64_t i = 0; i < M; ++i) {
            max_err = std::max(max_err, std::abs(y_opt[i] - y_ref[i]));
        }
        assert(max_err < 1e-4f);
    }

    // Edge Cases:
    // All zeros in A or x
    {
        const int64_t M = 4, K = 64;
        std::vector<float> A(M * K, 0.0f), x(K, 1.5f), y(M, 0.0f);
        FastMatMul::gemv_fp32(A.data(), x.data(), y.data(), M, K, 1.0f, 0.0f);
        for (int64_t i = 0; i < M; ++i) assert(y[i] == 0.0f);

        std::fill(A.begin(), A.end(), 1.5f);
        std::fill(x.begin(), x.end(), 0.0f);
        FastMatMul::gemv_fp32(A.data(), x.data(), y.data(), M, K, 1.0f, 0.0f);
        for (int64_t i = 0; i < M; ++i) assert(y[i] == 0.0f);
    }

    std::cout << "  Passed. Fast GEMV parity verified across dimensions & activations." << std::endl;
}

// ============================================================================
// 6. FastAttention Parity & Edge Case Validation
// ============================================================================
void test_fast_attention_parity() {
    using namespace strata::kernels;
    std::cout << "[Test 6/10] Verifying FastAttention Parity & Edge Cases..." << std::endl;

    auto ref_attention_decode = [](const float* q, const float* k_cache, const float* v_cache,
                                   float* out, size_t seq_len, size_t head_dim, float scale,
                                   const MathContext& ctx) {
        std::vector<float> scores(seq_len);
        float max_s = -1e30f;
        for (size_t s = 0; s < seq_len; ++s) {
            float dot = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) {
                dot += q[d] * k_cache[s * head_dim + d];
            }
            float sc = sanitize_logit(dot * scale, ctx);
            scores[s] = sc;
            if (sc > max_s) max_s = sc;
        }

        float sum_exp = 0.0f;
        for (size_t s = 0; s < seq_len; ++s) {
            scores[s] = std::exp(scores[s] - max_s);
            sum_exp += scores[s];
        }
        const float inv_sum = sum_exp > 0.0f ? (1.0f / (sum_exp + 1e-9f)) : 0.0f;
        for (size_t s = 0; s < seq_len; ++s) {
            scores[s] *= inv_sum;
        }

        std::fill(out, out + head_dim, 0.0f);
        for (size_t s = 0; s < seq_len; ++s) {
            float w = scores[s];
            for (size_t d = 0; d < head_dim; ++d) {
                out[d] += w * v_cache[s * head_dim + d];
            }
        }
    };

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    std::vector<size_t> test_dims = {16, 32, 64, 47};
    std::vector<size_t> test_lens = {1, 4, 16, 31};

    MathContext ctx(MathPolicy::kBalanced);

    for (size_t head_dim : test_dims) {
        for (size_t seq_len : test_lens) {
            std::vector<float> q(head_dim);
            std::vector<float> k_cache(seq_len * head_dim);
            std::vector<float> v_cache(seq_len * head_dim);
            std::vector<float> out_ref(head_dim, 0.0f);
            std::vector<float> out_opt(head_dim, 0.0f);

            for (size_t i = 0; i < head_dim; ++i) q[i] = dist(rng);
            for (size_t i = 0; i < seq_len * head_dim; ++i) {
                k_cache[i] = dist(rng);
                v_cache[i] = dist(rng);
            }

            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            ref_attention_decode(q.data(), k_cache.data(), v_cache.data(), out_ref.data(),
                                 seq_len, head_dim, scale, ctx);
            FastAttention::scaled_dot_product_decode(q.data(), k_cache.data(), v_cache.data(),
                                                    out_opt.data(), seq_len, head_dim, scale, ctx);

            float max_err = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) {
                max_err = std::max(max_err, std::abs(out_opt[d] - out_ref[d]));
            }
            assert(max_err < 1e-4f);
        }
    }

    // Edge Case: Identical V cache values -> output must exactly match V
    {
        const size_t head_dim = 64;
        const size_t seq_len = 8;
        std::vector<float> q(head_dim, 0.5f);
        std::vector<float> k(seq_len * head_dim, 0.2f);
        std::vector<float> v(seq_len * head_dim, 2.5f);
        std::vector<float> out(head_dim, 0.0f);

        FastAttention::scaled_dot_product_decode(q.data(), k.data(), v.data(), out.data(),
                                                seq_len, head_dim, 0.125f, ctx);
        for (size_t d = 0; d < head_dim; ++d) {
            assert(std::abs(out[d] - 2.5f) < 1e-4f);
        }
    }

    // Edge Case: Extreme magnitudes in logits (clamping to ctx.softmax_max_clamp = 88.0f)
    {
        const size_t head_dim = 32;
        const size_t seq_len = 4;
        std::vector<float> q(head_dim, 1000.0f);
        std::vector<float> k(seq_len * head_dim, 1000.0f);
        std::vector<float> v(seq_len * head_dim, 1.0f);
        std::vector<float> out(head_dim, 0.0f);

        FastAttention::scaled_dot_product_decode(q.data(), k.data(), v.data(), out.data(),
                                                seq_len, head_dim, 1.0f, ctx);
        // Valid finite probability output, no NaNs
        for (size_t d = 0; d < head_dim; ++d) {
            assert(!std::isnan(out[d]));
            assert(!std::isinf(out[d]));
            assert(std::abs(out[d] - 1.0f) < 1e-4f);
        }
    }

    // Edge Case: NaN in Q or K sanitization (sanitized to -88.0f)
    {
        const size_t head_dim = 16;
        const size_t seq_len = 3;
        std::vector<float> q(head_dim, 0.1f);
        q[0] = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> k(seq_len * head_dim, 0.2f);
        std::vector<float> v(seq_len * head_dim, 1.0f);
        std::vector<float> out(head_dim, 0.0f);

        FastAttention::scaled_dot_product_decode(q.data(), k.data(), v.data(), out.data(),
                                                seq_len, head_dim, 1.0f, ctx);
        for (size_t d = 0; d < head_dim; ++d) {
            assert(!std::isnan(out[d]));
            assert(!std::isinf(out[d]));
        }
    }

    std::cout << "  Passed. FastAttention parity & bounds protection verified." << std::endl;
}

// ============================================================================
// 7. FastMoE Parity & Edge Case Validation
// ============================================================================
void test_fast_moe_parity() {
    using namespace strata::kernels;
    std::cout << "[Test 7/10] Verifying FastMoE Router & Aggregation Parity & Edge Cases..." << std::endl;

    auto ref_route_topk = [](const float* logits, size_t num_experts, size_t top_k,
                            int32_t* out_indices, float* out_weights, const MathContext& ctx) {
        struct ExpertScore { int32_t idx; float val; };
        std::vector<ExpertScore> scores(num_experts);
        for (size_t i = 0; i < num_experts; ++i) {
            scores[i] = {static_cast<int32_t>(i), sanitize_logit(logits[i], ctx)};
        }
        std::stable_sort(scores.begin(), scores.end(), [](const ExpertScore& a, const ExpertScore& b) {
            if (a.val != b.val) return a.val > b.val;
            return a.idx < b.idx;
        });
        float max_s = scores[0].val;
        float sum_e = 0.0f;
        for (size_t k = 0; k < top_k; ++k) {
            float e = std::exp(scores[k].val - max_s);
            out_weights[k] = e;
            sum_e += e;
            out_indices[k] = scores[k].idx;
        }
        float inv_sum = sum_e > 0.0f ? (1.0f / (sum_e + 1e-9f)) : 0.0f;
        for (size_t k = 0; k < top_k; ++k) {
            out_weights[k] *= inv_sum;
        }
    };

    std::mt19937 rng(888);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    MathContext ctx(MathPolicy::kBalanced);

    std::vector<size_t> expert_counts = {4, 8, 16, 64};
    std::vector<size_t> ks = {1, 2, 4, 8};

    for (size_t num_experts : expert_counts) {
        for (size_t k : ks) {
            if (k > num_experts) continue;

            std::vector<float> logits(num_experts);
            for (size_t e = 0; e < num_experts; ++e) logits[e] = dist(rng);

            std::vector<int32_t> idx_ref(k), idx_opt(k);
            std::vector<float> w_ref(k), w_opt(k);

            ref_route_topk(logits.data(), num_experts, k, idx_ref.data(), w_ref.data(), ctx);
            FastMoE::route_topk(logits.data(), num_experts, k, idx_opt.data(), w_opt.data(), ctx);

            for (size_t i = 0; i < k; ++i) {
                assert(idx_opt[i] == idx_ref[i]);
                assert(std::abs(w_opt[i] - w_ref[i]) < 1e-5f);
            }

            // Normalization check: sum of weights = 1.0f
            float sum_w = 0.0f;
            for (size_t i = 0; i < k; ++i) sum_w += w_opt[i];
            assert(std::abs(sum_w - 1.0f) < 1e-4f);
        }
    }

    // Edge Cases for route_topk:
    // Identical logits -> stable tie-breaking by index
    {
        std::vector<float> same_logits(8, 2.0f);
        std::vector<int32_t> idx(4);
        std::vector<float> w(4);
        FastMoE::route_topk(same_logits.data(), 8, 4, idx.data(), w.data(), ctx);
        for (size_t i = 0; i < 4; ++i) {
            assert(idx[i] == static_cast<int32_t>(i));
            assert(std::abs(w[i] - 0.25f) < 1e-5f);
        }
    }

    // Extreme logits (+500, -500, NaN)
    {
        std::vector<float> ext_logits = {500.0f, -500.0f, std::numeric_limits<float>::quiet_NaN(), 1.0f};
        std::vector<int32_t> idx(2);
        std::vector<float> w(2);
        FastMoE::route_topk(ext_logits.data(), 4, 2, idx.data(), w.data(), ctx);
        assert(idx[0] == 0); // Sanitized to 88.0f
        assert(!std::isnan(w[0]) && !std::isnan(w[1]));
        assert(std::abs((w[0] + w[1]) - 1.0f) < 1e-4f);
    }

    // Weighted Aggregation Parity
    {
        const size_t top_k = 4;
        const size_t dim = 128;
        std::vector<std::vector<float>> exp_outs(top_k, std::vector<float>(dim));
        std::vector<const float*> exp_ptrs(top_k);
        std::vector<float> weights(top_k, 0.25f);
        std::vector<float> out_ref(dim, 0.0f);
        std::vector<float> out_opt(dim, 0.0f);

        for (size_t k = 0; k < top_k; ++k) {
            for (size_t d = 0; d < dim; ++d) exp_outs[k][d] = dist(rng);
            exp_ptrs[k] = exp_outs[k].data();
            for (size_t d = 0; d < dim; ++d) {
                out_ref[d] += weights[k] * exp_outs[k][d];
            }
        }

        FastMoE::weighted_aggregate(exp_ptrs.data(), weights.data(), top_k, dim, out_opt.data());
        for (size_t d = 0; d < dim; ++d) {
            assert(std::abs(out_opt[d] - out_ref[d]) < 1e-5f);
        }
    }

    std::cout << "  Passed. FastMoE router & aggregation parity verified." << std::endl;
}

// ============================================================================
// 8. FastSoftmax Parity & Edge Case Validation
// ============================================================================
void test_fast_softmax_parity() {
    using namespace strata::kernels;
    std::cout << "[Test 8/10] Verifying FastSoftmax & Log-Sum-Exp Parity & Edge Cases..." << std::endl;

    auto ref_softmax = [](float* x, size_t n, float temp) {
        if (n == 0) return;
        if (temp <= 0.0f) {
            int best = 0;
            float mv = x[0];
            for (size_t i = 1; i < n; ++i) {
                if (x[i] > mv) { mv = x[i]; best = static_cast<int>(i); }
            }
            std::fill(x, x + n, 0.0f);
            x[best] = 1.0f;
            return;
        }
        float inv_t = 1.0f / temp;
        float max_v = -1e30f;
        for (size_t i = 0; i < n; ++i) {
            float v = x[i] * inv_t;
            if (v > max_v) max_v = v;
        }
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            x[i] = std::exp(x[i] * inv_t - max_v);
            sum += x[i];
        }
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < n; ++i) x[i] *= inv_sum;
    };

    auto ref_lse = [](const float* x, size_t n) {
        if (n == 0) return -std::numeric_limits<float>::infinity();
        float max_v = -1e30f;
        for (size_t i = 0; i < n; ++i) {
            if (x[i] > max_v) max_v = x[i];
        }
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            sum += std::exp(x[i] - max_v);
        }
        return max_v + std::log(sum);
    };

    std::mt19937 rng(666);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    std::vector<size_t> test_sizes = {16, 32, 64, 128, 512, 37, 127};

    for (size_t n : test_sizes) {
        std::vector<float> x_ref(n), x_opt(n);
        for (size_t i = 0; i < n; ++i) {
            x_ref[i] = dist(rng);
            x_opt[i] = x_ref[i];
        }

        // Standard softmax parity (temperature = 1.0f)
        ref_softmax(x_ref.data(), n, 1.0f);
        FastSoftmax::softmax_inplace(x_opt.data(), n, 1.0f);

        float sum_p = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            assert(std::abs(x_opt[i] - x_ref[i]) < 1e-5f);
            sum_p += x_opt[i];
        }
        assert(std::abs(sum_p - 1.0f) < 1e-4f);

        // LSE parity
        std::vector<float> x_lse(n);
        for (size_t i = 0; i < n; ++i) x_lse[i] = dist(rng);
        float lse_ref = ref_lse(x_lse.data(), n);
        float lse_opt = FastSoftmax::log_sum_exp(x_lse.data(), n);
        assert(std::abs(lse_opt - lse_ref) < 1e-4f);

        // ArgMax parity
        int32_t argmax_ref = 0;
        float max_v = x_lse[0];
        for (size_t i = 1; i < n; ++i) {
            if (x_lse[i] > max_v) { max_v = x_lse[i]; argmax_ref = static_cast<int32_t>(i); }
        }
        int32_t argmax_opt = FastSoftmax::argmax(x_lse.data(), n);
        assert(argmax_opt == argmax_ref);
    }

    // Edge Cases:
    // Temperature = 0.0f (greedy one-hot delta)
    {
        std::vector<float> x = {1.0f, 5.0f, 2.0f, -1.0f};
        FastSoftmax::softmax_inplace(x.data(), x.size(), 0.0f);
        assert(x[1] == 1.0f);
        assert(x[0] == 0.0f && x[2] == 0.0f && x[3] == 0.0f);
    }

    // Extreme values (+1000.0f, -1000.0f)
    {
        std::vector<float> x = {1000.0f, 999.0f, -1000.0f};
        FastSoftmax::softmax_inplace(x.data(), x.size(), 1.0f);
        assert(!std::isnan(x[0]) && !std::isnan(x[1]) && !std::isnan(x[2]));
        assert(std::abs((x[0] + x[1] + x[2]) - 1.0f) < 1e-4f);
    }

    // All zeros -> uniform 1/N
    {
        std::vector<float> x(10, 0.0f);
        FastSoftmax::softmax_inplace(x.data(), x.size(), 1.0f);
        for (float p : x) assert(std::abs(p - 0.1f) < 1e-5f);
    }

    std::cout << "  Passed. FastSoftmax & Log-Sum-Exp parity verified." << std::endl;
}

// ============================================================================
// 9. FastRoPE Parity & Edge Case Validation
// ============================================================================
void test_fast_rope_parity() {
    using namespace strata::kernels;
    std::cout << "[Test 9/10] Verifying FastRoPE Parity & Edge Cases..." << std::endl;

    auto ref_rope_neox = [](float* vec, int64_t pos, int64_t rot_dim, float freq_base) {
        const int64_t n_pairs = rot_dim / 2;
        float* x1 = vec;
        float* x2 = vec + n_pairs;
        for (int64_t i = 0; i < n_pairs; ++i) {
            float exponent = static_cast<float>(2 * i) / static_cast<float>(rot_dim);
            float inv_freq = 1.0f / std::pow(freq_base, exponent);
            float angle = static_cast<float>(pos) * inv_freq;
            float c = std::cos(angle);
            float s = std::sin(angle);
            float v1 = x1[i];
            float v2 = x2[i];
            x1[i] = v1 * c - v2 * s;
            x2[i] = v1 * s + v2 * c;
        }
    };

    auto ref_rope_interleaved = [](float* vec, int64_t pos, int64_t rot_dim, float freq_base) {
        const int64_t n_pairs = rot_dim / 2;
        for (int64_t i = 0; i < n_pairs; ++i) {
            float exponent = static_cast<float>(2 * i) / static_cast<float>(rot_dim);
            float inv_freq = 1.0f / std::pow(freq_base, exponent);
            float angle = static_cast<float>(pos) * inv_freq;
            float c = std::cos(angle);
            float s = std::sin(angle);
            float v0 = vec[2 * i];
            float v1 = vec[2 * i + 1];
            vec[2 * i + 0] = v0 * c - v1 * s;
            vec[2 * i + 1] = v0 * s + v1 * c;
        }
    };

    std::mt19937 rng(555);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    std::vector<int64_t> rot_dims = {16, 32, 64, 128, 48};
    std::vector<int64_t> positions = {0, 1, 17, 128, 2048};

    for (int64_t rot_dim : rot_dims) {
        const int64_t n_pairs = rot_dim / 2;
        std::vector<float> inv_freqs(n_pairs);
        FastRoPE::compute_inv_freqs(inv_freqs.data(), rot_dim, 10000.0f);

        for (int64_t pos : positions) {
            std::vector<float> cos_tab(n_pairs), sin_tab(n_pairs);
            FastRoPE::compute_cos_sin(pos, inv_freqs.data(), cos_tab.data(), sin_tab.data(), n_pairs);

            std::vector<float> vec_ref(rot_dim), vec_opt(rot_dim);
            for (int64_t i = 0; i < rot_dim; ++i) {
                vec_ref[i] = dist(rng);
                vec_opt[i] = vec_ref[i];
            }

            // NeoX rotation
            ref_rope_neox(vec_ref.data(), pos, rot_dim, 10000.0f);
            FastRoPE::apply_neox(vec_opt.data(), cos_tab.data(), sin_tab.data(), rot_dim);

            float max_err = 0.0f;
            for (int64_t i = 0; i < rot_dim; ++i) {
                max_err = std::max(max_err, std::abs(vec_opt[i] - vec_ref[i]));
            }
            assert(max_err < 1e-5f);

            // Interleaved rotation
            for (int64_t i = 0; i < rot_dim; ++i) {
                vec_ref[i] = dist(rng);
                vec_opt[i] = vec_ref[i];
            }
            ref_rope_interleaved(vec_ref.data(), pos, rot_dim, 10000.0f);
            FastRoPE::apply_interleaved(vec_opt.data(), cos_tab.data(), sin_tab.data(), rot_dim);

            max_err = 0.0f;
            for (int64_t i = 0; i < rot_dim; ++i) {
                max_err = std::max(max_err, std::abs(vec_opt[i] - vec_ref[i]));
            }
            assert(max_err < 1e-5f);
        }
    }

    // Edge Cases:
    // pos = 0 must be exact identity rotation (cos=1, sin=0)
    {
        const int64_t rot_dim = 64;
        const int64_t n_pairs = rot_dim / 2;
        std::vector<float> inv_freqs(n_pairs), cos_tab(n_pairs), sin_tab(n_pairs);
        FastRoPE::compute_inv_freqs(inv_freqs.data(), rot_dim);
        FastRoPE::compute_cos_sin(0, inv_freqs.data(), cos_tab.data(), sin_tab.data(), n_pairs);

        std::vector<float> orig(rot_dim), rotated(rot_dim);
        for (int64_t i = 0; i < rot_dim; ++i) {
            orig[i] = dist(rng);
            rotated[i] = orig[i];
        }
        FastRoPE::apply_neox(rotated.data(), cos_tab.data(), sin_tab.data(), rot_dim);
        for (int64_t i = 0; i < rot_dim; ++i) {
            assert(rotated[i] == orig[i]);
        }
    }

    // Orthogonality / L2 norm preservation under rotation
    {
        const int64_t rot_dim = 64;
        const int64_t n_pairs = rot_dim / 2;
        std::vector<float> inv_freqs(n_pairs), cos_tab(n_pairs), sin_tab(n_pairs);
        FastRoPE::compute_inv_freqs(inv_freqs.data(), rot_dim);
        FastRoPE::compute_cos_sin(42, inv_freqs.data(), cos_tab.data(), sin_tab.data(), n_pairs);

        std::vector<float> vec(rot_dim);
        float norm_before = 0.0f;
        for (int64_t i = 0; i < rot_dim; ++i) {
            vec[i] = dist(rng);
            norm_before += vec[i] * vec[i];
        }
        FastRoPE::apply_neox(vec.data(), cos_tab.data(), sin_tab.data(), rot_dim);
        float norm_after = 0.0f;
        for (int64_t i = 0; i < rot_dim; ++i) norm_after += vec[i] * vec[i];

        assert(std::abs(norm_before - norm_after) < 1e-4f);
    }

    std::cout << "  Passed. FastRoPE NeoX & Interleaved parity and norm preservation verified." << std::endl;
}

// ============================================================================
// 10. FastQuant Parity & Edge Case Validation
// ============================================================================
void test_fast_quant_parity() {
    using namespace strata::kernels;
    std::cout << "[Test 10/10] Verifying FastQuant Parity & Edge Cases..." << std::endl;

    std::mt19937 rng(333);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    const size_t n = 1024;

    std::vector<float> in(n);
    for (size_t i = 0; i < n; ++i) in[i] = dist(rng);

    // 10a. FP32 <-> FP16 conversion parity
    std::vector<uint16_t> fp16_opt(n);
    std::vector<float> fp32_opt(n);
    FastQuant::f32_to_f16(in.data(), fp16_opt.data(), n);
    FastQuant::f16_to_f32(fp16_opt.data(), fp32_opt.data(), n);

    for (size_t i = 0; i < n; ++i) {
        uint16_t ref_h = f16_from_f32(in[i]);
        assert(fp16_opt[i] == ref_h);
        float ref_f = f32_from_f16(ref_h);
        assert(fp32_opt[i] == ref_f);
    }

    // 10b. FP32 <-> BF16 conversion parity
    std::vector<uint16_t> bf16_opt(n);
    std::vector<float> bf32_opt(n);
    FastQuant::f32_to_bf16(in.data(), bf16_opt.data(), n);
    FastQuant::bf16_to_f32(bf16_opt.data(), bf32_opt.data(), n);

    for (size_t i = 0; i < n; ++i) {
        uint16_t ref_bf = bf16_from_f32(in[i]);
        assert(bf16_opt[i] == ref_bf);
        float ref_f = f32_from_bf16(ref_bf);
        assert(bf32_opt[i] == ref_f);
    }

    // 10c. Q8_0 Symmetric Block Quantizer Parity
    const size_t n_blocks = 32;
    const size_t total_elems = n_blocks * 32;
    std::vector<float> q8_in(total_elems);
    for (size_t i = 0; i < total_elems; ++i) q8_in[i] = dist(rng);

    std::vector<int8_t> q8_quants(total_elems);
    std::vector<float> q8_scales(n_blocks);
    FastQuant::quantize_q8_0(q8_in.data(), q8_quants.data(), q8_scales.data(), n_blocks);

    std::vector<float> q8_dequant(total_elems);
    FastQuant::dequantize_q8_0(q8_quants.data(), q8_scales.data(), q8_dequant.data(), n_blocks);

    // Verify against scalar math reference for each block
    for (size_t b = 0; b < n_blocks; ++b) {
        const float* src = q8_in.data() + b * 32;
        float amax = 0.0f;
        for (size_t j = 0; j < 32; ++j) amax = std::max(amax, std::abs(src[j]));
        float expected_scale = amax / 127.0f;
        assert(std::abs(q8_scales[b] - expected_scale) < 1e-6f);

        for (size_t j = 0; j < 32; ++j) {
            float reconstructed = q8_dequant[b * 32 + j];
            assert(std::abs(reconstructed - src[j]) <= expected_scale + 1e-4f);
        }
    }

    // 10d. Q4_0 Block Dequantizer Parity
    {
        std::vector<uint8_t> nibbles(16);
        for (size_t i = 0; i < 16; ++i) nibbles[i] = static_cast<uint8_t>(i * 17);
        float scale = 0.5f;
        std::vector<float> q4_out(32, 0.0f);
        FastQuant::dequantize_q4_0(nibbles.data(), scale, q4_out.data(), 32);

        for (size_t i = 0; i < 16; ++i) {
            uint8_t b = nibbles[i];
            int8_t lo = static_cast<int8_t>((b & 0x0F) - 8);
            int8_t hi = static_cast<int8_t>(((b >> 4) & 0x0F) - 8);
            assert(q4_out[i * 2 + 0] == static_cast<float>(lo) * scale);
            assert(q4_out[i * 2 + 1] == static_cast<float>(hi) * scale);
        }
    }

    // Edge Cases:
    // Zeros in FP16, BF16, Q8_0
    {
        std::vector<float> zeros(32, 0.0f);
        std::vector<uint16_t> h_zeros(32);
        FastQuant::f32_to_f16(zeros.data(), h_zeros.data(), 32);
        for (uint16_t h : h_zeros) assert(h == 0);

        std::vector<int8_t> q_zeros(32);
        float scale_zero = 0.0f;
        FastQuant::quantize_q8_0(zeros.data(), q_zeros.data(), &scale_zero, 1);
        assert(scale_zero == 0.0f);
        for (int8_t q : q_zeros) assert(q == 0);
    }

    std::cout << "  Passed. FastQuant FP16/BF16/Q8_0/Q4_0 parity verified." << std::endl;
}

} // anonymous namespace

// ============================================================================
// Main Validation Runner
// ============================================================================
int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "Running Fast Math Parity & Edge Case Validation..." << std::endl;
    std::cout << "=================================================================" << std::endl;

    test_edge_and_nan_sanitization();
    test_rmsnorm_parity();
    test_fused_rmsnorm_residual_parity();
    test_swiglu_parity();
    test_gemv_parity();
    test_fast_attention_parity();
    test_fast_moe_parity();
    test_fast_softmax_parity();
    test_fast_rope_parity();
    test_fast_quant_parity();

    std::cout << "=================================================================" << std::endl;
    std::cout << "All Math Parity Tests Passed!" << std::endl;
    std::cout << "=================================================================" << std::endl;
    return 0;
}
