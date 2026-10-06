// tests/kernels/test_math_policy.cpp
#include "strata/kernels/math_policy.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

int main() {
    using namespace strata::kernels;

    // Default construction (Balanced)
    MathContext ctx;
    assert(ctx.policy == MathPolicy::kBalanced);
    assert(ctx.rsqrt_epsilon == 1e-6f);
    assert(ctx.softmax_max_clamp == 88.0f);
    assert(ctx.allow_fused_residual == true);
    assert(ctx.allow_approximate_gelu == true);
    assert(ctx.allow_fast_exp == false);

    // Switching to Strict
    ctx.policy = MathPolicy::kStrict;
    assert(ctx.policy == MathPolicy::kStrict);
    assert(!ctx.allow_approximate_gelu);
    assert(!ctx.allow_fast_exp);

    // Switching to Fast
    ctx.policy = MathPolicy::kFast;
    assert(ctx.policy == MathPolicy::kFast);
    assert(ctx.allow_approximate_gelu);
    assert(ctx.allow_fast_exp);

    // Switching to Balanced
    ctx.policy = MathPolicy::kBalanced;
    assert(ctx.policy == MathPolicy::kBalanced);
    assert(ctx.allow_approximate_gelu);
    assert(!ctx.allow_fast_exp);

    // Explicit constructor variants
    MathContext ctx_strict(MathPolicy::kStrict);
    assert(ctx_strict.policy == MathPolicy::kStrict);
    assert(!ctx_strict.allow_approximate_gelu);
    assert(!ctx_strict.allow_fast_exp);

    MathContext ctx_fast(MathPolicy::kFast);
    assert(ctx_fast.policy == MathPolicy::kFast);
    assert(ctx_fast.allow_approximate_gelu);
    assert(ctx_fast.allow_fast_exp);

    // Copy isolation
    MathContext ctx_copy = ctx_fast;
    assert(ctx_copy.policy == MathPolicy::kFast);
    ctx_copy.policy = MathPolicy::kStrict;
    assert(ctx_copy.policy == MathPolicy::kStrict);
    assert(ctx_fast.policy == MathPolicy::kFast); // original untouched

    // Bounds sanitization and validation tests
    assert(sanitize_logit(100.0f, ctx) == 88.0f);
    assert(sanitize_logit(-100.0f, ctx) == -88.0f);
    assert(sanitize_logit(std::numeric_limits<float>::quiet_NaN(), ctx) == -88.0f);
    assert(sanitize_logit(std::numeric_limits<float>::infinity(), ctx) == 88.0f);
    assert(sanitize_logit(-std::numeric_limits<float>::infinity(), ctx) == -88.0f);
    assert(sanitize_logit(0.0f, ctx) == 0.0f);

    assert(validate_math_bounds(50.0f, ctx) == 50.0f);
    assert(validate_math_bounds(120.0f, ctx) == 88.0f);
    assert(validate_math_bounds(-120.0f, ctx) == -88.0f);

    std::cout << "test_math_policy passed" << std::endl;
    return 0;
}
