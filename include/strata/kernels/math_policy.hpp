// include/strata/kernels/math_policy.hpp
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace strata::kernels {

enum class MathPolicy : uint8_t {
    kStrict = 0,
    kBalanced = 1,
    kFast = 2
};

struct MathContext {
    struct PolicyProxy {
        MathContext* parent = nullptr;
        MathPolicy value = MathPolicy::kBalanced;

        PolicyProxy() = default;
        PolicyProxy(MathContext* p, MathPolicy v) : parent(p), value(v) {}

        PolicyProxy& operator=(MathPolicy p) {
            value = p;
            if (parent) {
                parent->apply_policy(p);
            }
            return *this;
        }

        operator MathPolicy() const { return value; }
        bool operator==(MathPolicy p) const { return value == p; }
        bool operator!=(MathPolicy p) const { return value != p; }
        friend bool operator==(MathPolicy p, const PolicyProxy& pp) { return p == pp.value; }
        friend bool operator!=(MathPolicy p, const PolicyProxy& pp) { return p != pp.value; }
        friend bool operator==(const PolicyProxy& a, const PolicyProxy& b) { return a.value == b.value; }
        friend bool operator!=(const PolicyProxy& a, const PolicyProxy& b) { return a.value != b.value; }
    };

    PolicyProxy policy{this, MathPolicy::kBalanced};
    float rsqrt_epsilon = 1e-6f;
    float softmax_max_clamp = 88.0f;
    bool allow_fused_residual = true;
    bool allow_approximate_gelu = true;
    bool allow_fast_exp = false;

    void apply_policy(MathPolicy p) {
        if (p == MathPolicy::kStrict) {
            allow_approximate_gelu = false;
            allow_fast_exp = false;
        } else if (p == MathPolicy::kFast) {
            allow_approximate_gelu = true;
            allow_fast_exp = true;
        } else if (p == MathPolicy::kBalanced) {
            allow_approximate_gelu = true;
            allow_fast_exp = false;
        }
    }

    void set_policy(MathPolicy p) {
        policy = p;
    }

    MathContext() : policy(this, MathPolicy::kBalanced) {}

    explicit MathContext(MathPolicy p) : policy(this, p) {
        apply_policy(p);
    }

    MathContext(const MathContext& other)
        : policy(this, other.policy.value),
          rsqrt_epsilon(other.rsqrt_epsilon),
          softmax_max_clamp(other.softmax_max_clamp),
          allow_fused_residual(other.allow_fused_residual),
          allow_approximate_gelu(other.allow_approximate_gelu),
          allow_fast_exp(other.allow_fast_exp) {}

    MathContext& operator=(const MathContext& other) {
        if (this != &other) {
            policy = other.policy.value;
            rsqrt_epsilon = other.rsqrt_epsilon;
            softmax_max_clamp = other.softmax_max_clamp;
            allow_fused_residual = other.allow_fused_residual;
            allow_approximate_gelu = other.allow_approximate_gelu;
            allow_fast_exp = other.allow_fast_exp;
        }
        return *this;
    }

    MathContext(MathContext&& other) noexcept
        : policy(this, other.policy.value),
          rsqrt_epsilon(other.rsqrt_epsilon),
          softmax_max_clamp(other.softmax_max_clamp),
          allow_fused_residual(other.allow_fused_residual),
          allow_approximate_gelu(other.allow_approximate_gelu),
          allow_fast_exp(other.allow_fast_exp) {}

    MathContext& operator=(MathContext&& other) noexcept {
        if (this != &other) {
            policy = other.policy.value;
            rsqrt_epsilon = other.rsqrt_epsilon;
            softmax_max_clamp = other.softmax_max_clamp;
            allow_fused_residual = other.allow_fused_residual;
            allow_approximate_gelu = other.allow_approximate_gelu;
            allow_fast_exp = other.allow_fast_exp;
        }
        return *this;
    }
};

inline float sanitize_logit(float x, const MathContext& ctx) {
    if (std::isnan(x)) return -ctx.softmax_max_clamp;
    if (x > ctx.softmax_max_clamp) return ctx.softmax_max_clamp;
    if (x < -ctx.softmax_max_clamp) return -ctx.softmax_max_clamp;
    return x;
}

inline float validate_math_bounds(float x, const MathContext& ctx) {
    return sanitize_logit(x, ctx);
}

} // namespace strata::kernels
