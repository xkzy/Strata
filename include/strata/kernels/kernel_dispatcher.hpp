// include/strata/kernels/kernel_dispatcher.hpp
#pragma once

#include "strata/kernels/math_policy.hpp"
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

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
    OpType op = OpType::kMatMul;
    int64_t M = 0;
    int64_t N = 0;
    int64_t K = 0;
    int32_t dtype = 0;
    MathPolicy policy = MathPolicy::kBalanced;

    bool operator==(const KernelKey& o) const noexcept {
        return op == o.op && M == o.M && N == o.N && K == o.K &&
               dtype == o.dtype && policy == o.policy;
    }

    bool operator!=(const KernelKey& o) const noexcept {
        return !(*this == o);
    }
};

struct KernelKeyHash {
    size_t operator()(const KernelKey& k) const noexcept {
        size_t h = static_cast<size_t>(k.op);
        auto combine = [&h](size_t val) {
            h ^= val + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        combine(static_cast<size_t>(k.M));
        combine(static_cast<size_t>(k.N));
        combine(static_cast<size_t>(k.K));
        combine(static_cast<size_t>(k.dtype));
        combine(static_cast<size_t>(k.policy));
        return h;
    }
};

class KernelDispatcher {
public:
    static KernelDispatcher& instance();

    KernelDispatcher(const KernelDispatcher&) = delete;
    KernelDispatcher& operator=(const KernelDispatcher&) = delete;
    KernelDispatcher(KernelDispatcher&&) = delete;
    KernelDispatcher& operator=(KernelDispatcher&&) = delete;

    void set_default_policy(MathPolicy policy);
    MathPolicy default_policy() const;

    std::string select_kernel(const KernelKey& key);
    void register_kernel(const KernelKey& key, const std::string& kernel_name);

    void clear_cache();
    size_t cache_size() const;
    bool has_kernel(const KernelKey& key) const;

private:
    KernelDispatcher() = default;

    MathPolicy policy_ = MathPolicy::kBalanced;
    std::unordered_map<KernelKey, std::string, KernelKeyHash> cache_;
    mutable std::mutex mu_;
};

} // namespace strata::kernels
