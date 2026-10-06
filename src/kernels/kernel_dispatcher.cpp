// src/kernels/kernel_dispatcher.cpp
#include "strata/kernels/kernel_dispatcher.hpp"

namespace strata::kernels {

KernelDispatcher& KernelDispatcher::instance() {
    static KernelDispatcher inst;
    return inst;
}

void KernelDispatcher::set_default_policy(MathPolicy policy) {
    std::lock_guard<std::mutex> lock(mu_);
    policy_ = policy;
}

MathPolicy KernelDispatcher::default_policy() const {
    std::lock_guard<std::mutex> lock(mu_);
    return policy_;
}

std::string KernelDispatcher::select_kernel(const KernelKey& key) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second;
    }

    // Cost model heuristic
    std::string choice;
    switch (key.op) {
        case OpType::kGEMV:
            choice = (key.M <= 8) ? "simd_row_split_gemv" : "tiled_gemm";
            break;
        case OpType::kFusedNormResidual:
            choice = "fused_single_pass_rmsnorm";
            break;
        case OpType::kMoERouting:
            choice = "partial_select_topk";
            break;
        case OpType::kAttention:
            choice = (key.M == 1) ? "incremental_decode_attn" : "tiled_flash_attn";
            break;
        default:
            choice = "default_vectorized";
            break;
    }
    cache_[key] = choice;
    return choice;
}

void KernelDispatcher::register_kernel(const KernelKey& key, const std::string& kernel_name) {
    std::lock_guard<std::mutex> lock(mu_);
    cache_[key] = kernel_name;
}

void KernelDispatcher::clear_cache() {
    std::lock_guard<std::mutex> lock(mu_);
    cache_.clear();
}

size_t KernelDispatcher::cache_size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return cache_.size();
}

bool KernelDispatcher::has_kernel(const KernelKey& key) const {
    std::lock_guard<std::mutex> lock(mu_);
    return cache_.find(key) != cache_.end();
}

} // namespace strata::kernels
