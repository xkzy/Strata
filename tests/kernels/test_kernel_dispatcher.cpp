// tests/kernels/test_kernel_dispatcher.cpp
#include "strata/kernels/kernel_dispatcher.hpp"
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>

int main() {
    using namespace strata::kernels;

    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.clear_cache();

    // 1. Policy configuration
    dispatcher.set_default_policy(MathPolicy::kBalanced);
    assert(dispatcher.default_policy() == MathPolicy::kBalanced);

    dispatcher.set_default_policy(MathPolicy::kStrict);
    assert(dispatcher.default_policy() == MathPolicy::kStrict);

    dispatcher.set_default_policy(MathPolicy::kBalanced);
    assert(dispatcher.default_policy() == MathPolicy::kBalanced);

    // 2. Cost model heuristic selections
    // GEMV small M (<= 8) -> simd_row_split_gemv
    KernelKey key_gemv_small{OpType::kGEMV, 1, 4096, 4096, 0, MathPolicy::kBalanced};
    auto dec_gemv_small = dispatcher.select_kernel(key_gemv_small);
    assert(dec_gemv_small == "simd_row_split_gemv");

    // GEMV large M (> 8) -> tiled_gemm
    KernelKey key_gemv_large{OpType::kGEMV, 16, 4096, 4096, 0, MathPolicy::kBalanced};
    auto dec_gemv_large = dispatcher.select_kernel(key_gemv_large);
    assert(dec_gemv_large == "tiled_gemm");

    // FusedNormResidual -> fused_single_pass_rmsnorm
    KernelKey key_norm{OpType::kFusedNormResidual, 1, 4096, 4096, 0, MathPolicy::kBalanced};
    auto dec_norm = dispatcher.select_kernel(key_norm);
    assert(dec_norm == "fused_single_pass_rmsnorm");

    // MoERouting -> partial_select_topk
    KernelKey key_moe{OpType::kMoERouting, 1, 8, 4096, 0, MathPolicy::kBalanced};
    auto dec_moe = dispatcher.select_kernel(key_moe);
    assert(dec_moe == "partial_select_topk");

    // Attention M==1 -> incremental_decode_attn
    KernelKey key_attn_decode{OpType::kAttention, 1, 64, 4096, 0, MathPolicy::kBalanced};
    auto dec_attn_decode = dispatcher.select_kernel(key_attn_decode);
    assert(dec_attn_decode == "incremental_decode_attn");

    // Attention M>1 -> tiled_flash_attn
    KernelKey key_attn_prefill{OpType::kAttention, 128, 64, 4096, 0, MathPolicy::kBalanced};
    auto dec_attn_prefill = dispatcher.select_kernel(key_attn_prefill);
    assert(dec_attn_prefill == "tiled_flash_attn");

    // Fallback default
    KernelKey key_matmul{OpType::kMatMul, 32, 32, 32, 0, MathPolicy::kBalanced};
    auto dec_matmul = dispatcher.select_kernel(key_matmul);
    assert(dec_matmul == "default_vectorized");

    // 3. Cache lookup and manual registration override
    assert(dispatcher.has_kernel(key_matmul));
    dispatcher.register_kernel(key_matmul, "custom_opt_gemm");
    assert(dispatcher.select_kernel(key_matmul) == "custom_opt_gemm");

    // 4. Key comparison and hashing isolation
    KernelKey key_gemv_strict{OpType::kGEMV, 1, 4096, 4096, 0, MathPolicy::kStrict};
    assert(key_gemv_small != key_gemv_strict);
    assert(!(key_gemv_small == key_gemv_strict));

    // 5. Multithreaded concurrent dispatch and registration
    const int num_threads = 8;
    const int ops_per_thread = 500;
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&dispatcher, t]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                KernelKey k{OpType::kGEMV, (i % 16) + 1, 2048, 2048, t, MathPolicy::kBalanced};
                auto res = dispatcher.select_kernel(k);
                assert(!res.empty());

                if (i % 50 == 0) {
                    dispatcher.register_kernel(k, "threaded_kernel_" + std::to_string(t));
                    auto custom = dispatcher.select_kernel(k);
                    assert(!custom.empty());
                }
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }

    assert(dispatcher.cache_size() > 0);

    std::cout << "test_kernel_dispatcher passed" << std::endl;
    return 0;
}
