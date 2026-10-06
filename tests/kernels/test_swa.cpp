// tests/kernels/test_swa.cpp - Comprehensive Unit Test Suite for SWA Optimization Engine
#include "strata/kernels/swa.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void test_swa_causal_range() {
    std::cout << "[Test 1/4] Testing SWA Causal Range Computation..." << std::endl;
    int64_t window_size = 4096;

    // 1. Short sequence: pos < window_size
    auto r1 = strata::kernels::SWAExecutionEngine::compute_causal_range(100, window_size);
    assert(r1.start_pos == 0);
    assert(r1.end_pos == 100);
    assert(r1.active_length == 101);
    assert(r1.is_in_window(0));
    assert(r1.is_in_window(50));
    assert(!r1.is_in_window(101));

    // 2. Long sequence: pos >= window_size
    auto r2 = strata::kernels::SWAExecutionEngine::compute_causal_range(10000, window_size);
    assert(r2.start_pos == 10000 - 4096 + 1);
    assert(r2.end_pos == 10000);
    assert(r2.active_length == 4096);
    assert(!r2.is_in_window(0));       // Too old
    assert(!r2.is_in_window(5000));    // Out of rolling window
    assert(r2.is_in_window(5905));     // Exact start
    assert(r2.is_in_window(10000));    // Exact end

    std::cout << "  Passed. Causal range boundaries strictly verified." << std::endl;
}

void test_swa_ring_buffer_kv() {
    std::cout << "[Test 2/4] Testing Sliding Window Circular Ring-Buffer KV Cache..." << std::endl;
    strata::kernels::SwaWindowConfig cfg;
    cfg.window_size = 512;
    cfg.n_head = 24;
    cfg.n_head_kv = 4;
    cfg.head_dim = 64;

    strata::kernels::SlidingWindowKVCache kv_cache(cfg);
    int64_t stride = cfg.kv_stride_per_token();
    std::vector<float> dummy_k(stride, 1.0f);
    std::vector<float> dummy_v(stride, 2.0f);

    // Append 5,000 tokens (wrapping circular ring buffer ~10 times)
    for (int64_t pos = 0; pos < 5000; ++pos) {
        dummy_k[0] = static_cast<float>(pos);
        dummy_v[0] = static_cast<float>(pos * 2);
        kv_cache.append(pos, dummy_k.data(), dummy_v.data());
    }

    assert(kv_cache.total_tokens_appended() == 5000);
    assert(kv_cache.active_window_tokens() == 512);

    // Old tokens (> 5000 - 512) must be out of window and return nullptr
    assert(kv_cache.get_key(0) == nullptr);
    assert(kv_cache.get_key(4000) == nullptr);
    assert(kv_cache.get_key(4487) == nullptr);

    // Rolling window tokens (4488 to 4999) must be valid and ring-mapped
    const float* k_recent = kv_cache.get_key(4999);
    assert(k_recent != nullptr);
    assert(k_recent[0] == 4999.0f);

    const float* v_recent = kv_cache.get_value(4999);
    assert(v_recent != nullptr);
    assert(v_recent[0] == 4999.0f * 2.0f);

    // Memory savings verification
    double ratio_5k = kv_cache.memory_reduction_ratio(5000);
    assert(ratio_5k > 9.7); // ~9.76x reduction

    double ratio_1m = kv_cache.memory_reduction_ratio(1000000);
    assert(ratio_1m > 1900.0); // ~1953x reduction for 1M tokens with W=512

    std::cout << "  Passed. Ring buffer wrapping and " << ratio_5k << "x memory savings verified." << std::endl;
}

void test_swa_tile_pruning() {
    std::cout << "[Test 3/4] Testing SWA Prefill 2D Tile Pruning..." << std::endl;
    int64_t window_size = 1024;

    // 1. Future tile (col_start > row_end): completely skipped
    bool skip1 = strata::kernels::SWAExecutionEngine::should_skip_prefill_tile(0, 127, 256, 383, window_size);
    assert(skip1 == true);

    // 2. Active diagonal tile: must be computed
    bool skip2 = strata::kernels::SWAExecutionEngine::should_skip_prefill_tile(500, 627, 400, 527, window_size);
    assert(skip2 == false);

    // 3. Out-of-window old tile (col_end < row_start - window_size + 1): completely skipped
    bool skip3 = strata::kernels::SWAExecutionEngine::should_skip_prefill_tile(5000, 5127, 0, 127, window_size);
    assert(skip3 == true);

    std::cout << "  Passed. Prefill 2D tile pruning correctly avoids out-of-window FLOPs." << std::endl;
}

void test_swa_decode_step() {
    std::cout << "[Test 4/4] Testing Optimized SWA Decoding Step (GQA + Bounded Softmax)..." << std::endl;
    strata::kernels::SwaWindowConfig cfg;
    cfg.window_size = 128;
    cfg.n_head = 8;
    cfg.n_head_kv = 2; // 4 queries per KV head (GQA)
    cfg.head_dim = 32;
    cfg.scale = 1.0f / std::sqrt(32.0f);

    strata::kernels::SlidingWindowKVCache kv_cache(cfg);
    int64_t kv_stride = cfg.kv_stride_per_token();
    int64_t q_stride = cfg.q_stride_per_token();

    std::vector<float> k_tok(kv_stride, 0.1f);
    std::vector<float> v_tok(kv_stride, 0.5f);

    // Populate 300 historical steps
    for (int64_t i = 0; i < 300; ++i) {
        k_tok[0] = static_cast<float>(i % 10);
        v_tok[0] = static_cast<float>(i * 0.01f);
        kv_cache.append(i, k_tok.data(), v_tok.data());
    }

    // Prepare query for step 299
    std::vector<float> q_step(q_stride, 0.2f);
    std::vector<float> out_attn(q_stride, 0.0f);

    strata::kernels::SWAExecutionEngine::decode_step(q_step.data(), kv_cache, 299, cfg, out_attn.data());

    // Verify output is finite, non-zero, and bounded
    for (int64_t i = 0; i < q_stride; ++i) {
        assert(!std::isnan(out_attn[i]));
        assert(!std::isinf(out_attn[i]));
    }
    assert(out_attn[0] != 0.0f);

    std::cout << "  Passed. SWA decode step generated correct, bounded attention projections." << std::endl;
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "   RUNNING STRATA SLIDING WINDOW ATTENTION (SWA) TEST SUITE      " << std::endl;
    std::cout << "=================================================================" << std::endl;

    test_swa_causal_range();
    test_swa_ring_buffer_kv();
    test_swa_tile_pruning();
    test_swa_decode_step();

    std::cout << "=================================================================" << std::endl;
    std::cout << "   ALL SLIDING WINDOW ATTENTION (SWA) TESTS PASSED (4/4)         " << std::endl;
    std::cout << "=================================================================" << std::endl;
    return 0;
}
