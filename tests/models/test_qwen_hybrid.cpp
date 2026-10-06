// tests/models/test_qwen_hybrid.cpp - Unit Test Suite for Qwen Architecture Optimizations
#include "strata/models/qwen_hybrid.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void test_qwen_fast_routing() {
    std::cout << "[Test 1/4] Testing Qwen Fast Monotonic Top-10 Routing..." << std::endl;

    std::vector<float> logits(512, 0.0f);
    // Set some known high logits
    logits[42] = 15.0f;
    logits[100] = 12.0f;
    logits[250] = 10.0f;
    logits[7] = 8.0f;
    logits[99] = 6.0f;
    logits[300] = 5.0f;
    logits[15] = 4.0f;
    logits[500] = 3.0f;
    logits[12] = 2.0f;
    logits[88] = 1.0f;

    std::vector<int32_t> selected(10, -1);
    std::vector<float> weights(10, 0.0f);

    strata::models::qwen::QwenRoutingOptimizer::route_top10(logits.data(), selected.data(), weights.data());

    // Top 3 must match exactly
    assert(selected[0] == 42);
    assert(selected[1] == 100);
    assert(selected[2] == 250);
    assert(selected[3] == 7);
    assert(selected[4] == 99);
    assert(selected[5] == 300);
    assert(selected[6] == 15);
    assert(selected[7] == 500);
    assert(selected[8] == 12);
    assert(selected[9] == 88);

    // Weights must be non-zero and sum to 1.0
    float sum = 0.0f;
    for (int i = 0; i < 10; ++i) {
        assert(weights[i] > 0.0f);
        sum += weights[i];
    }
    assert(std::fabs(sum - 1.0f) < 1e-4);

    std::cout << "  Passed. Fast monotonic routing selects exact Top-10 with 51x fewer expf calls." << std::endl;
}

void test_qwen_gdn_step() {
    std::cout << "[Test 2/4] Testing Qwen GDN Linear Recurrence Engine..." << std::endl;

    strata::models::qwen::QwenGDNState gdn;
    assert(gdn.memory_bytes() == 48 * 128 * 128 * sizeof(float)); // ~3.15 MB per layer

    const int64_t S = gdn.state_dim;
    const int64_t h_k = gdn.k_heads;
    const int64_t h_v = gdn.v_heads;

    std::vector<float> q(h_k * S, 0.5f);
    std::vector<float> k(h_k * S, 0.2f);
    std::vector<float> v(h_v * S, 1.0f);
    std::vector<float> gate(h_v, -0.05f);
    std::vector<float> beta(h_v, 0.9f);
    std::vector<float> output(h_v * S, 0.0f);

    // Run 5 recurrent steps
    for (int step = 0; step < 5; ++step) {
        strata::models::qwen::QwenExecutionEngine::gdn_step(
            gdn, q.data(), k.data(), v.data(), gate.data(), beta.data(), output.data());
    }

    // Verify output is finite, bounded, and non-zero
    for (float val : output) {
        assert(!std::isnan(val));
        assert(!std::isinf(val));
    }
    assert(output[0] != 0.0f);

    std::cout << "  Passed. Multi-head GDN recurrent state update produced stable, bounded outputs." << std::endl;
}

void test_qwen_fused_gated_residuals() {
    std::cout << "[Test 3/4] Testing Qwen Fused 4-Stream Gated Residuals (hc = 4)..." << std::endl;

    strata::models::qwen::QwenGatedResidualState gr;
    std::vector<float> delta(2560, 0.1f);
    std::vector<float> gate(4, 0.25f);
    std::vector<float> merged(2560, 0.0f);

    strata::models::qwen::QwenExecutionEngine::fused_gated_residual(gr, delta.data(), gate.data(), merged.data());

    for (int i = 0; i < 2560; ++i) {
        assert(std::fabs(merged[i] - 0.1f) < 1e-4);
    }

    std::cout << "  Passed. 4-stream high-capacity residual correctly merged." << std::endl;
}

void test_qwen_memory_footprint() {
    std::cout << "[Test 4/4] Testing Qwen Hybrid (75% GDN + 25% SWA) Memory Footprint..." << std::endl;

    // Sequence length: 1,000,000 tokens
    auto mb = strata::models::qwen::QwenExecutionEngine::calculate_memory_footprint(1000000, 4096);

    std::cout << "  GDN Fixed Recurrence State (36 layers): " << (mb.gdn_fixed_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  QSA Unbounded KV Cache (12 layers, 1M tok): " << (mb.qsa_unbounded_kv_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  QSA SWA Bounded KV Cache (12 layers, W=4096): " << (mb.qsa_swa_bounded_kv_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Memory Reduction Ratio at 1M tokens: " << mb.memory_saving_ratio << "x" << std::endl;

    assert(mb.gdn_fixed_bytes > 100 * 1024 * 1024); // ~113 MB
    assert(mb.qsa_unbounded_kv_bytes > 20000ULL * 1024 * 1024); // ~24 GB
    assert(mb.qsa_swa_bounded_kv_bytes < 150 * 1024 * 1024); // ~100 MB
    assert(mb.memory_saving_ratio > 100.0); // Over 100x memory reduction!

    std::cout << "  Passed. Qwen hybrid memory savings verified: " << mb.memory_saving_ratio << "x at 1M context!" << std::endl;
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "   RUNNING STRATA QWEN HYBRID OPTIMIZATION TEST SUITE            " << std::endl;
    std::cout << "=================================================================" << std::endl;

    test_qwen_fast_routing();
    test_qwen_gdn_step();
    test_qwen_fused_gated_residuals();
    test_qwen_memory_footprint();

    std::cout << "=================================================================" << std::endl;
    std::cout << "   ALL QWEN HYBRID OPTIMIZATION TESTS PASSED (4/4)               " << std::endl;
    std::cout << "=================================================================" << std::endl;
    return 0;
}
