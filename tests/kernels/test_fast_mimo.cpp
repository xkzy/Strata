// tests/kernels/test_fast_mimo.cpp - Unit Tests for Fast MiMo-V2.6 Architecture Engine
#include "strata/kernels/fast_mimo.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

using namespace strata::kernels;

void test_asymmetric_dimensions() {
    std::cout << "[Test 1] Fast MiMo Asymmetric Head Dimensions (QK=192, V=128)..." << std::endl;
    MiMoFlashConfig cfg;
    assert(cfg.head_dim_qk == 192);
    assert(cfg.head_dim_v == 128);
    assert(cfg.swa_window == 128);
    assert(cfg.n_layers == 48);
    assert(cfg.n_swa_layers == 39);
    assert(cfg.n_ga_layers == 9);

    // Verify GA layers
    int64_t ga_count = 0;
    for (int64_t l = 0; l < cfg.n_layers; ++l) {
        if (cfg.is_ga_layer(l)) {
            ga_count++;
        }
    }
    assert(ga_count == 9);

    // Verify dot_qk_192 vectorization against scalar reference
    std::vector<float> q(192), k(192);
    for (size_t i = 0; i < 192; ++i) {
        q[i] = static_cast<float>(i % 7) * 0.1f;
        k[i] = static_cast<float>((i + 3) % 5) * 0.2f;
    }

    float expected_dot = 0.0f;
    for (size_t i = 0; i < 192; ++i) {
        expected_dot += q[i] * k[i];
    }

    float fast_dot = FastMiMoAttention::dot_qk_192(q.data(), k.data());
    float diff = std::abs(expected_dot - fast_dot);
    assert(diff < 1e-4f);
    std::cout << "  [PASSED] dot_qk_192 fast dot matches reference (diff=" << diff << ")" << std::endl;
}

void test_asymmetric_circular_kv_cache() {
    std::cout << "[Test 2] Fast MiMo Circular 128-token SWA KV Cache..." << std::endl;
    MiMoFlashConfig cfg;
    FastMiMoAsymmetricKVCache cache(cfg, /*is_swa=*/true);

    int64_t k_stride = cfg.swa_n_head_kv * cfg.head_dim_qk; // 8 * 192 = 1536
    int64_t v_stride = cfg.swa_n_head_kv * cfg.head_dim_v;  // 8 * 128 = 1024

    std::vector<float> k_tok(k_stride, 1.0f);
    std::vector<float> v_tok(v_stride, 2.0f);

    // Append 200 tokens (greater than sliding window 128)
    for (int64_t t = 0; t < 200; ++t) {
        k_tok[0] = static_cast<float>(t);
        v_tok[0] = static_cast<float>(t * 2);
        cache.append(t, k_tok.data(), v_tok.data());
    }

    assert(cache.total_tokens() == 200);

    // Token 50 is outside the rolling window (200 - 128 = 72)
    assert(cache.get_key(50) == nullptr);
    assert(cache.get_value(50) == nullptr);

    // Token 150 is inside the rolling window
    const float* k_150 = cache.get_key(150);
    const float* v_150 = cache.get_value(150);
    assert(k_150 != nullptr);
    assert(v_150 != nullptr);
    assert(std::abs(k_150[0] - 150.0f) < 1e-5f);
    assert(std::abs(v_150[0] - 300.0f) < 1e-5f);

    std::cout << "  [PASSED] SWA ring buffer correctly drops expired keys and preserves active window." << std::endl;
}

void test_mimo_attention_decode_step() {
    std::cout << "[Test 3] Fast MiMo Asymmetric Attention Decode Step..." << std::endl;
    MiMoFlashConfig cfg;
    FastMiMoAsymmetricKVCache cache(cfg, /*is_swa=*/true);

    int64_t k_stride = cfg.swa_n_head_kv * cfg.head_dim_qk;
    int64_t v_stride = cfg.swa_n_head_kv * cfg.head_dim_v;

    // Fill 10 tokens in cache
    for (int64_t t = 0; t < 10; ++t) {
        std::vector<float> k_tok(k_stride, 0.05f * (t + 1));
        std::vector<float> v_tok(v_stride, 0.1f * (t + 1));
        cache.append(t, k_tok.data(), v_tok.data());
    }

    // Step at query pos 9
    std::vector<float> q_step(cfg.n_head * cfg.head_dim_qk, 0.1f);
    std::vector<float> out_step(cfg.n_head * cfg.head_dim_v, 0.0f);

    FastMiMoAttention::decode_step(q_step.data(), cache, 9, 1 /*SWA layer*/, cfg, out_step.data());

    // Validate that output is finite and non-zero
    float sum_out = 0.0f;
    for (float val : out_step) {
        assert(!std::isnan(val) && !std::isinf(val));
        sum_out += std::abs(val);
    }
    assert(sum_out > 0.0f);
    std::cout << "  [PASSED] Decode step produced valid attention outputs (total abs sum=" << sum_out << ")" << std::endl;
}

void test_mimo_moe_routing() {
    std::cout << "[Test 4] Fast MiMo 256-Expert Top-8 Routing..." << std::endl;
    std::vector<float> logits(256, 0.0f);
    // Give experts 10, 20, 30, 40, 50, 60, 70, 80 highest scores
    std::vector<int32_t> top_indices = {10, 20, 30, 40, 50, 60, 70, 80};
    for (size_t i = 0; i < top_indices.size(); ++i) {
        logits[top_indices[i]] = 10.0f + static_cast<float>(i);
    }

    int32_t selected[8];
    float weights[8];
    FastMiMoMoERouter::route_top8(logits.data(), selected, weights, /*use_softmax=*/true);

    float sum_w = 0.0f;
    for (int i = 0; i < 8; ++i) {
        sum_w += weights[i];
        // The top expert must be 80 (highest score)
        if (i == 0) {
            assert(selected[i] == 80);
        }
    }
    assert(std::abs(sum_w - 1.0f) < 1e-5f);
    std::cout << "  [PASSED] FastMiMoMoERouter correctly selected top-8 experts and normalized weights to 1.0." << std::endl;
}

void test_mimo_mtp_speculative_verifier() {
    std::cout << "[Test 5] Fast MiMo 7-Token Speculative Drafter..." << std::endl;
    int32_t draft_tokens[7] = {101, 102, 103, 104, 105, 106, 107};
    int64_t vocab_size = 1000;
    std::vector<float> target_logits(7 * vocab_size, 0.0f);

    // Let the target model agree on the first 4 tokens, but disagree on token 5
    for (int64_t s = 0; s < 4; ++s) {
        target_logits[s * vocab_size + draft_tokens[s]] = 10.0f;
    }
    // Token 5 disagrees: target predicts token 999 instead
    target_logits[4 * vocab_size + 999] = 10.0f;
    target_logits[4 * vocab_size + draft_tokens[4]] = 0.0f; // low probability

    int32_t accepted[8];
    int64_t accepted_count = FastMiMoMTPVerifier::verify_draft_sequence(
        draft_tokens, target_logits.data(), vocab_size, accepted);

    // Should accept 4 draft tokens + 1 correct target token = 5 tokens
    assert(accepted_count == 5);
    for (int64_t i = 0; i < 4; ++i) {
        assert(accepted[i] == draft_tokens[i]);
    }
    assert(accepted[4] == 999);
    std::cout << "  [PASSED] MTP Speculative Verifier correctly accepted prefix and recovered correction." << std::endl;
}

int main() {
    std::cout << "=== Running Fast MiMo-V2.6 Architecture Engine Tests ===" << std::endl;
    test_asymmetric_dimensions();
    test_asymmetric_circular_kv_cache();
    test_mimo_attention_decode_step();
    test_mimo_moe_routing();
    test_mimo_mtp_speculative_verifier();
    std::cout << "All MiMo-V2.6 architecture engine tests passed successfully!" << std::endl;
    return 0;
}
