// tests/kernels/test_fast_attention.cpp
#include "strata/kernels/fast_attention.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

void test_basic_decode() {
    using namespace strata::kernels;

    const size_t head_dim = 64;
    const size_t seq_len = 8;
    std::vector<float> q(head_dim, 0.5f);
    std::vector<float> k_cache(seq_len * head_dim, 0.2f);
    std::vector<float> v_cache(seq_len * head_dim, 1.0f);
    std::vector<float> out(head_dim, 0.0f);

    FastAttention::scaled_dot_product_decode(
        q.data(), k_cache.data(), v_cache.data(), out.data(),
        seq_len, head_dim, 1.0f / std::sqrt(static_cast<float>(head_dim))
    );

    // With identical V values (=1.0f), output must be 1.0f across all dimensions
    for (size_t i = 0; i < head_dim; ++i) {
        assert(std::abs(out[i] - 1.0f) < 1e-4f);
    }
}

void test_reference_parity_and_simd() {
    using namespace strata::kernels;

    // Test multiple head dimensions (both multiples of 16 for AVX-512/AVX2 and unaligned/odd sizes)
    std::vector<size_t> test_dims = {16, 32, 64, 128, 96, 67};
    std::vector<size_t> test_lens = {1, 4, 17, 64, 128};

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t head_dim : test_dims) {
        for (size_t seq_len : test_lens) {
            std::vector<float> q(head_dim);
            std::vector<float> k_cache(seq_len * head_dim);
            std::vector<float> v_cache(seq_len * head_dim);
            std::vector<float> out(head_dim, 0.0f);

            for (size_t i = 0; i < head_dim; ++i) q[i] = dist(rng);
            for (size_t i = 0; i < seq_len * head_dim; ++i) {
                k_cache[i] = dist(rng);
                v_cache[i] = dist(rng);
            }

            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            MathContext ctx(MathPolicy::kBalanced);

            FastAttention::scaled_dot_product_decode(
                q.data(), k_cache.data(), v_cache.data(), out.data(),
                seq_len, head_dim, scale, ctx
            );

            // Compute scalar reference
            std::vector<float> ref_scores(seq_len, 0.0f);
            float max_score = -1e30f;
            for (size_t s = 0; s < seq_len; ++s) {
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    dot += q[d] * k_cache[s * head_dim + d];
                }
                float score = sanitize_logit(dot * scale, ctx);
                ref_scores[s] = score;
                if (score > max_score) max_score = score;
            }

            float sum_exp = 0.0f;
            for (size_t s = 0; s < seq_len; ++s) {
                ref_scores[s] = std::exp(ref_scores[s] - max_score);
                sum_exp += ref_scores[s];
            }
            float inv_sum = 1.0f / (sum_exp + 1e-9f);
            for (size_t s = 0; s < seq_len; ++s) {
                ref_scores[s] *= inv_sum;
            }

            std::vector<float> ref_out(head_dim, 0.0f);
            for (size_t s = 0; s < seq_len; ++s) {
                float w = ref_scores[s];
                for (size_t d = 0; d < head_dim; ++d) {
                    ref_out[d] += w * v_cache[s * head_dim + d];
                }
            }

            for (size_t d = 0; d < head_dim; ++d) {
                float diff = std::abs(out[d] - ref_out[d]);
                assert(diff < 1e-4f);
            }
        }
    }
}

void test_attention_masking() {
    using namespace strata::kernels;

    const size_t head_dim = 32;
    const size_t seq_len = 4;
    std::vector<float> q(head_dim, 1.0f);
    std::vector<float> k_cache(seq_len * head_dim, 1.0f);
    std::vector<float> v_cache(seq_len * head_dim, 0.0f);
    // Give each token's value a distinct marker
    for (size_t s = 0; s < seq_len; ++s) {
        for (size_t d = 0; d < head_dim; ++d) {
            v_cache[s * head_dim + d] = static_cast<float>(s + 1);
        }
    }

    // Mask out all tokens except index 2 (third token)
    std::vector<float> mask = {-1e9f, -1e9f, 0.0f, -1e9f};
    std::vector<float> out(head_dim, 0.0f);

    FastAttention::scaled_dot_product_decode(
        q.data(), k_cache.data(), v_cache.data(), out.data(),
        seq_len, head_dim, 1.0f / std::sqrt(static_cast<float>(head_dim)),
        mask.data()
    );

    // Because only token 2 is unmasked, output must be equal to v_cache of token 2 (= 3.0f)
    for (size_t d = 0; d < head_dim; ++d) {
        assert(std::abs(out[d] - 3.0f) < 1e-4f);
    }
}

void test_bounds_and_nan_sanitization() {
    using namespace strata::kernels;

    const size_t head_dim = 16;
    const size_t seq_len = 3;

    std::vector<float> q(head_dim, 1000.0f); // huge query vector
    std::vector<float> k_cache(seq_len * head_dim, 1000.0f); // huge key vector
    std::vector<float> v_cache(seq_len * head_dim, 1.0f);
    std::vector<float> out(head_dim, 0.0f);

    MathContext ctx(MathPolicy::kBalanced);
    // Ensure no overflow or NaN occurs despite huge inner products
    FastAttention::scaled_dot_product_decode(
        q.data(), k_cache.data(), v_cache.data(), out.data(),
        seq_len, head_dim, 1.0f, ctx
    );

    for (size_t d = 0; d < head_dim; ++d) {
        assert(!std::isnan(out[d]));
        assert(!std::isinf(out[d]));
        assert(std::abs(out[d] - 1.0f) < 1e-3f);
    }
}

void test_multi_head_decode() {
    using namespace strata::kernels;

    const size_t num_heads = 4;
    const size_t num_kv_heads = 2; // GQA 2:1
    const size_t head_dim = 32;
    const size_t seq_len = 5;

    std::vector<float> q(num_heads * head_dim, 0.5f);
    std::vector<float> k_cache(seq_len * num_kv_heads * head_dim, 0.2f);
    std::vector<float> v_cache(seq_len * num_kv_heads * head_dim, 2.0f);
    std::vector<float> out(num_heads * head_dim, 0.0f);

    FastAttention::multi_head_scaled_dot_product_decode(
        q.data(), k_cache.data(), v_cache.data(), out.data(),
        num_heads, num_kv_heads, seq_len, head_dim,
        1.0f / std::sqrt(static_cast<float>(head_dim))
    );

    for (size_t i = 0; i < num_heads * head_dim; ++i) {
        assert(std::abs(out[i] - 2.0f) < 1e-4f);
    }
}

void test_edge_cases() {
    using namespace strata::kernels;

    std::vector<float> q(16, 1.0f);
    std::vector<float> k(16, 1.0f);
    std::vector<float> v(16, 1.0f);
    std::vector<float> out(16, 999.0f);

    // seq_len == 0: no-op
    FastAttention::scaled_dot_product_decode(q.data(), k.data(), v.data(), out.data(), 0, 16, 1.0f);
    assert(out[0] == 999.0f);

    // head_dim == 0: no-op
    FastAttention::scaled_dot_product_decode(q.data(), k.data(), v.data(), out.data(), 1, 0, 1.0f);
    assert(out[0] == 999.0f);

    // null pointers: graceful return
    FastAttention::scaled_dot_product_decode(nullptr, k.data(), v.data(), out.data(), 1, 16, 1.0f);
    assert(out[0] == 999.0f);
}

void test_alias_and_prefill() {
    using namespace strata::kernels;

    const size_t head_dim = 16;
    const size_t seq_len = 4;
    std::vector<float> q(head_dim, 0.5f);
    std::vector<float> k(seq_len * head_dim, 0.2f);
    std::vector<float> v(seq_len * head_dim, 1.0f);
    std::vector<float> out1(head_dim, 0.0f);
    std::vector<float> out2(head_dim, 0.0f);

    FastAttention::scaled_dot_product_decode(
        q.data(), k.data(), v.data(), out1.data(),
        seq_len, head_dim, 0.25f
    );

    FastAttention::scaled_dot_product_attention_decode(
        q.data(), k.data(), v.data(), out2.data(),
        seq_len, head_dim, 0.25f
    );

    for (size_t d = 0; d < head_dim; ++d) {
        assert(std::abs(out1[d] - out2[d]) < 1e-6f);
    }

    // Prefill test
    std::vector<float> q_seq(seq_len * head_dim, 0.5f);
    std::vector<float> out_prefill(seq_len * head_dim, 0.0f);
    FastAttention::flash_attention_prefill(
        q_seq.data(), k.data(), v.data(), out_prefill.data(),
        seq_len, head_dim, 0.25f, true
    );
    for (size_t i = 0; i < seq_len * head_dim; ++i) {
        assert(std::abs(out_prefill[i] - 1.0f) < 1e-4f);
    }
}

void test_mask_isolation_with_negative_logits() {
    using namespace strata::kernels;

    const size_t head_dim = 16;
    const size_t seq_len = 3;

    // Construct q and k such that unmasked token dot product * scale produces a large negative logit (<= -88.0f)
    // Sanitized logit for unmasked token: -88.0f
    // Masked token: mask = -1e9f
    // With sanitization BEFORE mask addition:
    // unmasked score = -88.0f, masked score = 88.0f + (-1e9f) = -999999912.0f
    // Unmasked token completely dominates and output is 42.0f
    std::vector<float> q(head_dim, 1.0f);
    std::vector<float> k_cache(seq_len * head_dim, 0.0f);
    for (size_t d = 0; d < head_dim; ++d) {
        k_cache[0 * head_dim + d] = -10.0f; // unmasked token produces negative dot product (-160)
        k_cache[1 * head_dim + d] = 10.0f;  // masked token
        k_cache[2 * head_dim + d] = 10.0f;  // masked token
    }

    std::vector<float> v_cache(seq_len * head_dim, 0.0f);
    for (size_t d = 0; d < head_dim; ++d) {
        v_cache[0 * head_dim + d] = 42.0f; // target output from unmasked token
        v_cache[1 * head_dim + d] = 99.0f; // masked value
        v_cache[2 * head_dim + d] = 99.0f; // masked value
    }

    std::vector<float> mask = {0.0f, -1e9f, -1e9f};
    std::vector<float> out(head_dim, 0.0f);

    MathContext ctx;
    FastAttention::scaled_dot_product_decode(
        q.data(), k_cache.data(), v_cache.data(), out.data(),
        seq_len, head_dim, 1.0f, mask.data(), ctx
    );

    for (size_t d = 0; d < head_dim; ++d) {
        assert(std::abs(out[d] - 42.0f) < 1e-4f);
    }
}

void test_large_sequence_tiled_prefill_parity() {
    using namespace strata::kernels;

    const size_t head_dim = 64;
    const size_t seq_len = 256;

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> q(seq_len * head_dim);
    std::vector<float> k(seq_len * head_dim);
    std::vector<float> v(seq_len * head_dim);
    std::vector<float> out_tiled(seq_len * head_dim, 0.0f);
    std::vector<float> out_ref(seq_len * head_dim, 0.0f);

    for (size_t i = 0; i < seq_len * head_dim; ++i) {
        q[i] = dist(rng);
        k[i] = dist(rng);
        v[i] = dist(rng);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    MathContext ctx;

    // 1. Tiled FlashAttention-2 Prefill
    FastAttention::flash_attention_prefill(
        q.data(), k.data(), v.data(), out_tiled.data(),
        seq_len, head_dim, scale, true, ctx
    );

    // 2. Reference sequential decode per row
    for (size_t r = 0; r < seq_len; ++r) {
        const float* q_r = q.data() + r * head_dim;
        float* out_r = out_ref.data() + r * head_dim;
        FastAttention::scaled_dot_product_decode(
            q_r, k.data(), v.data(), out_r,
            r + 1, head_dim, scale, nullptr, ctx
        );
    }

    // Verify numerical parity across all 256 x 64 elements
    for (size_t i = 0; i < seq_len * head_dim; ++i) {
        float diff = std::abs(out_tiled[i] - out_ref[i]);
        assert(diff < 1e-4f);
    }
}

void test_mha_gqa_prefill() {
    using namespace strata::kernels;

    const size_t head_dim = 32;
    const size_t seq_len = 64;
    const size_t num_q_heads = 8;
    const size_t num_kv_heads = 2; // GQA 4:1

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> q(seq_len * num_q_heads * head_dim);
    std::vector<float> k(seq_len * num_kv_heads * head_dim);
    std::vector<float> v(seq_len * num_kv_heads * head_dim);
    std::vector<float> out(seq_len * num_q_heads * head_dim, 0.0f);

    for (auto& x : q) x = dist(rng);
    for (auto& x : k) x = dist(rng);
    for (auto& x : v) x = dist(rng);

    FastAttention::flash_attention_prefill_mha(
        q.data(), k.data(), v.data(), out.data(),
        seq_len, num_q_heads, num_kv_heads, head_dim,
        1.0f / std::sqrt(static_cast<float>(head_dim)), true
    );

    // Ensure non-zero, finite outputs
    for (float val : out) {
        assert(!std::isnan(val) && !std::isinf(val));
    }
}

int main() {
    std::cout << "Running Fast Attention unit tests..." << std::endl;
    test_basic_decode();
    test_reference_parity_and_simd();
    test_attention_masking();
    test_mask_isolation_with_negative_logits();
    test_bounds_and_nan_sanitization();
    test_multi_head_decode();
    test_edge_cases();
    test_alias_and_prefill();
    test_large_sequence_tiled_prefill_parity();
    test_mha_gqa_prefill();

    std::cout << "test_fast_attention passed" << std::endl;
    return 0;
}


