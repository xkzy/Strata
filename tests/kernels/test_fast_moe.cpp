// tests/kernels/test_fast_moe.cpp
#include "strata/kernels/fast_moe.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

void test_basic_route_topk() {
    using namespace strata::kernels;

    const size_t num_experts = 8;
    const size_t top_k = 2;
    std::vector<float> logits = {0.1f, 2.5f, -1.0f, 0.4f, 3.2f, 0.0f, -0.5f, 1.2f};
    std::vector<int32_t> selected_experts(top_k);
    std::vector<float> selected_weights(top_k);

    FastMoE::route_topk(
        logits.data(), num_experts, top_k,
        selected_experts.data(), selected_weights.data()
    );

    assert(selected_experts[0] == 4); // Highest is index 4 (3.2)
    assert(selected_experts[1] == 1); // Second highest is index 1 (2.5)
    assert(selected_weights[0] > selected_weights[1]);
    assert(std::abs((selected_weights[0] + selected_weights[1]) - 1.0f) < 1e-4f);
}

void test_reference_parity_and_sorting() {
    using namespace strata::kernels;

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<size_t> expert_counts = {4, 8, 16, 64, 128};
    std::vector<size_t> ks = {1, 2, 4, 8};

    for (size_t num_experts : expert_counts) {
        for (size_t k : ks) {
            if (k > num_experts) continue;

            std::vector<float> logits(num_experts);
            for (size_t e = 0; e < num_experts; ++e) {
                logits[e] = dist(rng);
            }

            std::vector<int32_t> out_indices(k);
            std::vector<float> out_weights(k);
            MathContext ctx(MathPolicy::kBalanced);

            FastMoE::route_topk(
                logits.data(), num_experts, k,
                out_indices.data(), out_weights.data(), ctx
            );

            // Compute scalar full sort reference
            struct ExpertItem {
                int32_t idx;
                float score;
            };
            std::vector<ExpertItem> ref_scores(num_experts);
            for (size_t e = 0; e < num_experts; ++e) {
                ref_scores[e] = {static_cast<int32_t>(e), sanitize_logit(logits[e], ctx)};
            }
            std::stable_sort(ref_scores.begin(), ref_scores.end(),
                             [](const ExpertItem& a, const ExpertItem& b) {
                                 return a.score > b.score;
                             });

            // Compare top-k indices
            for (size_t i = 0; i < k; ++i) {
                assert(out_indices[i] == ref_scores[i].idx);
            }

            // Compare weights
            float max_score = ref_scores[0].score;
            float sum_exp = 0.0f;
            std::vector<float> ref_weights(k);
            for (size_t i = 0; i < k; ++i) {
                ref_weights[i] = std::exp(ref_scores[i].score - max_score);
                sum_exp += ref_weights[i];
            }
            float inv_sum = 1.0f / (sum_exp + 1e-9f);
            for (size_t i = 0; i < k; ++i) {
                ref_weights[i] *= inv_sum;
                assert(std::abs(out_weights[i] - ref_weights[i]) < 1e-5f);
            }

            // Check sum of weights == 1.0
            float total_w = 0.0f;
            for (size_t i = 0; i < k; ++i) total_w += out_weights[i];
            assert(std::abs(total_w - 1.0f) < 1e-4f);
        }
    }
}

void test_batch_routing() {
    using namespace strata::kernels;

    const size_t num_tokens = 4;
    const size_t num_experts = 8;
    const size_t top_k = 2;

    std::vector<float> logits = {
        0.1f, 2.5f, -1.0f, 0.4f, 3.2f, 0.0f, -0.5f, 1.2f,  // Token 0: top experts 4, 1
        5.0f, 1.0f,  2.0f, 3.0f, 0.0f, 4.0f, -2.0f, 0.5f,  // Token 1: top experts 0, 5
       -1.0f, 0.0f,  0.5f, 0.2f, 0.8f, 1.5f,  2.0f, 1.9f,  // Token 2: top experts 6, 7
        1.1f, 1.2f,  1.3f, 1.4f, 1.5f, 1.6f,  1.7f, 1.8f   // Token 3: top experts 7, 6
    };

    std::vector<int32_t> out_indices(num_tokens * top_k);
    std::vector<float> out_weights(num_tokens * top_k);

    FastMoE::route_topk_batch(
        logits.data(), num_tokens, num_experts, top_k,
        out_indices.data(), out_weights.data()
    );

    // Token 0
    assert(out_indices[0] == 4 && out_indices[1] == 1);
    // Token 1
    assert(out_indices[2] == 0 && out_indices[3] == 5);
    // Token 2
    assert(out_indices[4] == 6 && out_indices[5] == 7);
    // Token 3
    assert(out_indices[6] == 7 && out_indices[7] == 6);

    for (size_t t = 0; t < num_tokens; ++t) {
        float sum_w = out_weights[t * top_k + 0] + out_weights[t * top_k + 1];
        assert(std::abs(sum_w - 1.0f) < 1e-4f);
    }
}

void test_weighted_aggregate() {
    using namespace strata::kernels;

    const size_t top_k = 3;
    const size_t dim = 64;

    std::vector<float> exp0(dim, 1.0f);
    std::vector<float> exp1(dim, 2.0f);
    std::vector<float> exp2(dim, 3.0f);

    const float* expert_ptrs[top_k] = {exp0.data(), exp1.data(), exp2.data()};
    std::vector<float> weights = {0.2f, 0.5f, 0.3f};
    std::vector<float> out(dim, 0.0f);

    // Test pointer array aggregation
    FastMoE::weighted_aggregate(expert_ptrs, weights.data(), top_k, dim, out.data());

    // Expected value: 0.2*1.0 + 0.5*2.0 + 0.3*3.0 = 0.2 + 1.0 + 0.9 = 2.1f
    for (size_t d = 0; d < dim; ++d) {
        assert(std::abs(out[d] - 2.1f) < 1e-4f);
    }

    // Test contiguous buffer aggregation
    std::vector<float> contig_experts(top_k * dim);
    std::copy(exp0.begin(), exp0.end(), contig_experts.begin() + 0 * dim);
    std::copy(exp1.begin(), exp1.end(), contig_experts.begin() + 1 * dim);
    std::copy(exp2.begin(), exp2.end(), contig_experts.begin() + 2 * dim);

    std::vector<float> out_contig(dim, 0.0f);
    FastMoE::weighted_aggregate(contig_experts.data(), weights.data(), top_k, dim, out_contig.data());

    for (size_t d = 0; d < dim; ++d) {
        assert(std::abs(out_contig[d] - 2.1f) < 1e-4f);
    }

    // Test alias weighted_expert_aggregate
    std::vector<float> out_alias(dim, 0.0f);
    FastMoE::weighted_expert_aggregate(expert_ptrs, weights.data(), top_k, dim, out_alias.data());
    for (size_t d = 0; d < dim; ++d) {
        assert(std::abs(out_alias[d] - 2.1f) < 1e-4f);
    }
}

void test_dispatch_permutation() {
    using namespace strata::kernels;

    const size_t num_tokens = 3;
    const size_t top_k = 2;
    const size_t num_experts = 4;
    const size_t dim = 16;

    // Tokens route to experts:
    // Token 0 -> experts [1, 3]
    // Token 1 -> experts [0, 1]
    // Token 2 -> experts [3, 0]
    std::vector<int32_t> expert_indices = {1, 3, 0, 1, 3, 0};
    std::vector<float> weights = {
        0.6f, 0.4f, // Token 0
        0.7f, 0.3f, // Token 1
        0.8f, 0.2f  // Token 2
    };

    std::vector<int32_t> expert_counts(num_experts, 0);
    std::vector<int32_t> expert_offsets(num_experts + 1, 0);
    std::vector<int32_t> token_indices(num_tokens * top_k, 0);
    std::vector<int32_t> slot_indices(num_tokens * top_k, 0);

    FastMoE::compute_dispatch_permutation(
        expert_indices.data(), num_tokens, top_k, num_experts,
        expert_counts.data(), expert_offsets.data(),
        token_indices.data(), slot_indices.data()
    );

    // Expert 0: assigned from Token 1 (slot 0) and Token 2 (slot 1) -> count = 2
    // Expert 1: assigned from Token 0 (slot 0) and Token 1 (slot 1) -> count = 2
    // Expert 2: count = 0
    // Expert 3: assigned from Token 0 (slot 1) and Token 2 (slot 0) -> count = 2
    assert(expert_counts[0] == 2);
    assert(expert_counts[1] == 2);
    assert(expert_counts[2] == 0);
    assert(expert_counts[3] == 2);

    assert(expert_offsets[0] == 0);
    assert(expert_offsets[1] == 2);
    assert(expert_offsets[2] == 4);
    assert(expert_offsets[3] == 4);
    assert(expert_offsets[4] == 6);

    // Verify tokens dispatched to expert 0: Token 1 and Token 2
    assert(token_indices[0] == 1 && slot_indices[0] == 0);
    assert(token_indices[1] == 2 && slot_indices[1] == 1);

    // Verify tokens dispatched to expert 1: Token 0 and Token 1
    assert(token_indices[2] == 0 && slot_indices[2] == 0);
    assert(token_indices[3] == 1 && slot_indices[3] == 1);

    // Expert 2 has 0 tokens (offsets 4 to 4)

    // Verify tokens dispatched to expert 3: Token 0 and Token 2
    assert(token_indices[4] == 0 && slot_indices[4] == 1);
    assert(token_indices[5] == 2 && slot_indices[5] == 0);

    // Now test token permutation
    std::vector<float> tokens(num_tokens * dim);
    for (size_t t = 0; t < num_tokens; ++t) {
        for (size_t d = 0; d < dim; ++d) {
            tokens[t * dim + d] = static_cast<float>((t + 1) * 100 + d);
        }
    }

    std::vector<float> permuted(num_tokens * top_k * dim);
    FastMoE::permute_tokens(
        tokens.data(), token_indices.data(),
        num_tokens * top_k, dim, permuted.data()
    );

    // Verify permuted contents match the dispatched token rows
    for (size_t i = 0; i < num_tokens * top_k; ++i) {
        int32_t t = token_indices[i];
        for (size_t d = 0; d < dim; ++d) {
            assert(permuted[i * dim + d] == tokens[t * dim + d]);
        }
    }

    // Simulate expert processing: for simplicity, let expert output = 2 * input
    std::vector<float> dispatched_outputs(num_tokens * top_k * dim);
    for (size_t i = 0; i < num_tokens * top_k * dim; ++i) {
        dispatched_outputs[i] = permuted[i] * 2.0f;
    }

    std::vector<float> combined(num_tokens * dim, 0.0f);
    FastMoE::combine_dispatched(
        dispatched_outputs.data(), token_indices.data(), slot_indices.data(),
        weights.data(), num_tokens * top_k, num_tokens, top_k, dim, combined.data()
    );

    // Verify combined result: for token t, each expert output is 2 * token[t],
    // so combined = 2 * token[t] * (w0 + w1) = 2 * token[t] * 1.0 = 2 * token[t].
    for (size_t t = 0; t < num_tokens; ++t) {
        for (size_t d = 0; d < dim; ++d) {
            float expected = 2.0f * tokens[t * dim + d];
            assert(std::abs(combined[t * dim + d] - expected) < 1e-4f);
        }
    }
}

void test_bounds_and_nan_sanitization() {
    using namespace strata::kernels;

    const size_t num_experts = 4;
    const size_t top_k = 2;

    // Logits with NaN and extreme numbers
    float nan_val = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> logits = {nan_val, 1000.0f, -1000.0f, 50.0f};
    std::vector<int32_t> out_indices(top_k);
    std::vector<float> out_weights(top_k);

    MathContext ctx(MathPolicy::kBalanced);
    FastMoE::route_topk(
        logits.data(), num_experts, top_k,
        out_indices.data(), out_weights.data(), ctx
    );

    // Sanitization clamps 1000.0f to 88.0f and nan_val to -88.0f, -1000.0f to -88.0f
    // Index 1 (clamped to 88.0f) should be top-1
    // Index 3 (50.0f) should be top-2
    assert(out_indices[0] == 1);
    assert(out_indices[1] == 3);

    for (size_t k = 0; k < top_k; ++k) {
        assert(!std::isnan(out_weights[k]));
        assert(!std::isinf(out_weights[k]));
    }
    assert(std::abs((out_weights[0] + out_weights[1]) - 1.0f) < 1e-4f);
}

void test_edge_cases() {
    using namespace strata::kernels;

    // Zero experts or zero top_k
    std::vector<float> logits = {1.0f, 2.0f};
    int32_t idx = 99;
    float w = 99.0f;
    FastMoE::route_topk(logits.data(), 0, 1, &idx, &w);
    assert(idx == 99 && w == 99.0f);

    FastMoE::route_topk(logits.data(), 2, 0, &idx, &w);
    assert(idx == 99 && w == 99.0f);

    // Null pointers
    FastMoE::route_topk(nullptr, 2, 1, &idx, &w);
    assert(idx == 99 && w == 99.0f);

    // top_k > num_experts: clamps top_k to num_experts
    std::vector<int32_t> idxs(5, -1);
    std::vector<float> ws(5, -1.0f);
    FastMoE::route_topk(logits.data(), 2, 5, idxs.data(), ws.data());
    assert(idxs[0] == 1 && idxs[1] == 0);
    assert(idxs[2] == -1); // untouched

    // Aggregate with top_k == 0 or dim == 0
    float out_val = 123.0f;
    const float* ptr = logits.data();
    FastMoE::weighted_aggregate(&ptr, ws.data(), 0, 1, &out_val);
    assert(out_val == 123.0f);

    FastMoE::weighted_aggregate(&ptr, ws.data(), 1, 0, &out_val);
    assert(out_val == 123.0f);

    FastMoE::weighted_aggregate(static_cast<const float* const*>(nullptr), ws.data(), 1, 1, &out_val);
    assert(out_val == 123.0f);

    // Batch routing null / empty edge cases
    FastMoE::route_topk_batch(nullptr, 0, 2, 1, idxs.data(), ws.data());
    FastMoE::weighted_aggregate_batch(nullptr, ws.data(), 0, 1, 1, &out_val);

    // Dispatch permutation edge case: out-of-bounds / negative expert ID
    std::vector<int32_t> invalid_experts = {-1, 99};
    std::vector<int32_t> c(4, 0), o(5, 0), t_idx(2, -99), s_idx(2, -99);
    FastMoE::compute_dispatch_permutation(invalid_experts.data(), 1, 2, 4, c.data(), o.data(), t_idx.data(), s_idx.data());
    // Since all were out of range, counts are 0 and indices are -1
    for (int count : c) assert(count == 0);
    assert(t_idx[0] == -1 && t_idx[1] == -1);
    assert(s_idx[0] == -1 && s_idx[1] == -1);
}

int main() {
    std::cout << "Running Fast MoE unit tests..." << std::endl;
    test_basic_route_topk();
    test_reference_parity_and_sorting();
    test_batch_routing();
    test_weighted_aggregate();
    test_dispatch_permutation();
    test_bounds_and_nan_sanitization();
    test_edge_cases();

    std::cout << "test_fast_moe passed" << std::endl;
    return 0;
}
