// tests/kernels/test_fast_paged_kv.cpp
#include "strata/kernels/fast_attention.hpp"
#include "strata/kernels/fast_paged_kv.hpp"
#include "strata/kernels/fast_speculative.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace strata::kernels;

void test_paged_kv_lifecycle_and_cow() {
    std::printf("[Test 1/5] Testing FastPagedKVCache Lifecycle & Copy-on-Write...\n");

    PagedKVCacheConfig cfg;
    cfg.block_size = 16;
    cfg.num_layers = 1;
    cfg.num_kv_heads = 4;
    cfg.head_dim = 64;
    cfg.initial_pool_blocks = 32;
    cfg.max_pool_blocks = 128;

    FastPagedKVCache cache(cfg);

    // 1. Create parent sequence
    uint64_t seq_parent = 101;
    assert(cache.create_sequence(seq_parent));

    // Append 30 tokens to parent (fills 1 block of 16 + 14 tokens in 2nd block = 2 blocks total)
    const size_t token_dim = cfg.num_kv_heads * cfg.head_dim;
    std::vector<float> k_tok(token_dim, 1.0f);
    std::vector<float> v_tok(token_dim, 2.0f);

    for (size_t t = 0; t < 30; ++t) {
        k_tok[0] = static_cast<float>(t);
        v_tok[0] = static_cast<float>(t * 2);
        assert(cache.append_token(seq_parent, k_tok.data(), v_tok.data()));
    }

    assert(cache.sequence_length(seq_parent) == 30);
    auto stats1 = cache.stats();
    assert(stats1.allocated_blocks == 2);
    assert(stats1.shared_cow_blocks == 0);

    // 2. Fork child sequence using Copy-on-Write
    uint64_t seq_child = 102;
    assert(cache.fork_sequence(seq_parent, seq_child));
    assert(cache.sequence_length(seq_child) == 30);

    auto stats2 = cache.stats();
    // After COW fork, no new blocks are allocated, both blocks are shared!
    assert(stats2.allocated_blocks == 2);
    assert(stats2.shared_cow_blocks == 2);
    assert(stats2.memory_saved_by_cow_mb > 0.0);

    // Verify token data matches in both sequences
    const float* p_k = cache.resolve_token_k(seq_parent, 10, 0);
    const float* c_k = cache.resolve_token_k(seq_child, 10, 0);
    assert(p_k != nullptr && c_k != nullptr);
    assert(p_k[0] == 10.0f && c_k[0] == 10.0f);

    // 3. Append to child: should trigger Copy-on-Write split of the tail block
    k_tok[0] = 999.0f;
    v_tok[0] = 888.0f;
    assert(cache.append_token(seq_child, k_tok.data(), v_tok.data()));

    assert(cache.sequence_length(seq_child) == 31);
    assert(cache.sequence_length(seq_parent) == 30); // Parent remains intact

    // Check parent token 10 is still 10.0
    assert(cache.resolve_token_k(seq_parent, 10, 0)[0] == 10.0f);
    // Check child newly appended token
    assert(cache.resolve_token_k(seq_child, 30, 0)[0] == 999.0f);

    // 4. Destroy parent sequence
    assert(cache.destroy_sequence(seq_parent));
    // Child sequence should still be valid!
    assert(cache.sequence_length(seq_child) == 31);
    assert(cache.resolve_token_k(seq_child, 10, 0)[0] == 10.0f);

    // 5. Destroy child sequence
    assert(cache.destroy_sequence(seq_child));
    auto stats3 = cache.stats();
    assert(stats3.allocated_blocks == 0);
    assert(stats3.free_blocks == stats3.total_blocks);

    std::printf("  -> FastPagedKVCache Lifecycle & COW: PASS (zero leaks, COW split verified)\n");
}

void test_paged_attention_parity() {
    std::printf("[Test 2/5] Testing PagedAttention vs Contiguous Attention Numerical Parity...\n");

    const size_t seq_len = 48; // spans 3 blocks of 16
    const size_t head_dim = 64;
    const size_t num_kv_heads = 1;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    PagedKVCacheConfig cfg;
    cfg.block_size = 16;
    cfg.num_layers = 1;
    cfg.num_kv_heads = num_kv_heads;
    cfg.head_dim = head_dim;

    FastPagedKVCache cache(cfg);
    uint64_t seq_id = 1;
    cache.create_sequence(seq_id);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> contiguous_k(seq_len * head_dim);
    std::vector<float> contiguous_v(seq_len * head_dim);

    for (size_t s = 0; s < seq_len; ++s) {
        std::vector<float> k_tok(head_dim);
        std::vector<float> v_tok(head_dim);
        for (size_t d = 0; d < head_dim; ++d) {
            k_tok[d] = dist(rng);
            v_tok[d] = dist(rng);
            contiguous_k[s * head_dim + d] = k_tok[d];
            contiguous_v[s * head_dim + d] = v_tok[d];
        }
        cache.append_token(seq_id, k_tok.data(), v_tok.data());
    }

    std::vector<float> q(head_dim);
    for (size_t d = 0; d < head_dim; ++d) {
        q[d] = dist(rng);
    }

    // 1. Reference contiguous decode
    std::vector<float> ref_out(head_dim, 0.0f);
    FastAttention::scaled_dot_product_decode(
        q.data(), contiguous_k.data(), contiguous_v.data(),
        ref_out.data(), seq_len, head_dim, scale);

    // 2. Paged decode
    std::vector<float> paged_out(head_dim, 0.0f);
    FastAttention::paged_scaled_dot_product_decode(
        q.data(), cache, seq_id, 0, paged_out.data(), scale);

    // Compare
    float max_diff = 0.0f;
    for (size_t d = 0; d < head_dim; ++d) {
        float diff = std::fabs(ref_out[d] - paged_out[d]);
        if (diff > max_diff) max_diff = diff;
    }

    std::printf("  -> Max discrepancy: %.2e\n", max_diff);
    assert(max_diff < 1e-4f);
    std::printf("  -> PagedAttention single-head parity: PASS\n");
}

void test_paged_gqa_parity() {
    std::printf("[Test 3/5] Testing Paged GQA Attention Parity...\n");

    const size_t seq_len = 64; // 4 blocks of 16
    const size_t num_heads = 8;
    const size_t num_kv_heads = 2; // GQA group size = 4
    const size_t head_dim = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    PagedKVCacheConfig cfg;
    cfg.block_size = 16;
    cfg.num_layers = 1;
    cfg.num_kv_heads = num_kv_heads;
    cfg.head_dim = head_dim;

    FastPagedKVCache cache(cfg);
    uint64_t seq_id = 2;
    cache.create_sequence(seq_id);

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const size_t kv_stride = num_kv_heads * head_dim;
    std::vector<float> contiguous_k(seq_len * kv_stride);
    std::vector<float> contiguous_v(seq_len * kv_stride);

    for (size_t s = 0; s < seq_len; ++s) {
        std::vector<float> k_tok(kv_stride);
        std::vector<float> v_tok(kv_stride);
        for (size_t i = 0; i < kv_stride; ++i) {
            k_tok[i] = dist(rng);
            v_tok[i] = dist(rng);
            contiguous_k[s * kv_stride + i] = k_tok[i];
            contiguous_v[s * kv_stride + i] = v_tok[i];
        }
        cache.append_token(seq_id, k_tok.data(), v_tok.data());
    }

    std::vector<float> q(num_heads * head_dim);
    for (size_t i = 0; i < num_heads * head_dim; ++i) {
        q[i] = dist(rng);
    }

    // 1. Reference COW / contiguous GQA decode
    std::vector<float> ref_out(num_heads * head_dim, 0.0f);
    FastAttention::cow_multi_head_scaled_dot_product_decode(
        q.data(), contiguous_k.data(), contiguous_v.data(), seq_len,
        nullptr, nullptr, 0,
        ref_out.data(), num_heads, num_kv_heads, head_dim, scale);

    // 2. Paged GQA decode
    std::vector<float> paged_out(num_heads * head_dim, 0.0f);
    FastAttention::paged_gqa_decode(
        q.data(), cache, seq_id, paged_out.data(),
        num_heads, num_kv_heads, head_dim, scale);

    float max_diff = 0.0f;
    for (size_t i = 0; i < num_heads * head_dim; ++i) {
        float diff = std::fabs(ref_out[i] - paged_out[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::printf("  -> Max GQA discrepancy: %.2e\n", max_diff);
    assert(max_diff < 1e-4f);
    std::printf("  -> Paged GQA Attention parity: PASS\n");
}

void test_speculative_verification() {
    std::printf("[Test 4/5] Testing FastSpeculativeVerifier (Greedy & Rejection Sampling)...\n");

    const size_t vocab_size = 32000;
    const size_t num_drafts = 4;

    // Simulate target model logits for 5 positions (4 drafts + 1 bonus position)
    std::vector<float> target_logits((num_drafts + 1) * vocab_size, 0.0f);

    // Set clear argmax peaks
    target_logits[0 * vocab_size + 42] = 10.0f;   // Pos 0 argmax = 42
    target_logits[1 * vocab_size + 105] = 10.0f;  // Pos 1 argmax = 105
    target_logits[2 * vocab_size + 777] = 10.0f;  // Pos 2 argmax = 777
    target_logits[3 * vocab_size + 999] = 10.0f;  // Pos 3 argmax = 999
    target_logits[4 * vocab_size + 1234] = 10.0f; // Pos 4 bonus argmax = 1234

    // Case A: All 4 drafts match!
    int32_t draft_perfect[4] = {42, 105, 777, 999};
    auto res_perfect = FastSpeculativeVerifier::verify_greedy(
        draft_perfect, num_drafts, target_logits.data(), vocab_size);

    assert(res_perfect.drafts_accepted == 4);
    assert(res_perfect.acceptance_rate == 1.0f);
    assert(res_perfect.terminal_token == 1234); // Got the bonus token!
    assert(res_perfect.effective_tokens_per_step == 5.0f);

    // Case B: Draft 2 mismatches (expected 777, draft has 888)
    int32_t draft_mismatch[4] = {42, 105, 888, 999};
    auto res_mismatch = FastSpeculativeVerifier::verify_greedy(
        draft_mismatch, num_drafts, target_logits.data(), vocab_size);

    assert(res_mismatch.drafts_accepted == 2);
    assert(res_mismatch.accepted_tokens.size() == 2);
    assert(res_mismatch.accepted_tokens[0] == 42 && res_mismatch.accepted_tokens[1] == 105);
    assert(res_mismatch.terminal_token == 777); // Correction token
    assert(res_mismatch.effective_tokens_per_step == 3.0f);

    std::printf("  -> Speculative Greedy Verification: PASS (All-match & Mismatch cases correct)\n");
}

void test_speculative_rejection_sampling() {
    std::printf("[Test 5/5] Testing Speculative Rejection Sampling...\n");

    const size_t vocab_size = 100;
    const size_t num_drafts = 2;

    std::vector<float> target_probs((num_drafts + 1) * vocab_size, 0.0f);
    std::vector<float> draft_probs(num_drafts * vocab_size, 0.0f);

    // Position 0: Perfect agreement
    target_probs[0 * vocab_size + 10] = 0.9f;
    draft_probs[0 * vocab_size + 10] = 0.9f;

    // Position 1: Strong disagreement
    target_probs[1 * vocab_size + 20] = 0.01f;
    draft_probs[1 * vocab_size + 20] = 0.95f;
    target_probs[1 * vocab_size + 30] = 0.95f;

    // Bonus pos
    target_probs[2 * vocab_size + 50] = 1.0f;

    int32_t drafts[2] = {10, 20};
    auto res = FastSpeculativeVerifier::verify_rejection_sampling(
        drafts, num_drafts, target_probs.data(), draft_probs.data(), vocab_size, 12345);

    assert(res.accepted_tokens.size() >= 1);
    assert(res.accepted_tokens[0] == 10);
    std::printf("  -> Speculative Rejection Sampling: PASS (drafts accepted: %zu)\n", res.drafts_accepted);
}

int main() {
    std::printf("=====================================================\n");
    std::printf(" Strata Paged KV & Speculative Inference Test Suite \n");
    std::printf("=====================================================\n");

    test_paged_kv_lifecycle_and_cow();
    test_paged_attention_parity();
    test_paged_gqa_parity();
    test_speculative_verification();
    test_speculative_rejection_sampling();

    std::printf("\n>> ALL PAGED KV & SPECULATIVE TESTS PASSED!\n");
    return 0;
}
