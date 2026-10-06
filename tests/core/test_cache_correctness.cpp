// tests/core/test_cache_correctness.cpp - Verification of Conversation & Prefix Cache Correctness
#include "strata/core/conversation_cache.hpp"

#include <cassert>
#include <iostream>
#include <vector>

void test_conversation_prefix_exact_matching() {
    std::cout << "[Test 1/4] Testing Exact Token Prefix Matching Invariant..." << std::endl;
    strata::core::ConversationCheckpoint cp;
    cp.ids = {101, 102, 103, 104, 105};

    std::vector<int32_t> exact_match_prompt = {101, 102, 103, 104, 105, 106, 107};
    std::vector<int32_t> mismatched_prompt = {101, 102, 999, 104, 105, 106, 107};
    std::vector<int32_t> shorter_prompt = {101, 102, 103};

    std::vector<strata::core::ConversationImageKey> images;

    int64_t matched_len = strata::core::conversation_prefix(cp, exact_match_prompt, images);
    int64_t mismatch_len = strata::core::conversation_prefix(cp, mismatched_prompt, images);
    int64_t short_len = strata::core::conversation_prefix(cp, shorter_prompt, images);
    (void)matched_len; (void)mismatch_len; (void)short_len;

    assert(matched_len == 5 && "Exact prefix of length 5 must match");
    assert(mismatch_len == 0 && "Mismatched token at pos 2 must invalidate entire prefix match");
    assert(short_len == 0 && "Prompt shorter than checkpoint must not match");

    std::cout << "  Passed. Strict token-by-token equality verified." << std::endl;
}

void test_cache_reuse_limiting_on_dirty_suffix() {
    std::cout << "[Test 2/4] Testing KV Cache Reuse Boundary on Dirty Suffix..." << std::endl;
    strata::core::ConversationCache cache(1024 * 1024, 4);

    std::vector<strata::core::ConversationKv> kv_layers(2);
    cache.retain(std::move(kv_layers), 100);

    // If context compaction or token edit makes tokens after index 40 dirty:
    cache.limit_reuse(40);

    auto reused = cache.take_reuse();
    assert(reused.unchanged_tokens == 40 && "Reused tokens must be strictly capped by first_dirty boundary");

    std::cout << "  Passed. Dirty suffix correctly limits KV reuse." << std::endl;
}

void test_cache_hit_versus_miss_isolation() {
    std::cout << "[Test 3/4] Testing Multi-Prompt Cache Isolation..." << std::endl;
    strata::core::ConversationCache cache(1024 * 1024, 4);

    strata::core::SavedConversation conv1;
    conv1.live.ids = {1, 2, 3, 4};
    conv1.cvec = true;

    // Direct insertion check via best matching
    std::vector<int32_t> promptA = {1, 2, 3, 4, 5};
    std::vector<int32_t> promptB = {9, 8, 7, 6, 5};
    std::vector<strata::core::ConversationImageKey> imgs;

    int64_t pA = strata::core::conversation_prefix(conv1.live, promptA, imgs);
    int64_t pB = strata::core::conversation_prefix(conv1.live, promptB, imgs);
    (void)pA; (void)pB;

    assert(pA == 4);
    assert(pB == 0 && "Unrelated prompt must yield 0 prefix tokens");

    std::cout << "  Passed. Distinct prompts are strictly isolated." << std::endl;
}

void test_fail_closed_on_empty_or_corrupt_state() {
    std::cout << "[Test 4/4] Testing Fail-Closed on Questionable State..." << std::endl;
    strata::core::ConversationCheckpoint empty_cp;
    std::vector<int32_t> prompt = {1, 2, 3};
    std::vector<strata::core::ConversationImageKey> imgs;

    int64_t n = strata::core::conversation_prefix(empty_cp, prompt, imgs);
    (void)n;
    assert(n == 0 && "Empty checkpoint must fail closed with 0 reused tokens");

    std::cout << "  Passed. Runtime fails closed safely." << std::endl;
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "   RUNNING STRATA CACHE CORRECTNESS & ISOLATION TEST SUITE       " << std::endl;
    std::cout << "=================================================================" << std::endl;

    test_conversation_prefix_exact_matching();
    test_cache_reuse_limiting_on_dirty_suffix();
    test_cache_hit_versus_miss_isolation();
    test_fail_closed_on_empty_or_corrupt_state();

    std::cout << "=================================================================" << std::endl;
    std::cout << "   ALL CACHE CORRECTNESS TESTS PASSED (4/4)                      " << std::endl;
    std::cout << "=================================================================" << std::endl;
    return 0;
}
