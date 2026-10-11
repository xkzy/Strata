// tests/runtime/test_in_flight_batcher.cpp
#include "strata/runtime/in_flight_batcher.hpp"
#include <cassert>
#include <cstdio>
#include <vector>

using namespace strata::runtime;
using namespace strata::kernels;

void test_in_flight_batcher_continuous_lifecycle() {
    std::printf("[Test 1/2] Testing InFlightBatcher Continuous Scheduling & Eviction...\n");

    // Configure Paged KV Cache
    PagedKVCacheConfig kv_cfg;
    kv_cfg.block_size = 16;
    kv_cfg.num_layers = 1;
    kv_cfg.num_kv_heads = 4;
    kv_cfg.head_dim = 64;
    auto cache = std::make_shared<FastPagedKVCache>(kv_cfg);

    // Configure Batcher: max 2 active requests, prefill chunk 4 tokens
    InFlightBatcherConfig b_cfg;
    b_cfg.max_batch_size = 2;
    b_cfg.prefill_chunk_size = 4;
    InFlightBatcher batcher(cache, b_cfg);

    // Submit Request 1: prompt length 6 tokens (requires 2 prefill chunks: 4 + 2), max 3 new tokens
    GenerationRequest r1;
    r1.request_id = 1;
    r1.prompt_tokens = {10, 11, 12, 13, 14, 15};
    r1.max_new_tokens = 3;
    r1.eos_token_id = 9999;
    assert(batcher.submit_request(r1));

    // Submit Request 2: prompt length 3 tokens (requires 1 prefill chunk), max 2 new tokens
    GenerationRequest r2;
    r2.request_id = 2;
    r2.prompt_tokens = {20, 21, 22};
    r2.max_new_tokens = 2;
    r2.eos_token_id = 9999;
    assert(batcher.submit_request(r2));

    // Submit Request 3: will wait in pending queue because max_batch_size = 2
    GenerationRequest r3;
    r3.request_id = 3;
    r3.prompt_tokens = {30, 31};
    r3.max_new_tokens = 2;
    r3.eos_token_id = 9999;
    assert(batcher.submit_request(r3));

    auto m0 = batcher.metrics();
    assert(m0.pending_requests == 3);
    assert(m0.active_requests == 0);

    // Iteration 1: Admits r1 and r2 into active batch.
    // r1 processes 4 prompt tokens (2 remaining). Phase remains kPrefill.
    // r2 processes 3 prompt tokens (0 remaining). Transitions to kDecode!
    auto step1 = batcher.step();
    assert(step1.num_prefill_requests == 2);
    assert(step1.prefill_tokens_processed == 7);
    assert(step1.newly_generated_tokens.empty());

    // Iteration 2:
    // r1 finishes prefill (processes 2 tokens), transitions to kDecode.
    // r2 decodes token 1!
    auto step2 = batcher.step();
    assert(step2.num_prefill_requests == 1);
    assert(step2.num_decode_requests == 1);
    assert(step2.newly_generated_tokens.size() == 1);
    assert(step2.newly_generated_tokens[0].first == 2); // r2 generated

    // Iteration 3:
    // r1 decodes token 1!
    // r2 decodes token 2 (hits max_new_tokens = 2)! Phase becomes kFinished.
    auto step3 = batcher.step();
    assert(step3.num_decode_requests == 2);
    assert(step3.newly_generated_tokens.size() == 2);

    // Iteration 4:
    // Evicts r2 (now completed). This frees a slot!
    // Admits r3 from pending queue!
    // r1 decodes token 2.
    // r3 starts prefill (processes 2 tokens, completes prefill, transitions to kDecode).
    auto step4 = batcher.step();
    assert(step4.completed_request_ids.size() == 1);
    assert(step4.completed_request_ids[0] == 2); // r2 evicted!

    // Verify r2 is retrievable as completed
    GenerationRequest completed_r2;
    assert(batcher.get_completed_request(2, completed_r2));
    assert(completed_r2.generated_tokens.size() == 2);

    // Run remaining steps until everything completes
    size_t loop_count = 0;
    while (batcher.has_active_or_pending_work() && loop_count < 20) {
        batcher.step();
        loop_count++;
    }

    assert(!batcher.has_active_or_pending_work());
    auto final_metrics = batcher.metrics();
    assert(final_metrics.total_completed_requests == 3);
    assert(final_metrics.pending_requests == 0);
    assert(final_metrics.active_requests == 0);

    // Paged KV cache blocks must be fully reclaimed back to pool
    auto cache_stats = cache->stats();
    assert(cache_stats.allocated_blocks == 0);

    std::printf("  -> Continuous scheduling, chunked prefill, and dynamic eviction: PASS\n");
}

void test_in_flight_early_eos_and_cancellation() {
    std::printf("[Test 2/2] Testing Early EOS Termination & Cancellation...\n");

    auto cache = std::make_shared<FastPagedKVCache>();
    InFlightBatcher batcher(cache);

    GenerationRequest req;
    req.request_id = 99;
    req.prompt_tokens = {1, 2, 3};
    req.max_new_tokens = 100;
    req.eos_token_id = 555;
    assert(batcher.submit_request(req));

    // Custom token predictor that emits 555 (EOS) on step 2
    int step_cnt = 0;
    auto predictor = [&](uint64_t, const std::vector<int32_t>&) -> int32_t {
        step_cnt++;
        if (step_cnt >= 2) return 555; // Trigger early EOS
        return 42;
    };

    // Step 1: prefill
    batcher.step(predictor);
    // Step 2: decode token 42
    batcher.step(predictor);
    // Step 3: decode token 555 (EOS reached)
    batcher.step(predictor);
    // Step 4: eviction
    auto res4 = batcher.step(predictor);
    assert(res4.completed_request_ids.size() == 1);
    assert(res4.completed_request_ids[0] == 99);

    GenerationRequest done;
    assert(batcher.get_completed_request(99, done));
    assert(done.generated_tokens.size() == 2);
    assert(done.generated_tokens.back() == 555);

    std::printf("  -> Early EOS eviction: PASS\n");
}

int main() {
    std::printf("=====================================================\n");
    std::printf(" Strata In-Flight Continuous Batching Test Suite     \n");
    std::printf("=====================================================\n");

    test_in_flight_batcher_continuous_lifecycle();
    test_in_flight_early_eos_and_cancellation();

    std::printf("\n>> ALL IN-FLIGHT BATCHING TESTS PASSED!\n");
    return 0;
}
