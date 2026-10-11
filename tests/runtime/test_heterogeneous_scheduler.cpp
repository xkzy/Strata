// tests/runtime/test_heterogeneous_scheduler.cpp
#include "strata/runtime/heterogeneous_scheduler.hpp"
#include <cassert>
#include <cstdio>

using namespace strata::runtime;

void test_moe_activation_offload_decision() {
    std::printf("[Test 1/3] Testing MoE Activation-Driven Offload vs PCIe Weight Transfer...\n");

    HardwareProfile hw;
    hw.pcie_bandwidth_mb_s = 15754.0; // PCIe 3.0 x16 (~15.7 GB/s)
    hw.gpu_tflops = 30.0;
    hw.cpu_gflops = 800.0;             // Multi-core CPU AVX2

    HeterogeneousScheduler scheduler(hw);

    // Qwen / MiMo style MoE layer: hidden=3584, intermediate=2048, batch_size=1
    // Total weight size per expert: 3 * 3584 * 2048 * 2 bytes = 44 MB!
    // Activation size per token: 2 * 1 * 3584 * 4 bytes = 28 KB!
    auto decision = scheduler.evaluate_moe_expert_dispatch(
        /*batch_size=*/1,
        /*hidden_dim=*/3584,
        /*intermediate_dim=*/2048,
        /*bytes_per_weight=*/2
    );

    std::printf("  GPU Transfer Latency: %.2f us, CPU Compute Latency: %.2f us (Speedup: %.2fx)\n",
                decision.estimated_gpu_transfer_latency_us,
                decision.estimated_cpu_latency_us,
                decision.speedup_ratio);

    assert(decision.offload_to_cpu == true);
    assert(decision.target_device == DeviceAffinity::kHybridAsync);
    assert(decision.estimated_gpu_transfer_latency_us > decision.estimated_cpu_latency_us);
    assert(decision.speedup_ratio > 1.0);
    std::printf("  [PASS] MoE Activation Offload verified!\n");
}

void test_chunked_prefill_piggybacking() {
    std::printf("[Test 2/3] Testing Chunked Prefill & Decode Piggyback Co-Scheduling...\n");

    HeterogeneousScheduler scheduler;

    // Case A: Long prompt with 2048 tokens remaining and 4 active decode streams
    auto sched_long = scheduler.compute_piggyback_schedule(
        /*prompt_tokens_remaining=*/2048,
        /*active_decode_count=*/4,
        /*hidden_dim=*/3584
    );

    std::printf("  Long Prefill + Decode: Chunk=%zu, Decodes=%zu, Total=%zu, AI=%.2f FLOPs/byte\n",
                sched_long.prefill_chunk_tokens,
                sched_long.num_decode_streams,
                sched_long.total_batch_tokens,
                sched_long.estimated_arithmetic_intensity);

    assert(sched_long.prefill_chunk_tokens == 256);
    assert(sched_long.total_batch_tokens == 260);
    assert(sched_long.saturate_compute_cores == true);

    // Case B: Short prompt with 64 tokens remaining and 2 decode streams
    auto sched_short = scheduler.compute_piggyback_schedule(
        /*prompt_tokens_remaining=*/64,
        /*active_decode_count=*/2,
        /*hidden_dim=*/3584
    );

    assert(sched_short.prefill_chunk_tokens == 64);
    assert(sched_short.total_batch_tokens == 66);
    std::printf("  [PASS] Chunked prefill piggybacking verified!\n");
}

void test_asymmetric_speculative_pipeline() {
    std::printf("[Test 3/3] Testing Heterogeneous Speculative Pipeline Planning...\n");

    HeterogeneousScheduler scheduler;

    // Case A: High acceptance rate (85%)
    auto plan_high = scheduler.plan_speculative_pipeline(
        /*draft_acceptance_rate=*/0.85,
        /*target_vocab_size=*/151643
    );

    std::printf("  High Acceptance (85%%): Draft K=%zu, Drafter=%s, Speedup=%.2fx\n",
                plan_high.draft_k,
                plan_high.drafter_device == DeviceAffinity::kCPUWorkerPool ? "CPU" : "GPU",
                plan_high.estimated_speedup);

    assert(plan_high.draft_k == 6);
    assert(plan_high.drafter_device == DeviceAffinity::kCPUWorkerPool);
    assert(plan_high.verifier_device == DeviceAffinity::kGPU);
    assert(plan_high.estimated_speedup > 2.0);

    // Case B: Moderate acceptance rate (65%)
    auto plan_mod = scheduler.plan_speculative_pipeline(
        /*draft_acceptance_rate=*/0.65,
        /*target_vocab_size=*/151643
    );
    assert(plan_mod.draft_k == 4);
    assert(plan_mod.estimated_speedup > 1.5);

    std::printf("  [PASS] Asymmetric speculative pipeline planning verified!\n");
}

int main() {
    std::printf("=== Running Strata Heterogeneous Task Scheduler Test Suite ===\n");
    test_moe_activation_offload_decision();
    test_chunked_prefill_piggybacking();
    test_asymmetric_speculative_pipeline();
    std::printf("=== All Heterogeneous Task Scheduler Tests Passed Successfully! ===\n");
    return 0;
}
