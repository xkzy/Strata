// include/strata/runtime/heterogeneous_scheduler.hpp
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::runtime {

enum class DeviceAffinity {
    kGPU = 0,               // Discrete GPU (Tensor Cores / High Memory Bandwidth)
    kCPUWorkerPool = 1,     // Host Multi-core CPU OpenMP Thread Pool (AVX2/AVX-512)
    kAPUShared = 2,         // Integrated GPU / APU with unified GTT memory
    kHybridAsync = 3        // Split execution (GPU attention + CPU activation-offloaded experts)
};

enum class TaskType {
    kPrefill = 0,           // Compute-bound prompt prefill
    kDecode = 1,            // Memory-bandwidth bound autoregressive generation
    kMoEExpert = 2,         // Sparse mixture-of-experts matrix multiplication
    kSpeculativeDraft = 3,  // Fast candidate token generation
    kSymbolicCAS = 4        // Independent symbolic algebra / logic verification
};

struct HardwareProfile {
    double pcie_bandwidth_mb_s = 15754.0;   // PCIe 3.0 x16 ~ 15.75 GB/s default
    double gpu_tflops = 30.0;               // GPU peak FP16 TFLOPs
    double cpu_gflops = 800.0;              // CPU peak AVX2/AVX-512 GFLOPs
    double vram_bandwidth_gb_s = 448.0;     // GPU VRAM bandwidth in GB/s
    double ram_bandwidth_gb_s = 50.0;       // Host RAM bandwidth in GB/s
    size_t cpu_worker_threads = 8;
};

struct MoEOffloadDecision {
    DeviceAffinity target_device = DeviceAffinity::kGPU;
    bool offload_to_cpu = false;
    double estimated_cpu_latency_us = 0.0;
    double estimated_gpu_transfer_latency_us = 0.0;
    double speedup_ratio = 1.0;
    std::string rationale;
};

struct PiggybackSchedule {
    size_t prefill_chunk_tokens = 0;
    size_t num_decode_streams = 0;
    size_t total_batch_tokens = 0;
    double estimated_arithmetic_intensity = 0.0; // FLOPs per byte
    bool saturate_compute_cores = false;
};

struct SpeculativePipelinePlan {
    size_t draft_k = 4;
    DeviceAffinity drafter_device = DeviceAffinity::kCPUWorkerPool;
    DeviceAffinity verifier_device = DeviceAffinity::kGPU;
    bool overlap_cpu_gpu = true;
    double estimated_speedup = 1.0;
};

/// High-Performance Heterogeneous Task Distribution Scheduler
/// Optimizes resource utilization across discrete GPU, PCIe bus, and multi-core CPU.
class HeterogeneousScheduler {
public:
    explicit HeterogeneousScheduler(const HardwareProfile& profile = HardwareProfile())
        : profile_(profile) {}

    ~HeterogeneousScheduler() = default;

    void update_hardware_profile(const HardwareProfile& profile) {
        profile_ = profile;
    }

    const HardwareProfile& profile() const noexcept {
        return profile_;
    }

    /// Evaluates whether an MoE expert should be computed on CPU via activation offloading
    /// or transferred over PCIe to GPU VRAM based on the Fiddler & KTransformers principle.
    MoEOffloadDecision evaluate_moe_expert_dispatch(
        size_t batch_size,
        size_t hidden_dim,
        size_t intermediate_dim,
        size_t bytes_per_weight = 2) const {

        MoEOffloadDecision decision;

        // Weight matrix size (Gate + Up + Down projections) in bytes
        // Each expert has 3 projection matrices: gate(h, d), up(h, d), down(d, h) -> 3 * h * d * bytes
        double weight_bytes = 3.0 * static_cast<double>(hidden_dim * intermediate_dim * bytes_per_weight);

        // Activation size to transfer over PCIe (input activation X: batch * hidden_dim * 4 bytes FP32)
        double activation_bytes = 2.0 * static_cast<double>(batch_size * hidden_dim * sizeof(float)); // H2D and D2H

        // FLOPs required for 1 expert: 2 * 3 * batch * hidden * intermediate
        double expert_flops = 6.0 * static_cast<double>(batch_size * hidden_dim * intermediate_dim);

        // 1. PCIe weight transfer time (swapping full weights into GPU VRAM)
        double pcie_bw_bytes_per_us = (profile_.pcie_bandwidth_mb_s * 1e6) / 1e6; // bytes per us
        double transfer_weight_us = weight_bytes / pcie_bw_bytes_per_us;

        // GPU execution time (FP16 compute)
        double gpu_flops_per_us = (profile_.gpu_tflops * 1e12) / 1e6;
        double gpu_compute_us = expert_flops / gpu_flops_per_us;
        decision.estimated_gpu_transfer_latency_us = transfer_weight_us + gpu_compute_us;

        // 2. CPU execution time with activation offload (transfer activation over PCIe + compute on CPU OpenMP threads)
        double transfer_activation_us = activation_bytes / pcie_bw_bytes_per_us;
        double cpu_flops_per_us = (profile_.cpu_gflops * 1e9) / 1e6;
        double cpu_compute_us = expert_flops / cpu_flops_per_us;
        decision.estimated_cpu_latency_us = transfer_activation_us + cpu_compute_us;

        // If transfer_weight_us > CPU execution time, offloading activations to CPU is strictly faster!
        if (decision.estimated_gpu_transfer_latency_us > decision.estimated_cpu_latency_us) {
            decision.offload_to_cpu = true;
            decision.target_device = DeviceAffinity::kHybridAsync;
            decision.speedup_ratio = decision.estimated_gpu_transfer_latency_us / std::max(1.0, decision.estimated_cpu_latency_us);
            decision.rationale = "Activation offload to CPU is faster than swapping multi-megabyte weights over PCIe";
        } else {
            decision.offload_to_cpu = false;
            decision.target_device = DeviceAffinity::kGPU;
            decision.speedup_ratio = decision.estimated_cpu_latency_us / std::max(1.0, decision.estimated_gpu_transfer_latency_us);
            decision.rationale = "Resident GPU compute is faster for small layers or saturated CPU";
        }

        return decision;
    }

    /// Computes optimal piggyback batch schedule combining chunked prefill and active decodes
    /// based on Sarathi-Serve arithmetic intensity co-batching.
    PiggybackSchedule compute_piggyback_schedule(
        size_t prompt_tokens_remaining,
        size_t active_decode_count,
        size_t hidden_dim) const {

        PiggybackSchedule sched;
        sched.num_decode_streams = active_decode_count;

        // Determine prefill chunk size (128 - 512 tokens)
        size_t chunk = 0;
        if (prompt_tokens_remaining > 0) {
            if (prompt_tokens_remaining <= 128) {
                chunk = prompt_tokens_remaining;
            } else if (prompt_tokens_remaining <= 256) {
                chunk = 256;
            } else {
                // If there are many active decode streams, throttle chunk size to maintain ITL latency
                chunk = (active_decode_count > 8) ? 128 : 256;
            }
        }
        sched.prefill_chunk_tokens = chunk;
        sched.total_batch_tokens = chunk + active_decode_count;

        // Compute approximate arithmetic intensity: FLOPs / Memory Access Bytes
        // For prefill: ~ 2 * tokens * hidden^2 FLOPs, weights read once -> high AI
        // For decode: ~ 2 * decode_count * hidden^2 FLOPs, weights read once -> low AI
        double weight_memory_bytes = static_cast<double>(hidden_dim * hidden_dim * 2);
        double total_flops = 2.0 * static_cast<double>(sched.total_batch_tokens * hidden_dim * hidden_dim);

        if (weight_memory_bytes > 0.0) {
            sched.estimated_arithmetic_intensity = total_flops / weight_memory_bytes;
        }

        // Saturation achieved when tokens in iteration >= 128
        sched.saturate_compute_cores = (sched.total_batch_tokens >= 128);

        return sched;
    }

    /// Plans asynchronous CPU drafting with parallel GPU verification (Dovetail principle)
    SpeculativePipelinePlan plan_speculative_pipeline(
        double draft_acceptance_rate,
        size_t target_vocab_size) const {

        SpeculativePipelinePlan plan;
        plan.drafter_device = DeviceAffinity::kCPUWorkerPool;
        plan.verifier_device = DeviceAffinity::kGPU;
        plan.overlap_cpu_gpu = true;

        // Optimal K based on acceptance rate alpha:
        // Expected tokens per step: E = (1 - alpha^(K+1)) / (1 - alpha)
        if (draft_acceptance_rate >= 0.80) {
            plan.draft_k = 6;
        } else if (draft_acceptance_rate >= 0.60) {
            plan.draft_k = 4;
        } else {
            plan.draft_k = 2;
        }

        // Theoretical speedup: E / (1 + K * (Cost_draft / Cost_verify))
        double alpha = std::clamp(draft_acceptance_rate, 0.1, 0.95);
        double expected_accepted = (1.0 - std::pow(alpha, plan.draft_k + 1)) / (1.0 - alpha);
        plan.estimated_speedup = expected_accepted / (1.0 + (static_cast<double>(plan.draft_k) * 0.08));

        return plan;
    }

private:
    HardwareProfile profile_;
};

} // namespace strata::runtime
