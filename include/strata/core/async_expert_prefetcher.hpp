// include/strata/core/async_expert_prefetcher.hpp - Double-Buffered Asynchronous Expert Prefetcher
//
// Overlaps PCIe DMA transfers of RAM-cached MoE experts with GPU compute:
// - Double-buffered staging slots (Buffer A compute, Buffer B transfer).
// - Dedicated asynchronous transfer stream (cudaStream / hipStream) with hardware event sync.
// - Direct DMA integration with RamExpertCache (Tier 2) and ExpertCache (Tier 0).
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/ram_expert_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <queue>
#include <chrono>

namespace strata::core {

struct PrefetchItem {
    int64_t layer = -1;
    int64_t expert = -1;
    const uint8_t* host_ptr = nullptr;
    int64_t bytes = 0;
    int32_t target_slot = -1;
};

struct PrefetchStats {
    uint64_t prefetches_issued = 0;
    uint64_t prefetches_completed = 0;
    uint64_t prefetches_hit = 0;
    uint64_t sync_wait_count = 0;
    double total_sync_wait_ms = 0.0;
    double total_bytes_transferred = 0.0;
};

class AsyncExpertPrefetcher {
public:
    AsyncExpertPrefetcher();
    ~AsyncExpertPrefetcher();
    AsyncExpertPrefetcher(const AsyncExpertPrefetcher&) = delete;
    AsyncExpertPrefetcher& operator=(const AsyncExpertPrefetcher&) = delete;

    /// Initialize the prefetcher with device stream and double-buffer capacity.
    /// buffer_capacity: max number of concurrent expert slots per buffer bank (e.g. 16)
    /// slot_bytes: max size in bytes of a single expert blob
    bool initialize(void* device_stream, int64_t buffer_capacity, int64_t slot_bytes, std::string* err = nullptr);

    /// Shut down and synchronize all background transfer streams.
    void shutdown();

    /// Issue an asynchronous prefetch request for an expert from RAM cache to the staging buffer.
    /// Returns true if queued or already in-flight.
    bool prefetch_expert(int64_t layer, int64_t expert, const RamExpertCache& ram_cache);

    /// Issue batch prefetch requests for the next layer's predicted routed experts.
    int64_t prefetch_layer_experts(int64_t layer, const std::vector<int32_t>& routed_experts,
                                  const RamExpertCache& ram_cache);

    /// Synchronize and acquire the prefetched expert pointer on the active compute stream.
    /// Returns device pointer to the staged expert weights, or nullptr if not available.
    const uint8_t* acquire_expert(int64_t layer, int64_t expert, void* compute_stream);

    /// Advance the double buffer bank for the next compute cycle (swaps active and prefetch banks).
    void advance_buffer_bank();

    /// Reset internal state between requests.
    void reset();

    /// Performance and telemetry metrics
    const PrefetchStats& stats() const { return stats_; }

private:
    void* transfer_stream_ = nullptr;
    void* completion_event_ = nullptr;
    bool own_stream_ = false;
    bool initialized_ = false;

    int64_t buffer_capacity_ = 0;
    int64_t slot_bytes_ = 0;
    uint8_t* d_staging_bank_a_ = nullptr;
    uint8_t* d_staging_bank_b_ = nullptr;

    // Active compute bank: 0 for A, 1 for B
    int active_bank_idx_ = 0;

    // Track staged experts in each bank: key -> (slot_idx, bytes)
    struct StagedEntry {
        int64_t layer;
        int64_t expert;
        int32_t slot_idx;
        int64_t bytes;
        bool ready;
    };
    std::vector<StagedEntry> bank_a_entries_;
    std::vector<StagedEntry> bank_b_entries_;

    std::mutex mutex_;
    PrefetchStats stats_;
};

} // namespace strata::core
