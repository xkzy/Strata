// include/strata/core/ram_expert_cache.hpp - Pinned / Host RAM Expert Cache Tier
//
// Multi-Tier Heterogeneous Expert Placement:
//   Tier 0: Fast VRAM / dGPU accelerator memory (ExpertCache)
//   Tier 1: Intermediate iGPU / Unified DRAM domain (zero-copy memory domain)
//   Tier 2: Pinned / Host DDR4/DDR5 RAM Expert Cache (RamExpertCache) -> Prevents disk thrashing & speeds up CPU pool & PCIe DMA
//   Tier 3: Memory-mapped disk storage (experts.bin fallback)
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {

inline constexpr int32_t kRamNotResident = -1;

class RamExpertCache {
public:
    RamExpertCache();
    ~RamExpertCache();
    RamExpertCache(const RamExpertCache&) = delete;
    RamExpertCache& operator=(const RamExpertCache&) = delete;

    /// Opens the RAM expert cache with a given number of slots and uniform blob size.
    /// If pinned is true, attempts to page-lock/pin host memory for maximum PCIe DMA transfer speed.
    bool open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
              bool pinned = true, std::string* err = nullptr);

    /// Opens the RAM expert cache with variable per-slot sizes.
    bool open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_layers, int64_t n_expert,
                    bool pinned = true, std::string* err = nullptr);

    void close();

    bool valid() const { return base_ != nullptr; }
    int64_t slots() const { return slots_; }
    int64_t resident() const { return resident_count_; }
    int64_t bytes() const { return total_bytes_; }
    double gib() const { return static_cast<double>(total_bytes_) / 1073741824.0; }
    bool is_pinned() const { return pinned_; }

    /// Look up expert residency in RAM cache. Returns slot index or kRamNotResident.
    int32_t slot_of(int64_t layer, int64_t expert) const;

    /// Get direct host pointer to a cached slot in RAM.
    const uint8_t* host_slot(int32_t slot) const;
    uint8_t* host_slot(int32_t slot);

    /// Get direct host pointer for (layer, expert) or nullptr if not present in RAM cache.
    const uint8_t* get_expert_ptr(int64_t layer, int64_t expert) const;

    /// Admit/store an expert into the RAM cache.
    int32_t admit(int64_t layer, int64_t expert, const uint8_t* src_blob, int64_t bytes = 0);

    /// Evict an expert from the RAM cache.
    bool evict(int64_t layer, int64_t expert);

    /// Preload/populate the RAM cache from an authoritative source.
    int64_t preload_all(int64_t n_layers, int64_t n_expert,
                        const std::function<const uint8_t*(int64_t, int64_t, int64_t&)>& get_blob_fn);

    /// Record access for heat / LRU tracking
    void record_access(int64_t layer, int64_t expert);

    /// Statistics
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    double hit_rate() const {
        uint64_t total = hits_ + misses_;
        return total > 0 ? static_cast<double>(hits_) / total : 0.0;
    }

private:
    uint8_t* base_ = nullptr;
    int64_t slots_ = 0;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    int64_t blob_bytes_ = 0;
    int64_t total_bytes_ = 0;
    int64_t resident_count_ = 0;
    int64_t next_free_slot_ = 0;
    bool pinned_ = false;

    std::vector<int32_t> residency_; // [n_layers * n_expert] -> slot or kRamNotResident
    std::vector<uint64_t> offsets_;
    std::vector<uint64_t> slot_sizes_;
    std::vector<std::pair<int32_t, int32_t>> slot_owners_; // slot -> (layer, expert)
    std::vector<uint64_t> access_counts_;
    std::vector<double> heat_scores_;

    mutable std::mutex mutex_;
    mutable uint64_t hits_ = 0;
    mutable uint64_t misses_ = 0;
};

} // namespace strata::core
