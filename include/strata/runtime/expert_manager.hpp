// include/strata/runtime/expert_manager.hpp - Generic Expert Management & Dynamic Placement
//
// Manages expert residency across arbitrary memory domains and devices.
// Calculates placement costs from topology measurements rather than hardcoded assumptions.
#pragma once

#include "strata/runtime/device.hpp"
#include "strata/runtime/memory.hpp"
#include "strata/runtime/topology.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::runtime {

struct ExpertIdentity {
    int64_t layer;
    int64_t expert_id;

    bool operator==(const ExpertIdentity& o) const {
        return layer == o.layer && expert_id == o.expert_id;
    }
};

struct ExpertIdentityHash {
    std::size_t operator()(const ExpertIdentity& k) const {
        return (static_cast<std::size_t>(k.layer) << 32) ^ static_cast<std::size_t>(k.expert_id);
    }
};

struct ExpertStats {
    uint64_t access_count = 0;
    double last_access_sec = 0.0;
    double heat_score = 0.0; // Exponentially decayed access frequency
};

enum class ExpertResidencyTier {
    kDeviceFast = 0,    // High-speed accelerator memory (e.g. Device 0, Device 1 VRAM)
    kIntermediateTier,  // Intermediate/P2P peer or unified memory
    kHostCapacityTier,  // System DRAM
    kStorageTier        // Direct NVMe / SSD
};

struct ExpertLocation {
    int memory_domain_id = -1;
    int device_id = -1;
    int32_t slot_index = -1;
    const void* host_address = nullptr;
    void* device_address = nullptr;
    uint64_t byte_size = 0;
    std::string quant_type = "IQ3_XXS";
};

class GenericExpertManager {
public:
    GenericExpertManager(int64_t n_layers, int64_t n_experts);
    ~GenericExpertManager();

    // Register expert blob information
    void register_expert(int64_t layer, int64_t expert_id, const void* host_blob,
                         uint64_t bytes, const std::string& quant_type);

    // Record token routing access to update heat statistics
    void record_access(int64_t layer, int64_t expert_id, double current_time_sec);

    // Dynamic placement evaluation
    // Computes the optimal memory domain / device location for each expert based on:
    // heat score, domain bandwidths, capacities, and transfer costs.
    void rebalance_placement();

    // Query expert location
    const ExpertLocation* get_location(int64_t layer, int64_t expert_id) const;
    ExpertStats get_stats(int64_t layer, int64_t expert_id) const;

    // Direct residency lookup for fast kernel execution (returns device pointer or nullptr)
    void* get_device_ptr(int64_t layer, int64_t expert_id, int device_id) const;

    // Migrate an expert from its current domain to target memory domain
    bool migrate_expert(int64_t layer, int64_t expert_id, int target_domain_id,
                        ComputeDevice* dev = nullptr, void* stream = nullptr);

    // Memory stats across domains
    uint64_t resident_bytes_in_domain(int domain_id) const;
    int64_t resident_experts_in_domain(int domain_id) const;

private:
    int64_t n_layers_;
    int64_t n_experts_;

    mutable std::mutex mutex_;
    std::unordered_map<ExpertIdentity, ExpertLocation, ExpertIdentityHash> locations_;
    std::unordered_map<ExpertIdentity, ExpertStats, ExpertIdentityHash> stats_;

    std::unordered_map<int, uint64_t> domain_usage_bytes_;
};

} // namespace strata::runtime
