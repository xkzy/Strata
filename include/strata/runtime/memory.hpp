// include/strata/runtime/memory.hpp - Generic Memory Domain and Topology Abstraction
//
// Models memory as generic domains (capacity, bandwidth, latency, supported devices,
// transfer paths) without hardcoding "GPU=VRAM, CPU=RAM".
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::runtime {

enum class MemoryDomainType {
    kHighBandwidthDevice = 0, // e.g. HBM, GDDR VRAM
    kHostSystem,              // e.g. System DDR/LPDDR RAM
    kUnifiedHostDevice,       // e.g. Unified/Zero-Copy memory
    kStorageBacked,           // e.g. NVMe SSD / Mapped File
    kRemoteNetwork            // e.g. RDMA / Peer Node
};

struct MemoryCharacteristics {
    uint64_t total_capacity_bytes = 0;
    uint64_t free_capacity_bytes = 0;
    double read_bandwidth_gbps = 0.0;
    double write_bandwidth_gbps = 0.0;
    double access_latency_ns = 0.0;
    bool direct_access_by_host = false;
    bool direct_access_by_device = true;
    bool is_pageable = false;
};

class MemoryDomain {
public:
    MemoryDomain(int id, const std::string& name, MemoryDomainType type,
                 const MemoryCharacteristics& chars);

    int id() const { return id_; }
    const std::string& name() const { return name_; }
    MemoryDomainType type() const { return type_; }
    const MemoryCharacteristics& characteristics() const { return chars_; }
    MemoryCharacteristics& mutable_characteristics() { return chars_; }

    void add_associated_device(int device_id);
    const std::vector<int>& associated_devices() const { return associated_devices_; }
    bool is_associated_with_device(int device_id) const;

private:
    int id_;
    std::string name_;
    MemoryDomainType type_;
    MemoryCharacteristics chars_;
    std::vector<int> associated_devices_;
};

struct TransferPath {
    int src_domain_id;
    int dst_domain_id;
    bool is_direct;                  // True if direct DMA / P2P is possible
    double bandwidth_gbps;           // Measured or estimated transfer bandwidth
    double latency_us;               // Transfer initiation & overhead latency
    std::vector<int> hops;           // Intermediate memory domains if staged
};

class MemoryTopology {
public:
    static MemoryTopology& instance();

    void register_domain(std::shared_ptr<MemoryDomain> domain);
    void register_path(const TransferPath& path);
    void clear();

    std::shared_ptr<MemoryDomain> get_domain(int id) const;
    const std::vector<std::shared_ptr<MemoryDomain>>& all_domains() const;

    // Find the lowest cost transfer path between two memory domains
    const TransferPath* find_path(int src_domain_id, int dst_domain_id) const;

    // Estimate transfer cost (time in seconds) for a given byte size
    double estimate_transfer_time_sec(int src_domain_id, int dst_domain_id, uint64_t bytes) const;

    // Measure / calibrate paths at runtime
    void calibrate_transfers();

private:
    MemoryTopology() = default;
    std::vector<std::shared_ptr<MemoryDomain>> domains_;
    std::unordered_map<uint64_t, TransferPath> paths_; // key: (src_id << 32) | dst_id

    static uint64_t make_key(int src, int dst) {
        return (static_cast<uint64_t>(src) << 32) | static_cast<uint32_t>(dst);
    }
};

} // namespace strata::runtime
