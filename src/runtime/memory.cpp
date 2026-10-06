// src/runtime/memory.cpp - Generic Memory Domain and Topology Implementation
#include "strata/runtime/memory.hpp"
#include "strata/runtime/device.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace strata::runtime {

MemoryDomain::MemoryDomain(int id, const std::string& name, MemoryDomainType type,
                           const MemoryCharacteristics& chars)
    : id_(id), name_(name), type_(type), chars_(chars) {}

void MemoryDomain::add_associated_device(int device_id) {
    if (std::find(associated_devices_.begin(), associated_devices_.end(), device_id) == associated_devices_.end()) {
        associated_devices_.push_back(device_id);
    }
}

bool MemoryDomain::is_associated_with_device(int device_id) const {
    return std::find(associated_devices_.begin(), associated_devices_.end(), device_id) != associated_devices_.end();
}

MemoryTopology& MemoryTopology::instance() {
    static MemoryTopology s_top;
    return s_top;
}

void MemoryTopology::register_domain(std::shared_ptr<MemoryDomain> domain) {
    if (domain) {
        domains_.push_back(std::move(domain));
    }
}

void MemoryTopology::register_path(const TransferPath& path) {
    uint64_t key = make_key(path.src_domain_id, path.dst_domain_id);
    paths_[key] = path;
}

void MemoryTopology::clear() {
    domains_.clear();
    paths_.clear();
}

std::shared_ptr<MemoryDomain> MemoryTopology::get_domain(int id) const {
    if (id >= 0 && id < static_cast<int>(domains_.size())) {
        return domains_[id];
    }
    return nullptr;
}

const std::vector<std::shared_ptr<MemoryDomain>>& MemoryTopology::all_domains() const {
    return domains_;
}

const TransferPath* MemoryTopology::find_path(int src_domain_id, int dst_domain_id) const {
    uint64_t key = make_key(src_domain_id, dst_domain_id);
    auto it = paths_.find(key);
    if (it != paths_.end()) {
        return &it->second;
    }
    return nullptr;
}

double MemoryTopology::estimate_transfer_time_sec(int src_domain_id, int dst_domain_id, uint64_t bytes) const {
    if (src_domain_id == dst_domain_id || bytes == 0) {
        return 0.0;
    }
    const TransferPath* path = find_path(src_domain_id, dst_domain_id);
    if (path) {
        double bw_bytes_sec = path->bandwidth_gbps * 1e9;
        if (bw_bytes_sec <= 0.0) bw_bytes_sec = 1e9;
        double transfer_time = static_cast<double>(bytes) / bw_bytes_sec;
        double latency_time = path->latency_us * 1e-6;
        return transfer_time + latency_time;
    }
    // Fallback: estimate based on domain read/write bandwidths
    auto src_dom = get_domain(src_domain_id);
    auto dst_dom = get_domain(dst_domain_id);
    double effective_bw = 10.0; // 10 GB/s fallback
    if (src_dom && dst_dom) {
        effective_bw = std::min(src_dom->characteristics().read_bandwidth_gbps,
                                dst_dom->characteristics().write_bandwidth_gbps);
        if (effective_bw <= 0.0) effective_bw = 10.0;
    }
    return (static_cast<double>(bytes) / (effective_bw * 1e9)) + 10e-6;
}

void MemoryTopology::calibrate_transfers() {
    // Calibrate all registered paths
    for (auto& kv : paths_) {
        TransferPath& path = kv.second;
        if (path.bandwidth_gbps <= 0.0) {
            path.bandwidth_gbps = path.is_direct ? 24.0 : 12.0; // Default PCIe Gen4/Gen5
        }
        if (path.latency_us <= 0.0) {
            path.latency_us = path.is_direct ? 3.0 : 10.0;
        }
    }
}

} // namespace strata::runtime
