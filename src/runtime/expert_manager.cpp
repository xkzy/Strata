// src/runtime/expert_manager.cpp - Generic Expert Management Implementation
#include "strata/runtime/expert_manager.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace strata::runtime {

GenericExpertManager::GenericExpertManager(int64_t n_layers, int64_t n_experts)
    : n_layers_(n_layers), n_experts_(n_experts) {}

GenericExpertManager::~GenericExpertManager() {
    // Release any allocated device memory
    for (auto& kv : locations_) {
        ExpertLocation& loc = kv.second;
        if (loc.device_address && loc.device_id >= 0) {
            auto dev = DeviceManager::instance().get_device(loc.device_id);
            if (dev) {
                dev->deallocate(loc.device_address);
                loc.device_address = nullptr;
            }
        }
    }
}

void GenericExpertManager::register_expert(int64_t layer, int64_t expert_id,
                                           const void* host_blob, uint64_t bytes,
                                           const std::string& quant_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    ExpertIdentity id{layer, expert_id};
    ExpertLocation loc;
    loc.host_address = host_blob;
    loc.byte_size = bytes;
    loc.quant_type = quant_type;
    loc.memory_domain_id = 0; // Starts in host/primary domain
    locations_[id] = loc;

    ExpertStats stats;
    stats.access_count = 0;
    stats.last_access_sec = 0.0;
    stats.heat_score = 0.0;
    stats_[id] = stats;

    domain_usage_bytes_[loc.memory_domain_id] += bytes;
}

void GenericExpertManager::record_access(int64_t layer, int64_t expert_id, double current_time_sec) {
    std::lock_guard<std::mutex> lock(mutex_);
    ExpertIdentity id{layer, expert_id};
    auto it = stats_.find(id);
    if (it != stats_.end()) {
        it->second.access_count++;
        double dt = current_time_sec - it->second.last_access_sec;
        if (dt > 0.0) {
            double decay = std::exp(-dt / 10.0); // 10-second half-life decay
            it->second.heat_score = it->second.heat_score * decay + 1.0;
        } else {
            it->second.heat_score += 1.0;
        }
        it->second.last_access_sec = current_time_sec;
    }
}

void GenericExpertManager::rebalance_placement() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Rank all experts by heat score
    std::vector<std::pair<ExpertIdentity, double>> ranked;
    ranked.reserve(stats_.size());
    for (const auto& kv : stats_) {
        ranked.push_back({kv.first, kv.second.heat_score});
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Query topology for available high-bandwidth memory domains
    auto domains = MemoryTopology::instance().all_domains();
    if (domains.empty()) return;

    // Place hottest experts in highest-bandwidth memory domains within capacity limits
    std::vector<std::shared_ptr<MemoryDomain>> sorted_domains = domains;
    std::sort(sorted_domains.begin(), sorted_domains.end(),
              [](const auto& a, const auto& b) {
                  return a->characteristics().read_bandwidth_gbps > b->characteristics().read_bandwidth_gbps;
              });

    size_t exp_idx = 0;
    for (const auto& dom : sorted_domains) {
        uint64_t domain_budget = (dom->characteristics().total_capacity_bytes * 7) / 10; // 70% for expert cache
        uint64_t allocated = 0;

        while (exp_idx < ranked.size()) {
            const auto& id = ranked[exp_idx].first;
            auto& loc = locations_[id];
            if (allocated + loc.byte_size > domain_budget) {
                break; // Domain full, proceed to next tier
            }
            if (loc.memory_domain_id != dom->id()) {
                // Mark for target domain
                loc.memory_domain_id = dom->id();
                if (!dom->associated_devices().empty()) {
                    loc.device_id = dom->associated_devices()[0];
                }
            }
            allocated += loc.byte_size;
            exp_idx++;
        }
    }
}

const ExpertLocation* GenericExpertManager::get_location(int64_t layer, int64_t expert_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    ExpertIdentity id{layer, expert_id};
    auto it = locations_.find(id);
    if (it != locations_.end()) {
        return &it->second;
    }
    return nullptr;
}

ExpertStats GenericExpertManager::get_stats(int64_t layer, int64_t expert_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    ExpertIdentity id{layer, expert_id};
    auto it = stats_.find(id);
    if (it != stats_.end()) {
        return it->second;
    }
    return ExpertStats{};
}

void* GenericExpertManager::get_device_ptr(int64_t layer, int64_t expert_id, int device_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    ExpertIdentity id{layer, expert_id};
    auto it = locations_.find(id);
    if (it != locations_.end()) {
        if (it->second.device_id == device_id && it->second.device_address) {
            return it->second.device_address;
        }
    }
    return nullptr;
}

bool GenericExpertManager::migrate_expert(int64_t layer, int64_t expert_id, int target_domain_id,
                                          ComputeDevice* dev, void* stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    ExpertIdentity id{layer, expert_id};
    auto it = locations_.find(id);
    if (it == locations_.end()) return false;

    ExpertLocation& loc = it->second;
    if (loc.memory_domain_id == target_domain_id) return true;

    if (!dev) {
        auto dom = MemoryTopology::instance().get_domain(target_domain_id);
        if (dom && !dom->associated_devices().empty()) {
            dev = DeviceManager::instance().get_device(dom->associated_devices()[0]).get();
        }
    }

    if (dev && loc.host_address) {
        if (!loc.device_address) {
            loc.device_address = dev->allocate(loc.byte_size);
        }
        if (loc.device_address) {
            dev->copy_to_device(loc.device_address, loc.host_address, loc.byte_size, stream);
            loc.memory_domain_id = target_domain_id;
            loc.device_id = dev->id();
            return true;
        }
    }
    return false;
}

uint64_t GenericExpertManager::resident_bytes_in_domain(int domain_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t total = 0;
    for (const auto& kv : locations_) {
        if (kv.second.memory_domain_id == domain_id) {
            total += kv.second.byte_size;
        }
    }
    return total;
}

int64_t GenericExpertManager::resident_experts_in_domain(int domain_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    int64_t count = 0;
    for (const auto& kv : locations_) {
        if (kv.second.memory_domain_id == domain_id) {
            count++;
        }
    }
    return count;
}

} // namespace strata::runtime
