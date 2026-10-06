// src/runtime/topology.cpp - Hardware Topology Discovery & Calibration Implementation
#include "strata/runtime/topology.hpp"
#include "strata/runtime/device.hpp"
#include "strata/runtime/memory.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

#if defined(STRATA_ENABLE_CUDA) || defined(STRATA_ENABLE_HIP)
#include <cuda_runtime.h>
#endif

namespace strata::runtime {

HardwareTopology& HardwareTopology::instance() {
    static HardwareTopology s_top;
    return s_top;
}

void HardwareTopology::discover() {
    auto& dev_mgr = DeviceManager::instance();
    dev_mgr.discover_all();

    auto& mem_top = MemoryTopology::instance();
    mem_top.clear();

    int next_mem_id = 0;
    int host_mem_id = -1;

    // 1. Create Memory Domains for devices
    for (const auto& dev : dev_mgr.all_devices()) {
        if (dev->type() == DeviceType::kGPU) {
            MemoryCharacteristics chars;
            chars.total_capacity_bytes = dev->total_memory_bytes();
            chars.free_capacity_bytes = dev->free_memory_bytes();
            chars.read_bandwidth_gbps = dev->metrics().memory_bandwidth_gbps;
            chars.write_bandwidth_gbps = dev->metrics().memory_bandwidth_gbps;
            chars.access_latency_ns = 50.0;
            chars.direct_access_by_host = false;
            chars.direct_access_by_device = true;

            std::string dom_name = "Memory Domain " + std::to_string(next_mem_id) + " (" + dev->name() + " VRAM)";
            auto dom = std::make_shared<MemoryDomain>(next_mem_id, dom_name,
                                                      MemoryDomainType::kHighBandwidthDevice, chars);
            dom->add_associated_device(dev->id());
            mem_top.register_domain(dom);
            next_mem_id++;
        } else if (dev->type() == DeviceType::kCPU) {
            if (host_mem_id < 0) {
                MemoryCharacteristics chars;
                chars.total_capacity_bytes = dev->total_memory_bytes();
                chars.free_capacity_bytes = dev->free_memory_bytes();
                chars.read_bandwidth_gbps = dev->metrics().memory_bandwidth_gbps;
                chars.write_bandwidth_gbps = dev->metrics().memory_bandwidth_gbps;
                chars.access_latency_ns = 70.0;
                chars.direct_access_by_host = true;
                chars.direct_access_by_device = false;

                host_mem_id = next_mem_id;
                std::string dom_name = "Memory Domain " + std::to_string(host_mem_id) + " (Host System DRAM)";
                auto dom = std::make_shared<MemoryDomain>(host_mem_id, dom_name,
                                                          MemoryDomainType::kHostSystem, chars);
                dom->add_associated_device(dev->id());
                mem_top.register_domain(dom);
                next_mem_id++;
            } else {
                auto dom = mem_top.get_domain(host_mem_id);
                if (dom) {
                    dom->add_associated_device(dev->id());
                }
            }
        }
    }

    // 2. Discover interconnects & transfer paths
    const auto& domains = mem_top.all_domains();
    for (size_t i = 0; i < domains.size(); ++i) {
        for (size_t j = 0; j < domains.size(); ++j) {
            if (i == j) continue;
            int src_id = domains[i]->id();
            int dst_id = domains[j]->id();

            TransferPath path;
            path.src_domain_id = src_id;
            path.dst_domain_id = dst_id;

            bool is_p2p = false;
#if defined(STRATA_ENABLE_CUDA) || defined(STRATA_ENABLE_HIP)
            if (domains[i]->type() == MemoryDomainType::kHighBandwidthDevice &&
                domains[j]->type() == MemoryDomainType::kHighBandwidthDevice) {
                int dev_i = domains[i]->associated_devices().empty() ? 0 : domains[i]->associated_devices()[0];
                int dev_j = domains[j]->associated_devices().empty() ? 0 : domains[j]->associated_devices()[0];
                int can_access = 0;
                if (cudaDeviceCanAccessPeer(&can_access, dev_i, dev_j) == cudaSuccess && can_access) {
                    is_p2p = true;
                }
            }
#endif

            path.is_direct = is_p2p || (domains[i]->type() == MemoryDomainType::kHostSystem) ||
                                       (domains[j]->type() == MemoryDomainType::kHostSystem);
            path.bandwidth_gbps = is_p2p ? 40.0 : (path.is_direct ? 24.0 : 12.0);
            path.latency_us = is_p2p ? 2.0 : (path.is_direct ? 5.0 : 12.0);

            if (!path.is_direct && host_mem_id >= 0) {
                path.hops = {host_mem_id};
            }

            mem_top.register_path(path);
        }
    }

    discovered_ = true;
}

void HardwareTopology::calibrate(bool quick) {
    (void)quick;
    if (!discovered_) {
        discover();
    }
    MemoryTopology::instance().calibrate_transfers();
    calibrated_ = true;
}

double HardwareTopology::transfer_cost_sec(int src_device_id, int dst_device_id, uint64_t bytes) const {
    if (src_device_id == dst_device_id || bytes == 0) {
        return 0.0;
    }
    // Find associated memory domains
    int src_dom = -1, dst_dom = -1;
    for (const auto& dom : MemoryTopology::instance().all_domains()) {
        if (dom->is_associated_with_device(src_device_id) && src_dom < 0) {
            src_dom = dom->id();
        }
        if (dom->is_associated_with_device(dst_device_id) && dst_dom < 0) {
            dst_dom = dom->id();
        }
    }
    if (src_dom < 0) src_dom = 0;
    if (dst_dom < 0) dst_dom = 0;

    return MemoryTopology::instance().estimate_transfer_time_sec(src_dom, dst_dom, bytes);
}

std::string HardwareTopology::print_summary() const {
    std::ostringstream ss;
    ss << "================== STRATA HARDWARE TOPOLOGY GRAPH ==================\n";
    ss << "Discovered Compute Devices:\n";
    for (const auto& dev : DeviceManager::instance().all_devices()) {
        ss << "  [Device " << dev->id() << "] " << dev->name()
           << " | Type: " << static_cast<int>(dev->type())
           << " | Memory: " << (dev->total_memory_bytes() / (1024 * 1024 * 1024)) << " GB"
           << " | Peak TFLOPS (FP16): " << std::fixed << std::setprecision(1) << dev->metrics().compute_tflops_fp16
           << " | Mem Bandwidth: " << dev->metrics().memory_bandwidth_gbps << " GB/s\n";
    }

    ss << "\nDiscovered Memory Domains:\n";
    for (const auto& dom : MemoryTopology::instance().all_domains()) {
        ss << "  [Memory Domain " << dom->id() << "] " << dom->name()
           << " | Capacity: " << (dom->characteristics().total_capacity_bytes / (1024 * 1024 * 1024)) << " GB"
           << " | Bandwidth: " << dom->characteristics().read_bandwidth_gbps << " GB/s\n";
    }

    ss << "\nTransfer Cost & Interconnect Matrix:\n";
    const auto& domains = MemoryTopology::instance().all_domains();
    for (size_t i = 0; i < domains.size(); ++i) {
        for (size_t j = 0; j < domains.size(); ++j) {
            if (i == j) continue;
            const auto* path = MemoryTopology::instance().find_path(domains[i]->id(), domains[j]->id());
            if (path) {
                ss << "  Domain " << domains[i]->id() << " -> Domain " << domains[j]->id()
                   << ": " << (path->is_direct ? "DIRECT" : "STAGED")
                   << " | Bandwidth: " << path->bandwidth_gbps << " GB/s"
                   << " | Latency: " << path->latency_us << " us\n";
            }
        }
    }
    ss << "====================================================================\n";
    return ss.str();
}

} // namespace strata::runtime
