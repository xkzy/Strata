// include/strata/runtime/topology.hpp - Hardware Topology Discovery & Calibration
//
// Automatically discovers all compute devices, memory domains, and their interconnects.
// Builds a unified topology graph and profiles actual bandwidths / latencies.
#pragma once

#include "strata/runtime/device.hpp"
#include "strata/runtime/memory.hpp"

#include <memory>
#include <string>
#include <vector>

namespace strata::runtime {

struct TopologyEdge {
    int src_node;
    int dst_node;
    double bandwidth_gbps;
    double latency_us;
    bool is_p2p;
};

class HardwareTopology {
public:
    static HardwareTopology& instance();

    // Full discovery: scans all backends, discovers devices & memory domains, and establishes topology
    void discover();

    // Hardware-agnostic benchmarking and calibration
    void calibrate(bool quick = false);

    // Prints human-readable topology graph report
    std::string print_summary() const;

    // Direct access to devices & domains
    const std::vector<std::shared_ptr<ComputeDevice>>& devices() const {
        return DeviceManager::instance().all_devices();
    }
    const std::vector<std::shared_ptr<MemoryDomain>>& memory_domains() const {
        return MemoryTopology::instance().all_domains();
    }

    // Cost estimation for device-to-device or memory-to-device transfers
    double transfer_cost_sec(int src_device_id, int dst_device_id, uint64_t bytes) const;

private:
    HardwareTopology() = default;
    bool discovered_ = false;
    bool calibrated_ = false;
};

} // namespace strata::runtime
