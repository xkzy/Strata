// include/strata/runtime/device.hpp - Generic Compute Device Abstraction
//
// A hardware-agnostic device interface for heterogeneous execution.
// Hardware-specific APIs (CUDA, HIP, Vulkan, SYCL, OpenCL, CPU, etc.) are implemented
// strictly behind this interface without exposing vendor-specific logic to the scheduler.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::runtime {

enum class DeviceType {
    kGeneric = 0,
    kGPU,
    kCPU,
    kNPU,
    kAPU,
    kDSP,
    kAccelerator,
    kRemote
};

enum class OpType {
    kGEMV = 0,
    kGEMM,
    kAttention,
    kRecurrence,
    kRouter,
    kNorm,
    kElementwise,
    kQuantize,
    kDequantize,
    kMoEDispatch,
    kCustom
};

struct DeviceMetrics {
    double compute_tflops_fp32 = 0.0;
    double compute_tflops_fp16 = 0.0;
    double compute_tflops_bf16 = 0.0;
    double compute_tflops_int8 = 0.0;
    double compute_tflops_int4 = 0.0;
    double memory_bandwidth_gbps = 0.0;
    double transfer_latency_us = 0.0;
    double synchronization_cost_us = 0.0;
    int num_compute_units = 0;
    int clock_mhz = 0;
};

class ComputeDevice {
public:
    virtual ~ComputeDevice() = default;

    // Identification (vendor-neutral)
    virtual int id() const = 0;
    virtual DeviceType type() const = 0;
    virtual std::string name() const = 0;
    virtual std::string vendor_metadata() const = 0; // Informational only - not used by scheduler

    // Memory metrics
    virtual uint64_t total_memory_bytes() const = 0;
    virtual uint64_t free_memory_bytes() const = 0;

    // Compute & transfer metrics
    virtual const DeviceMetrics& metrics() const = 0;
    virtual double current_utilization() const = 0; // 0.0 to 1.0
    virtual int current_queue_depth() const = 0;

    // Operations supported
    virtual bool supports_op(OpType op) const = 0;

    // Memory operations
    virtual void* allocate(uint64_t bytes, uint64_t align = 256) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual bool copy_to_device(void* dst, const void* src, uint64_t bytes, void* stream = nullptr) = 0;
    virtual bool copy_to_host(void* dst, const void* src, uint64_t bytes, void* stream = nullptr) = 0;
    virtual bool copy_p2p(void* dst, const void* src, uint64_t bytes, ComputeDevice* src_device, void* stream = nullptr) = 0;

    // Execution & Synchronization
    virtual void synchronize() = 0;
    virtual void* create_stream() = 0;
    virtual void destroy_stream(void* stream) = 0;
    virtual void synchronize_stream(void* stream) = 0;
};

// Device registry & discovery
class DeviceManager {
public:
    static DeviceManager& instance();

    void register_device(std::shared_ptr<ComputeDevice> device);
    void clear_devices();

    int device_count() const;
    std::shared_ptr<ComputeDevice> get_device(int id) const;
    const std::vector<std::shared_ptr<ComputeDevice>>& all_devices() const;

    // Discover all available hardware devices across registered backends
    void discover_all();

private:
    DeviceManager() = default;
    std::vector<std::shared_ptr<ComputeDevice>> devices_;
};

} // namespace strata::runtime
