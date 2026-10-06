// src/runtime/device.cpp - Generic Compute Device Implementation
#include "strata/runtime/device.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>

#if defined(STRATA_ENABLE_CUDA) || defined(STRATA_ENABLE_HIP)
#include <cuda_runtime.h>
#endif

namespace strata::runtime {

namespace {

// Generic Host CPU Compute Device implementation
class HostCpuDevice : public ComputeDevice {
public:
    explicit HostCpuDevice(int id) : id_(id) {
        metrics_.num_compute_units = std::max(1u, std::thread::hardware_concurrency());
        metrics_.clock_mhz = 3000;
        // Estimate baseline CPU compute throughput (~15 GFLOPS per core FP32 AVX2/AVX-512)
        metrics_.compute_tflops_fp32 = 0.015 * metrics_.num_compute_units;
        metrics_.compute_tflops_fp16 = 0.030 * metrics_.num_compute_units;
        metrics_.compute_tflops_bf16 = 0.030 * metrics_.num_compute_units;
        metrics_.compute_tflops_int8 = 0.060 * metrics_.num_compute_units;
        metrics_.compute_tflops_int4 = 0.120 * metrics_.num_compute_units;
        metrics_.memory_bandwidth_gbps = 50.0; // Typical DDR4/DDR5 system bandwidth
        metrics_.transfer_latency_us = 0.1;
        metrics_.synchronization_cost_us = 0.2;
    }

    int id() const override { return id_; }
    DeviceType type() const override { return DeviceType::kCPU; }
    std::string name() const override { return "Host CPU (Multi-core)"; }
    std::string vendor_metadata() const override { return "Generic Host Processor"; }

    uint64_t total_memory_bytes() const override {
        return 64ULL * 1024 * 1024 * 1024; // 64 GB nominal or probe
    }
    uint64_t free_memory_bytes() const override {
        return 32ULL * 1024 * 1024 * 1024;
    }

    const DeviceMetrics& metrics() const override { return metrics_; }
    double current_utilization() const override { return utilization_; }
    int current_queue_depth() const override { return queue_depth_; }

    bool supports_op(OpType) const override {
        return true; // CPU fallback supports all operations
    }

    void* allocate(uint64_t bytes, uint64_t align = 256) override {
        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(bytes, align);
#else
        if (posix_memalign(&ptr, align, bytes) != 0) {
            ptr = nullptr;
        }
#endif
        return ptr;
    }

    void deallocate(void* ptr) override {
        if (!ptr) return;
#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    bool copy_to_device(void* dst, const void* src, uint64_t bytes, void* /*stream*/ = nullptr) override {
        if (!dst || !src) return false;
        std::memcpy(dst, src, bytes);
        return true;
    }

    bool copy_to_host(void* dst, const void* src, uint64_t bytes, void* /*stream*/ = nullptr) override {
        if (!dst || !src) return false;
        std::memcpy(dst, src, bytes);
        return true;
    }

    bool copy_p2p(void* dst, const void* src, uint64_t bytes, ComputeDevice* /*src_device*/, void* /*stream*/ = nullptr) override {
        if (!dst || !src) return false;
        std::memcpy(dst, src, bytes);
        return true;
    }

    void synchronize() override {}
    void* create_stream() override { return reinterpret_cast<void*>(1); }
    void destroy_stream(void*) override {}
    void synchronize_stream(void*) override {}

private:
    int id_;
    DeviceMetrics metrics_;
    double utilization_ = 0.0;
    int queue_depth_ = 0;
};

#if defined(STRATA_ENABLE_CUDA) || defined(STRATA_ENABLE_HIP)
// Generic GPU Compute Device implementation wrapping CUDA/HIP
class GpuDevice : public ComputeDevice {
public:
    GpuDevice(int id, int cuda_ordinal) : id_(id), cuda_ordinal_(cuda_ordinal) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, cuda_ordinal_) == cudaSuccess) {
            name_ = prop.name;
            vendor_info_ = "Compute Capability " + std::to_string(prop.major) + "." + std::to_string(prop.minor);
            total_mem_ = prop.totalGlobalMem;
            metrics_.num_compute_units = prop.multiProcessorCount;
            metrics_.clock_mhz = prop.clockRate / 1000;
            // Estimate TFLOPS based on SM count & modern tensor core throughput
            metrics_.compute_tflops_fp32 = (prop.multiProcessorCount * 128.0 * metrics_.clock_mhz * 2.0) / 1e6;
            metrics_.compute_tflops_fp16 = metrics_.compute_tflops_fp32 * 2.0;
            metrics_.compute_tflops_bf16 = metrics_.compute_tflops_fp16;
            metrics_.compute_tflops_int8 = metrics_.compute_tflops_fp16 * 2.0;
            metrics_.compute_tflops_int4 = metrics_.compute_tflops_int8 * 2.0;
            metrics_.memory_bandwidth_gbps = (prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) * 2.0) / 1e9;
            metrics_.transfer_latency_us = 5.0;
            metrics_.synchronization_cost_us = 2.0;
        } else {
            name_ = "Generic GPU " + std::to_string(id);
            vendor_info_ = "GPU Device";
            total_mem_ = 8ULL * 1024 * 1024 * 1024;
            metrics_.compute_tflops_fp32 = 10.0;
            metrics_.compute_tflops_fp16 = 20.0;
            metrics_.compute_tflops_bf16 = 20.0;
            metrics_.memory_bandwidth_gbps = 300.0;
        }
    }

    int id() const override { return id_; }
    DeviceType type() const override { return DeviceType::kGPU; }
    std::string name() const override { return name_; }
    std::string vendor_metadata() const override { return vendor_info_; }

    uint64_t total_memory_bytes() const override { return total_mem_; }
    uint64_t free_memory_bytes() const override {
        size_t free_b = 0, total_b = 0;
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(cuda_ordinal_);
        cudaMemGetInfo(&free_b, &total_b);
        cudaSetDevice(prev);
        return free_b;
    }

    const DeviceMetrics& metrics() const override { return metrics_; }
    double current_utilization() const override { return utilization_; }
    int current_queue_depth() const override { return queue_depth_; }

    bool supports_op(OpType) const override { return true; }

    void* allocate(uint64_t bytes, uint64_t /*align*/ = 256) override {
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(cuda_ordinal_);
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        cudaSetDevice(prev);
        return err == cudaSuccess ? ptr : nullptr;
    }

    void deallocate(void* ptr) override {
        if (!ptr) return;
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(cuda_ordinal_);
        cudaFree(ptr);
        cudaSetDevice(prev);
    }

    bool copy_to_device(void* dst, const void* src, uint64_t bytes, void* stream = nullptr) override {
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(cuda_ordinal_);
        cudaError_t err;
        if (stream) {
            err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream));
        } else {
            err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
        }
        cudaSetDevice(prev);
        return err == cudaSuccess;
    }

    bool copy_to_host(void* dst, const void* src, uint64_t bytes, void* stream = nullptr) override {
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(cuda_ordinal_);
        cudaError_t err;
        if (stream) {
            err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream));
        } else {
            err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
        }
        cudaSetDevice(prev);
        return err == cudaSuccess;
    }

    bool copy_p2p(void* dst, const void* src, uint64_t bytes, ComputeDevice* /*src_device*/, void* stream = nullptr) override {
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(cuda_ordinal_);
        cudaError_t err;
        if (stream) {
            err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream));
        } else {
            err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
        }
        cudaSetDevice(prev);
        return err == cudaSuccess;
    }

    void synchronize() override {
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(cuda_ordinal_);
        cudaDeviceSynchronize();
        cudaSetDevice(prev);
    }

    void* create_stream() override {
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(cuda_ordinal_);
        cudaStream_t stream = nullptr;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cudaSetDevice(prev);
        return static_cast<void*>(stream);
    }

    void destroy_stream(void* stream) override {
        if (!stream) return;
        cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    }

    void synchronize_stream(void* stream) override {
        if (!stream) return;
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    }

private:
    int id_;
    int cuda_ordinal_;
    std::string name_;
    std::string vendor_info_;
    uint64_t total_mem_ = 0;
    DeviceMetrics metrics_;
    double utilization_ = 0.0;
    int queue_depth_ = 0;
};
#endif

} // anonymous namespace

DeviceManager& DeviceManager::instance() {
    static DeviceManager s_mgr;
    return s_mgr;
}

void DeviceManager::register_device(std::shared_ptr<ComputeDevice> device) {
    if (device) {
        devices_.push_back(std::move(device));
    }
}

void DeviceManager::clear_devices() {
    devices_.clear();
}

int DeviceManager::device_count() const {
    return static_cast<int>(devices_.size());
}

std::shared_ptr<ComputeDevice> DeviceManager::get_device(int id) const {
    if (id >= 0 && id < static_cast<int>(devices_.size())) {
        return devices_[id];
    }
    return nullptr;
}

const std::vector<std::shared_ptr<ComputeDevice>>& DeviceManager::all_devices() const {
    return devices_;
}

void DeviceManager::discover_all() {
    clear_devices();
    int next_id = 0;

#if defined(STRATA_ENABLE_CUDA) || defined(STRATA_ENABLE_HIP)
    int gpu_count = 0;
    if (cudaGetDeviceCount(&gpu_count) == cudaSuccess && gpu_count > 0) {
        for (int i = 0; i < gpu_count; ++i) {
            register_device(std::make_shared<GpuDevice>(next_id++, i));
        }
    }
#endif

    // Always register Host CPU compute device
    register_device(std::make_shared<HostCpuDevice>(next_id++));
}

} // namespace strata::runtime
