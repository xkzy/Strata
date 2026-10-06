// src/runtime/scheduler.cpp - Dynamic Heterogeneous Computation Scheduler Implementation
#include "strata/runtime/scheduler.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace strata::runtime {

DynamicScheduler::DynamicScheduler(std::shared_ptr<GenericExpertManager> expert_mgr)
    : expert_mgr_(std::move(expert_mgr)) {}

DynamicScheduler::~DynamicScheduler() = default;

ExecutionCost DynamicScheduler::estimate_cost(const ComputeDevice& device, OpType /*op*/,
                                              uint64_t input_bytes, uint64_t output_bytes,
                                              uint64_t required_flops, int64_t layer,
                                              int64_t expert_id) const {
    ExecutionCost cost;

    // 1. Compute time calculation based on device throughput
    double tflops = device.metrics().compute_tflops_fp16;
    if (tflops <= 0.0) tflops = 1.0;
    cost.compute_time_sec = static_cast<double>(required_flops) / (tflops * 1e12);

    // 2. Data transfer cost calculation
    cost.transfer_time_sec = 0.0;
    if (expert_id >= 0 && layer >= 0 && expert_mgr_) {
        const auto* loc = expert_mgr_->get_location(layer, expert_id);
        if (loc) {
            if (loc->device_id != device.id()) {
                // Transfer needed from source domain/device to target device
                int src_dev = loc->device_id >= 0 ? loc->device_id : 0;
                cost.transfer_time_sec += HardwareTopology::instance().transfer_cost_sec(
                    src_dev, device.id(), loc->byte_size);
            }
        }
    }

    if (input_bytes > 0 || output_bytes > 0) {
        cost.transfer_time_sec += (static_cast<double>(input_bytes + output_bytes) /
                                  (device.metrics().memory_bandwidth_gbps * 1e9));
    }

    // 3. Synchronization overhead
    cost.sync_overhead_sec = device.metrics().synchronization_cost_us * 1e-6;

    // 4. Queue delay & utilization penalty
    double queue_delay = device.current_queue_depth() * 5e-6;
    double util_delay = device.current_utilization() * (cost.compute_time_sec * 0.5);
    cost.queue_delay_sec = queue_delay + util_delay;

    return cost;
}

std::vector<ScheduledTask> DynamicScheduler::schedule_active_experts(
    int64_t layer, const int32_t* active_experts, int64_t k,
    uint64_t expert_bytes, uint64_t flops_per_expert) {
    (void)expert_bytes;

    std::vector<ScheduledTask> tasks;
    if (!active_experts || k <= 0) return tasks;

    const auto& devices = DeviceManager::instance().all_devices();
    if (devices.empty()) return tasks;

    tasks.reserve(k);

    // Fast-path for single-device deployment (standard CPU-only or single-GPU)
    if (devices.size() == 1) {
        const auto& dev = devices[0];
        int dev_id = dev->id();
        double tflops = dev->metrics().compute_tflops_fp16;
        if (tflops <= 0.0) tflops = 1.0;
        double compute_sec = static_cast<double>(flops_per_expert) / (tflops * 1e12);

        for (int64_t i = 0; i < k; ++i) {
            int32_t expert_id = active_experts[i];
            if (expert_id < 0) continue;

            ScheduledTask task;
            task.task_id = static_cast<int>(tasks.size());
            task.op = OpType::kGEMV;
            task.target_device_id = dev_id;
            task.estimated_compute_sec = compute_sec;
            task.estimated_transfer_sec = 0.0;
            task.estimated_total_sec = compute_sec;
            task.layer = layer;
            task.expert_id = expert_id;

            tasks.push_back(task);
            tasks_scheduled_per_device_[dev_id]++;
            total_compute_time_per_device_[dev_id] += compute_sec;
        }
        return tasks;
    }

    for (int64_t i = 0; i < k; ++i) {
        int32_t expert_id = active_experts[i];
        if (expert_id < 0) continue;

        int best_device_id = -1;
        double min_total_cost = std::numeric_limits<double>::infinity();
        ExecutionCost best_cost;

        for (const auto& dev : devices) {
            if (!dev->supports_op(OpType::kMoEDispatch) && !dev->supports_op(OpType::kGEMV)) {
                continue;
            }
            ExecutionCost c = estimate_cost(*dev, OpType::kGEMV, 2560 * 2, 2560 * 2,
                                            flops_per_expert, layer, expert_id);
            if (c.total_cost() < min_total_cost) {
                min_total_cost = c.total_cost();
                best_device_id = dev->id();
                best_cost = c;
            }
        }

        if (best_device_id < 0 && !devices.empty()) {
            best_device_id = devices[0]->id();
        }

        ScheduledTask task;
        task.task_id = static_cast<int>(tasks.size());
        task.op = OpType::kGEMV;
        task.target_device_id = best_device_id;
        task.estimated_compute_sec = best_cost.compute_time_sec;
        task.estimated_transfer_sec = best_cost.transfer_time_sec;
        task.estimated_total_sec = best_cost.total_cost();
        task.layer = layer;
        task.expert_id = expert_id;

        tasks.push_back(task);
        tasks_scheduled_per_device_[best_device_id]++;
        total_compute_time_per_device_[best_device_id] += best_cost.compute_time_sec;
    }

    return tasks;
}

ScheduledTask DynamicScheduler::schedule_op(OpType op, uint64_t input_bytes,
                                            uint64_t output_bytes, uint64_t required_flops,
                                            int64_t layer) {
    const auto& devices = DeviceManager::instance().all_devices();
    if (devices.empty()) {
        ScheduledTask task;
        task.task_id = 0;
        task.op = op;
        task.target_device_id = 0;
        task.layer = layer;
        return task;
    }

    // Fast path for single device
    if (devices.size() == 1) {
        const auto& dev = devices[0];
        int dev_id = dev->id();
        ExecutionCost best_cost = estimate_cost(*dev, op, input_bytes, output_bytes, required_flops, layer, -1);
        ScheduledTask task;
        task.task_id = 0;
        task.op = op;
        task.target_device_id = dev_id;
        task.estimated_compute_sec = best_cost.compute_time_sec;
        task.estimated_transfer_sec = best_cost.transfer_time_sec;
        task.estimated_total_sec = best_cost.total_cost();
        task.layer = layer;
        task.expert_id = -1;

        tasks_scheduled_per_device_[dev_id]++;
        total_compute_time_per_device_[dev_id] += best_cost.compute_time_sec;
        return task;
    }

    int best_device_id = 0;
    double min_total_cost = std::numeric_limits<double>::infinity();
    ExecutionCost best_cost;

    for (const auto& dev : devices) {
        if (!dev->supports_op(op)) continue;
        ExecutionCost c = estimate_cost(*dev, op, input_bytes, output_bytes, required_flops, layer, -1);
        if (c.total_cost() < min_total_cost) {
            min_total_cost = c.total_cost();
            best_device_id = dev->id();
            best_cost = c;
        }
    }

    ScheduledTask task;
    task.task_id = 0;
    task.op = op;
    task.target_device_id = best_device_id;
    task.estimated_compute_sec = best_cost.compute_time_sec;
    task.estimated_transfer_sec = best_cost.transfer_time_sec;
    task.estimated_total_sec = best_cost.total_cost();
    task.layer = layer;
    task.expert_id = -1;

    tasks_scheduled_per_device_[best_device_id]++;
    total_compute_time_per_device_[best_device_id] += best_cost.compute_time_sec;

    return task;
}

void DynamicScheduler::update_device_load(int device_id, double current_utilization, int queue_depth) {
    (void)device_id;
    (void)current_utilization;
    (void)queue_depth;
}

std::string DynamicScheduler::print_distribution_summary() const {
    std::ostringstream ss;
    ss << "================ DYNAMIC SCHEDULER TASK DISTRIBUTION ================\n";
    for (const auto& dev : DeviceManager::instance().all_devices()) {
        int dev_id = dev->id();
        uint64_t tasks = 0;
        double compute_sec = 0.0;
        auto it_t = tasks_scheduled_per_device_.find(dev_id);
        if (it_t != tasks_scheduled_per_device_.end()) tasks = it_t->second;
        auto it_c = total_compute_time_per_device_.find(dev_id);
        if (it_c != total_compute_time_per_device_.end()) compute_sec = it_c->second;

        ss << "  [Device " << dev_id << "] " << dev->name()
           << " -> Scheduled Tasks: " << tasks
           << " | Total Scheduled Compute: " << std::fixed << std::setprecision(4)
           << (compute_sec * 1000.0) << " ms\n";
    }
    ss << "======================================================================\n";
    return ss.str();
}

} // namespace strata::runtime
