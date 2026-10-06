// include/strata/runtime/scheduler.hpp - Dynamic Heterogeneous Computation Scheduler
//
// Schedules computation across arbitrary heterogeneous devices based on real-time cost estimation:
// cost = compute_time + transfer_time + sync_time + queue_delay
// No vendor or hardware-specific assumptions.
#pragma once

#include "strata/runtime/device.hpp"
#include "strata/runtime/expert_manager.hpp"
#include "strata/runtime/memory.hpp"
#include "strata/runtime/topology.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::runtime {

struct ScheduledTask {
    int task_id;
    OpType op;
    int target_device_id;
    double estimated_compute_sec;
    double estimated_transfer_sec;
    double estimated_total_sec;
    int64_t layer;
    int64_t expert_id;
};

struct ExecutionCost {
    double compute_time_sec = 0.0;
    double transfer_time_sec = 0.0;
    double sync_overhead_sec = 0.0;
    double queue_delay_sec = 0.0;

    double total_cost() const {
        return compute_time_sec + transfer_time_sec + sync_overhead_sec + queue_delay_sec;
    }
};

class DynamicScheduler {
public:
    DynamicScheduler(std::shared_ptr<GenericExpertManager> expert_mgr);
    ~DynamicScheduler();

    // Cost estimation for executing an operation on a candidate device
    ExecutionCost estimate_cost(const ComputeDevice& device, OpType op,
                               uint64_t input_bytes, uint64_t output_bytes,
                               uint64_t required_flops, int64_t layer = -1,
                               int64_t expert_id = -1) const;

    // Schedule a collection of active experts for a token across available compute devices
    std::vector<ScheduledTask> schedule_active_experts(
        int64_t layer, const int32_t* active_experts, int64_t k,
        uint64_t expert_bytes, uint64_t flops_per_expert);

    // Schedule general tensor operations (Attention, Recurrence, Norm, Projection)
    ScheduledTask schedule_op(OpType op, uint64_t input_bytes, uint64_t output_bytes,
                              uint64_t required_flops, int64_t layer = -1);

    // Dynamic work-stealing and load rebalancing across devices
    void update_device_load(int device_id, double current_utilization, int queue_depth);

    // Heterogeneous distribution report
    std::string print_distribution_summary() const;

private:
    std::shared_ptr<GenericExpertManager> expert_mgr_;
    std::unordered_map<int, uint64_t> tasks_scheduled_per_device_;
    std::unordered_map<int, double> total_compute_time_per_device_;
};

} // namespace strata::runtime
