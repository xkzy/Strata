// include/strata/runtime/runtime.hpp - Generic MoE Inference Runtime
//
// The central coordinator integrating:
// - Model Adapters (architecture agnostic)
// - Generic Device & Memory Domains
// - Hardware Topology Graph
// - Dynamic Expert Placement
// - Dynamic Heterogeneous Scheduler
#pragma once

#include "strata/models/model_adapter.hpp"
#include "strata/runtime/device.hpp"
#include "strata/runtime/expert_manager.hpp"
#include "strata/runtime/memory.hpp"
#include "strata/runtime/scheduler.hpp"
#include "strata/runtime/topology.hpp"

#include <memory>
#include <string>
#include <vector>

namespace strata::runtime {

struct RuntimeOptions {
    std::string model_path;
    std::string architecture = "auto";
    int64_t max_context = 32768;
    int64_t batch_size = 1;
    bool enable_heterogeneous = true;
    bool enable_dynamic_placement = true;
    bool calibrate_at_startup = true;
    int primary_device_id = 0;
};

struct ForwardMetrics {
    double routing_time_ms = 0.0;
    double schedule_time_ms = 0.0;
    double attention_time_ms = 0.0;
    double moe_compute_time_ms = 0.0;
    double moe_transfer_time_ms = 0.0;
    double total_forward_ms = 0.0;
    int64_t experts_executed = 0;
    int64_t cache_hits = 0;
    int64_t cache_misses = 0;
};

class GenericMoERuntime {
public:
    GenericMoERuntime(const RuntimeOptions& options);
    ~GenericMoERuntime();

    // Initialize hardware, topology, devices, and memory domains
    bool initialize(std::string& err);

    // Load model weights using registered model adapter
    bool load_model(const std::string& model_path, std::string& err);

    // Execute one forward step for a token
    bool forward_token(int64_t token_pos, int32_t token_id, float* logits_out,
                       ForwardMetrics* metrics = nullptr, std::string* err = nullptr);

    // Profile & calibrate runtime
    void calibrate(bool quick = true);

    // Getters for subsystems
    const models::MoEModelAdapter* model_adapter() const { return model_adapter_.get(); }
    const GenericExpertManager* expert_manager() const { return expert_mgr_.get(); }
    const DynamicScheduler* scheduler() const { return scheduler_.get(); }
    const RuntimeOptions& options() const { return options_; }

    // Print runtime configuration, topology, and execution summary
    std::string print_runtime_summary() const;

private:
    RuntimeOptions options_;
    bool initialized_ = false;

    std::unique_ptr<models::MoEModelAdapter> model_adapter_;
    std::shared_ptr<GenericExpertManager> expert_mgr_;
    std::unique_ptr<DynamicScheduler> scheduler_;
    std::unique_ptr<core::WeightTable> weight_table_;
};

} // namespace strata::runtime
