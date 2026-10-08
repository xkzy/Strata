// include/strata/strata_unified.hpp - Strata Unified AI Runtime & OS
//
// Unifies:
// 1. Model Runtime (Generic MoE, Model Adapters, Expert Cache, Multi-model support)
// 2. Context Runtime (Virtual Context, Hierarchical Memory, RAG, Compaction, Token Budget)
// 3. Tool Runtime (Tool Execution, Content-Addressed Store, Structured Parsers, Bounded Observations)
// 4. Resource Runtime (Generic Devices, Memory Domains, Topology, Dynamic Scheduler)
#pragma once

#include "strata/context/context_runtime.hpp"
#include "strata/context/virtual_context.hpp"
#include "strata/generation/dynamic_temperature.hpp"
#include "strata/generation/generation_loop_detector.hpp"
#include "strata/guard/anti_loop_manager.hpp"
#include "strata/kernels/fast_activations.hpp"
#include "strata/kernels/fast_attention.hpp"
#include "strata/kernels/fast_matmul.hpp"
#include "strata/kernels/fast_moe.hpp"
#include "strata/kernels/fast_norm.hpp"
#include "strata/kernels/fast_quant.hpp"
#include "strata/kernels/fast_rope.hpp"
#include "strata/kernels/fast_softmax.hpp"
#include "strata/kernels/kernel_dispatcher.hpp"
#include "strata/kernels/math_policy.hpp"
#include "strata/logic_verification/logic_verification_runtime.hpp"
#include "strata/math/math_runtime.hpp"
#include "strata/models/model_adapter.hpp"
#include "strata/runtime/device.hpp"
#include "strata/runtime/memory.hpp"
#include "strata/runtime/runtime.hpp"
#include "strata/runtime/scheduler.hpp"
#include "strata/runtime/topology.hpp"
#include "strata/tools/tool_runtime.hpp"

#include <memory>
#include <string>

namespace strata {

struct UnifiedRuntimeOptions {
    std::string model_path;
    std::string architecture = "auto";
    int64_t physical_context_limit = 8192;
    int64_t virtual_context_limit = 2000000;
    bool enable_heterogeneous_scheduler = true;
    bool enable_auto_compaction = true;
    bool enable_tool_observation_compaction = true;
};

class StrataUnifiedRuntime {
public:
    explicit StrataUnifiedRuntime(const UnifiedRuntimeOptions& options = UnifiedRuntimeOptions());
    ~StrataUnifiedRuntime();

    // Initialize all runtime layers
    bool initialize(std::string& err);

    // Subsystems
    runtime::GenericMoERuntime& model_runtime() { return *model_runtime_; }
    context::VirtualContextManager& context_runtime() { return *vctx_; }
    tools::ToolRuntime& tool_runtime() { return *tool_runtime_; }
    math::MathRuntime& math_runtime() { return *math_runtime_; }
    logic::LogicVerificationRuntime& logic_verifier() { return *logic_verifier_; }
    runtime::HardwareTopology& topology() { return runtime::HardwareTopology::instance(); }

    // End-to-end interactive inference query
    std::string prepare_prompt_for_query(const std::string& user_query);

    // Diagnostics & Observability report
    std::string print_full_system_status() const;

private:
    UnifiedRuntimeOptions options_;
    bool initialized_ = false;

    std::unique_ptr<runtime::GenericMoERuntime> model_runtime_;
    std::shared_ptr<context::VirtualContextManager> vctx_;
    std::unique_ptr<tools::ToolRuntime> tool_runtime_;
    std::shared_ptr<math::MathRuntime> math_runtime_;
    std::unique_ptr<logic::LogicVerificationRuntime> logic_verifier_;
};

} // namespace strata
