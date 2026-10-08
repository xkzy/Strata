// src/strata_unified.cpp - Strata Unified AI Runtime Implementation
#include "strata/strata_unified.hpp"

#include <iostream>
#include <sstream>

namespace strata {

StrataUnifiedRuntime::StrataUnifiedRuntime(const UnifiedRuntimeOptions& options)
    : options_(options) {}

StrataUnifiedRuntime::~StrataUnifiedRuntime() = default;

bool StrataUnifiedRuntime::initialize(std::string& err) {
    try {
        // 1. Initialize Context Runtime
        vctx_ = std::make_shared<context::VirtualContextManager>(options_.physical_context_limit);

        // 2. Initialize Tool Runtime (connected to Virtual Context)
        tool_runtime_ = std::make_unique<tools::ToolRuntime>(vctx_);

        // 3. Initialize Math Runtime (connected to Virtual Context)
        math_runtime_ = std::make_shared<math::MathRuntime>(vctx_);

        // 4. Initialize Deterministic Logic Verification Runtime
        logic_verifier_ = std::make_unique<logic::LogicVerificationRuntime>(vctx_, math_runtime_);

        // 5. Initialize Model Runtime (if model path provided)
        if (!options_.model_path.empty()) {
            runtime::RuntimeOptions model_opts;
            model_opts.model_path = options_.model_path;
            model_opts.architecture = options_.architecture;
            model_opts.max_context = options_.physical_context_limit;
            model_opts.enable_heterogeneous = options_.enable_heterogeneous_scheduler;

            model_runtime_ = std::make_unique<runtime::GenericMoERuntime>(model_opts);
            if (!model_runtime_->initialize(err)) {
                return false;
            }
        }

        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        err = std::string("Unified Runtime initialization failed: ") + e.what();
        return false;
    }
}

std::string StrataUnifiedRuntime::prepare_prompt_for_query(const std::string& user_query) {
    if (!vctx_) return user_query;

    vctx_->append_user_message(user_query);
    auto payload = vctx_->select_and_assemble_context(user_query);
    return payload.assembled_prompt;
}

std::string StrataUnifiedRuntime::print_full_system_status() const {
    std::ostringstream ss;
    ss << "######################################################################\n";
    ss << "               STRATA UNIFIED AI RUNTIME & OPERATING SYSTEM          \n";
    ss << "######################################################################\n\n";

    if (model_runtime_) {
        ss << model_runtime_->print_runtime_summary() << "\n";
    }

    if (vctx_) {
        ss << vctx_->print_diagnostics() << "\n";
    }

    if (tool_runtime_) {
        ss << tool_runtime_->print_diagnostics() << "\n";
    }

    if (math_runtime_) {
        ss << math_runtime_->print_diagnostics() << "\n";
    }

    if (logic_verifier_) {
        ss << logic_verifier_->print_diagnostics() << "\n";
    }

    ss << "######################################################################\n";
    return ss.str();
}

} // namespace strata
