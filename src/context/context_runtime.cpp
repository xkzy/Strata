// src/context/context_runtime.cpp - Generic Long-Context Runtime Implementation
#include "strata/context/context_runtime.hpp"

#include <iostream>

namespace strata::context {

LongContextRuntime::LongContextRuntime(const LongContextRuntimeOptions& options)
    : options_(options) {
    vctx_ = std::make_unique<VirtualContextManager>(options_.physical_context_limit);
}

LongContextRuntime::~LongContextRuntime() = default;

std::string LongContextRuntime::prepare_prompt(const std::string& user_query) {
    if (!vctx_) return user_query;

    vctx_->append_user_message(user_query);
    auto payload = vctx_->select_and_assemble_context(user_query);
    return payload.assembled_prompt;
}

std::string LongContextRuntime::print_context_summary() const {
    if (!vctx_) return "";
    return vctx_->print_diagnostics();
}

} // namespace strata::context
