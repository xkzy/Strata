// include/strata/context/context_runtime.hpp - Generic Long-Context Inference Runtime
//
// Integrates Virtual Context Management with the Strata inference engine.
#pragma once

#include "strata/context/virtual_context.hpp"
#include "strata/runtime/runtime.hpp"

#include <memory>
#include <string>

namespace strata::context {

struct LongContextRuntimeOptions {
    int64_t physical_context_limit = 8192;   // Physical model context (8K, 16K, 32K)
    int64_t virtual_context_capacity = 2000000; // 2 Million+ virtual tokens
    bool enable_auto_compaction = true;
    bool enable_hierarchical_rag = true;
    bool enable_code_awareness = true;
    double compaction_threshold = 0.85;
};

class LongContextRuntime {
public:
    explicit LongContextRuntime(const LongContextRuntimeOptions& options = LongContextRuntimeOptions());
    ~LongContextRuntime();

    // Virtual context operations
    VirtualContextManager& virtual_context() { return *vctx_; }
    const VirtualContextManager& virtual_context() const { return *vctx_; }

    // Execute one query turn over virtual context
    std::string prepare_prompt(const std::string& user_query);

    // Diagnostics & stats
    std::string print_context_summary() const;

private:
    LongContextRuntimeOptions options_;
    std::unique_ptr<VirtualContextManager> vctx_;
};

} // namespace strata::context
