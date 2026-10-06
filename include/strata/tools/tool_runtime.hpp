// include/strata/tools/tool_runtime.hpp - Tool Runtime & Observation Engine
//
// Manages tool execution, parses structured state, enforces large tool output policies,
// and generates bounded observations for physical LLM context.
#pragma once

#include "strata/context/virtual_context.hpp"
#include "strata/tools/tool_parser.hpp"
#include "strata/tools/tool_store.hpp"
#include "strata/tools/tool_types.hpp"

#include <memory>
#include <string>

namespace strata::tools {

struct ToolExecutionRequest {
    std::string tool_name;
    std::string command_or_arguments;
    std::string raw_output;
    int exit_code = 0;
    double execution_time_ms = 0.0;
    int64_t request_id = 0;
};

struct ToolRuntimeStats {
    uint64_t total_tool_calls = 0;
    uint64_t total_raw_tool_tokens = 0;
    uint64_t total_llm_visible_tokens = 0;
    uint64_t total_raw_bytes_stored = 0;
    uint64_t total_leases_materialized = 0;
    uint64_t total_fragment_queries = 0;

    double context_reduction_ratio() const {
        if (total_llm_visible_tokens == 0) return 1.0;
        return static_cast<double>(total_raw_tool_tokens) / static_cast<double>(total_llm_visible_tokens);
    }
};

class ToolRuntime {
public:
    explicit ToolRuntime(std::shared_ptr<context::VirtualContextManager> vctx = nullptr);
    ~ToolRuntime();

    // Process a tool execution: store raw output, parse structure, generate compact observation, and index into RAG
    ToolResult process_tool_execution(const ToolExecutionRequest& request);

    // Retrieve fragment of stored raw output without bloating LLM context
    std::string retrieve_tool_fragment(int64_t result_id, const std::string& query,
                                       int line_start = 0, int line_count = 50,
                                       int64_t max_tokens = 500);

    // Materialize temporary context lease
    ContextLease materialize_result(int64_t result_id, int64_t max_tokens = 1000);

    // Diagnostics & Observability
    ToolRuntimeStats get_stats() const;
    std::string print_diagnostics() const;

    // Direct access to state store and parser
    ToolStateStore& store() { return *store_; }
    const ToolStateStore& store() const { return *store_; }

private:
    std::shared_ptr<context::VirtualContextManager> vctx_;
    std::unique_ptr<ToolStateStore> store_;
    std::unique_ptr<CompositeToolParser> parser_;

    mutable ToolRuntimeStats stats_;

    ResultCategory classify_output_size(int64_t raw_tokens) const;
    std::string generate_compact_observation(const ToolResult& res) const;
    int64_t estimate_tokens(const std::string& text) const;
};

} // namespace strata::tools
