// src/tools/tool_runtime.cpp - Tool Runtime Implementation
#include "strata/tools/tool_runtime.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace strata::tools {

ToolRuntime::ToolRuntime(std::shared_ptr<context::VirtualContextManager> vctx)
    : vctx_(std::move(vctx)),
      store_(std::make_unique<ToolStateStore>()),
      parser_(std::make_unique<CompositeToolParser>()) {}

ToolRuntime::~ToolRuntime() = default;

int64_t ToolRuntime::estimate_tokens(const std::string& text) const {
    if (text.empty()) return 0;
    return std::max<int64_t>(1, static_cast<int64_t>(text.size() / 4));
}

ResultCategory ToolRuntime::classify_output_size(int64_t raw_tokens) const {
    if (raw_tokens <= 300) return ResultCategory::kSmallInline;
    if (raw_tokens <= 1500) return ResultCategory::kMediumSummarized;
    return ResultCategory::kLargeExternal;
}

std::string ToolRuntime::generate_compact_observation(const ToolResult& res) const {
    std::ostringstream ss;
    const auto& s = res.structured;

    if (res.category == ResultCategory::kSmallInline) {
        // Inlined directly
        return res.raw_output;
    }

    if (s.is_build) {
        ss << "[Build " << (s.build_success ? "SUCCESS" : "FAILED")
           << " | Errors: " << s.error_count << ", Warnings: " << s.warning_count
           << " | Result ID: " << res.result_id << "]\n";
        if (!s.diagnostics.empty()) {
            ss << "Key Compiler Diagnostics:\n";
            for (size_t i = 0; i < std::min<size_t>(s.diagnostics.size(), 4); ++i) {
                const auto& d = s.diagnostics[i];
                ss << "  " << d.file << ":" << d.line << " [" << d.severity << "] " << d.message << "\n";
            }
        }
    } else if (s.is_test) {
        ss << "[Test Suite " << (s.tests_failed == 0 ? "PASSED" : "FAILED")
           << " | Passed: " << s.tests_passed << ", Failed: " << s.tests_failed
           << " | Result ID: " << res.result_id << "]\n";
        if (!s.failures.empty()) {
            ss << "Failures:\n";
            for (size_t i = 0; i < std::min<size_t>(s.failures.size(), 3); ++i) {
                ss << "  - " << s.failures[i].test_name << ": " << s.failures[i].failure_message << "\n";
            }
        }
    } else if (s.is_search) {
        ss << "[Search Query Completed | Total Matches: " << s.total_matches
           << " | Matched Files: " << s.matched_files.size()
           << " | Result ID: " << res.result_id << "]\n";
        if (!s.key_excerpts.empty()) {
            ss << "Sample Excerpts:\n";
            for (size_t i = 0; i < std::min<size_t>(s.key_excerpts.size(), 4); ++i) {
                ss << "  " << s.key_excerpts[i] << "\n";
            }
        }
    } else {
        // Generic Medium or Large Output
        ss << "[Tool " << res.tool_name << " Completed | Exit: " << s.exit_code
           << " | Raw Output: " << res.raw_tokens << " tokens (" << res.raw_bytes << " bytes)"
           << " | Result ID: " << res.result_id << "]\n";
        if (!s.key_excerpts.empty()) {
            ss << "Head Output Excerpt:\n";
            for (const auto& e : s.key_excerpts) {
                ss << "  " << e << "\n";
            }
        }
    }

    if (res.category == ResultCategory::kLargeExternal) {
        ss << "Note: Full output (" << res.raw_tokens << " tokens) archived in ToolStore. "
           << "Use retrieve_tool_result(result_id=" << res.result_id << ", query=...) to inspect specifics.";
    }

    return ss.str();
}

ToolResult ToolRuntime::process_tool_execution(const ToolExecutionRequest& request) {
    auto now_sec = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();

    ToolResult res;
    res.request_id = request.request_id;
    res.tool_name = request.tool_name;
    res.arguments_json = request.command_or_arguments;
    res.raw_output = request.raw_output;
    res.raw_bytes = request.raw_output.size();
    res.raw_tokens = estimate_tokens(request.raw_output);
    res.timestamp_sec = now_sec;
    res.status = (request.exit_code == 0) ? ToolStatus::kSuccess : ToolStatus::kFailed;
    res.category = classify_output_size(res.raw_tokens);

    // 1. Structured Parsing
    res.structured = parser_->parse(request.tool_name, request.raw_output, request.exit_code);
    res.structured.execution_time_ms = request.execution_time_ms;

    // 2. Generate Bounded Observation for Physical Context
    res.compact_observation = generate_compact_observation(res);
    res.observation_tokens = estimate_tokens(res.compact_observation);

    // 3. Save to Content-Addressed Store
    int64_t saved_id = store_->save_result(res);
    res.result_id = saved_id;

    // 4. Index into Virtual Context RAG
    if (vctx_) {
        vctx_->append_tool_result(request.tool_name, request.command_or_arguments,
                                 request.raw_output, request.exit_code, res.raw_tokens);
    }

    // 5. Log Event
    ToolEvent ev;
    ev.event_id = saved_id;
    ev.result_id = saved_id;
    ev.tool_name = request.tool_name;
    ev.action_description = request.command_or_arguments;
    ev.status = res.status;
    ev.timestamp_sec = now_sec;
    store_->log_event(ev);

    // Track statistics
    stats_.total_tool_calls++;
    stats_.total_raw_tool_tokens += res.raw_tokens;
    stats_.total_llm_visible_tokens += res.observation_tokens;
    stats_.total_raw_bytes_stored += res.raw_bytes;

    return res;
}

std::string ToolRuntime::retrieve_tool_fragment(int64_t result_id, const std::string& query,
                                               int line_start, int line_count, int64_t max_tokens) {
    stats_.total_fragment_queries++;
    ToolRetrievalQuery tq;
    tq.result_id = result_id;
    tq.query = query;
    tq.line_start = line_start;
    tq.line_count = line_count;
    tq.max_tokens = max_tokens;
    return store_->retrieve_fragment(tq);
}

ContextLease ToolRuntime::materialize_result(int64_t result_id, int64_t max_tokens) {
    stats_.total_leases_materialized++;
    return store_->materialize_lease(result_id, max_tokens);
}

ToolRuntimeStats ToolRuntime::get_stats() const {
    return stats_;
}

std::string ToolRuntime::print_diagnostics() const {
    auto st = get_stats();
    std::ostringstream ss;
    ss << "================== STRATA TOOL RUNTIME DIAGNOSTICS ==================\n";
    ss << "Total Tool Executions:      " << std::setw(10) << st.total_tool_calls << "\n";
    ss << "Raw Tool Tokens Produced:   " << std::setw(10) << st.total_raw_tool_tokens << " tokens\n";
    ss << "LLM-Visible Tokens Emitted: " << std::setw(10) << st.total_llm_visible_tokens << " tokens\n";
    ss << "Physical Context Reduction: " << std::setw(9) << std::fixed << std::setprecision(1)
       << st.context_reduction_ratio() << "x savings\n";
    ss << "Raw Bytes in CAS Storage:   " << std::setw(10) << (st.total_raw_bytes_stored / 1024) << " KB\n";
    ss << "Materialized Leases Issued: " << std::setw(10) << st.total_leases_materialized << "\n";
    ss << "Fragment Queries Handled:   " << std::setw(10) << st.total_fragment_queries << "\n";
    ss << "=====================================================================\n";
    return ss.str();
}

} // namespace strata::tools
