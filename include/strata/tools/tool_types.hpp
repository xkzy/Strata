// include/strata/tools/tool_types.hpp - Tool Execution State & Structured Result Types
//
// Separates raw tool execution data from LLM-visible observations so massive
// raw tool outputs (logs, builds, test outputs, search dumps) do not consume physical context.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::tools {

enum class ToolStatus {
    kPending = 0,
    kRunning,
    kSuccess,
    kFailed,
    kTimeout,
    kCancelled
};

enum class ResultCategory {
    kSmallInline = 0,      // <= 300 tokens: Inlined directly into observation
    kMediumSummarized,    // 300 - 1500 tokens: Compact summary + selected excerpt + result ID
    kLargeExternal,       // > 1500 tokens: Metadata + structured stats + findings + result ID
    kBinaryFile,          // Binary/file artifact: Path, size, hash, metadata
    kStreamingLog         // Incremental / rolling log window
};

struct CompilerDiagnostic {
    std::string file;
    int line = 0;
    int column = 0;
    std::string severity; // "error", "warning", "note"
    std::string message;
};

struct TestFailure {
    std::string test_name;
    std::string failure_message;
    std::string source_location;
};

struct StructuredToolData {
    // Command / Process state
    int exit_code = 0;
    double execution_time_ms = 0.0;
    uint64_t stdout_bytes = 0;
    uint64_t stderr_bytes = 0;
    uint64_t total_lines = 0;

    // Build & Compilation state
    bool is_build = false;
    bool build_success = true;
    int error_count = 0;
    int warning_count = 0;
    std::vector<CompilerDiagnostic> diagnostics;

    // Test execution state
    bool is_test = false;
    int tests_passed = 0;
    int tests_failed = 0;
    int tests_skipped = 0;
    std::vector<TestFailure> failures;

    // Search / File state
    bool is_search = false;
    uint64_t total_matches = 0;
    std::vector<std::string> matched_files;
    std::vector<std::string> key_excerpts;

    // Arbitrary structured key-value attributes
    std::unordered_map<std::string, std::string> attributes;
};

struct ToolResult {
    int64_t result_id = 0;
    int64_t request_id = 0;
    std::string tool_name;
    std::string arguments_json;
    double timestamp_sec = 0.0;
    ToolStatus status = ToolStatus::kSuccess;
    ResultCategory category = ResultCategory::kSmallInline;

    // Content-Addressed Storage & Raw Info
    std::string content_hash;
    uint64_t raw_bytes = 0;
    int64_t raw_tokens = 0;
    std::string raw_output; // Authoritative raw data (stored in store, not injected to LLM)

    // Structured Parsing & Summarization
    StructuredToolData structured;
    std::string summary;

    // LLM-Visible Observation (Bounded representation for physical context)
    std::string compact_observation;
    int64_t observation_tokens = 0;
};

struct ToolEvent {
    int64_t event_id = 0;
    int64_t result_id = 0;
    std::string tool_name;
    std::string action_description;
    ToolStatus status = ToolStatus::kSuccess;
    double timestamp_sec = 0.0;
};

} // namespace strata::tools
