// include/strata/guard/anti_loop_types.hpp - Strata Server-Side Anti-Loop & Runaway Guard Types
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::guard {

/// Loop escalation states
enum class EscalationState {
    kNormal = 0,    // Normal healthy execution
    kSuspected,     // Low progress or initial repetition detected; monitoring closely
    kThrottled,     // Slowed execution / cool-down delay / reduced step budget
    kBlocked,       // Operation suppressed / cached result returned / retry halted
    kCancelled      // Hard stop: runaway execution or no-progress threshold reached
};

inline const char* escalation_state_to_string(EscalationState s) {
    switch (s) {
        case EscalationState::kNormal: return "NORMAL";
        case EscalationState::kSuspected: return "SUSPECTED";
        case EscalationState::kThrottled: return "THROTTLED";
        case EscalationState::kBlocked: return "BLOCKED";
        case EscalationState::kCancelled: return "CANCELLED";
        default: return "UNKNOWN";
    }
}

/// Action kinds in the execution graph
enum class ActionKind {
    kInference = 0,
    kToolCall,
    kToolResult,
    kRetrieval,
    kContextCompaction,
    kAgentDelegation,
    kCrossSessionEvent
};

inline const char* action_kind_to_string(ActionKind k) {
    switch (k) {
        case ActionKind::kInference: return "inference";
        case ActionKind::kToolCall: return "tool_call";
        case ActionKind::kToolResult: return "tool_result";
        case ActionKind::kRetrieval: return "retrieval";
        case ActionKind::kContextCompaction: return "context_compaction";
        case ActionKind::kAgentDelegation: return "agent_delegation";
        case ActionKind::kCrossSessionEvent: return "cross_session_event";
        default: return "unknown";
    }
}

/// Categorization of errors for smart retry decisions
enum class ErrorCategory {
    kNone = 0,
    kTransientNetwork,  // Safe to retry with exponential backoff
    kRateLimited,       // Safe to retry after cooldown
    kSyntaxOrSchema,    // Deterministic: identical retry will fail
    kNotFoundOrPath,    // Deterministic unless environment changes
    kPermissionDenied,  // Deterministic
    kExecutionPanic,    // Deterministic
    kUnknown
};

inline const char* error_category_to_string(ErrorCategory c) {
    switch (c) {
        case ErrorCategory::kNone: return "NONE";
        case ErrorCategory::kTransientNetwork: return "TRANSIENT_NETWORK";
        case ErrorCategory::kRateLimited: return "RATE_LIMITED";
        case ErrorCategory::kSyntaxOrSchema: return "SYNTAX_OR_SCHEMA";
        case ErrorCategory::kNotFoundOrPath: return "NOT_FOUND";
        case ErrorCategory::kPermissionDenied: return "PERMISSION_DENIED";
        case ErrorCategory::kExecutionPanic: return "EXECUTION_PANIC";
        default: return "UNKNOWN";
    }
}

/// Record of an individual action in the execution graph
struct ActionRecord {
    std::string action_id;
    std::string parent_id;
    std::string tenant_id;
    std::string user_id;
    std::string agent_id;
    std::string session_id;
    std::string request_id;

    ActionKind kind = ActionKind::kInference;
    std::string target_name;        // e.g. tool name, query string, prompt snippet
    std::string normalized_payload; // canonicalized arguments/query
    uint64_t canonical_hash = 0;     // hash of normalized payload
    uint64_t state_fingerprint = 0;  // hash of relevant system/context state

    double timestamp_sec = 0.0;
    double duration_ms = 0.0;
    bool is_error = false;
    ErrorCategory error_category = ErrorCategory::kNone;
    std::string error_message;
    std::string result_summary;
};

/// Configurable multi-dimensional execution budgets
struct ExecutionBudget {
    int64_t max_requests = 1000;
    int64_t max_steps = 100;
    int64_t max_tool_calls = 50;
    int64_t max_retrievals = 40;
    int64_t max_generations = 80;
    double max_wall_time_sec = 300.0;
    int64_t max_tokens_budget = 2000000;

    // Progress and repetition limits
    int64_t max_no_progress_steps = 5;
    int64_t max_identical_tool_calls = 3;
    int64_t max_identical_failures = 3;
    int64_t max_identical_retrievals = 3;
    int64_t max_pattern_repetitions = 3; // e.g. A->B->A->B->A->B

    // Fan-out and recursion protection
    int64_t max_recursive_depth = 8;
    int64_t max_fan_out_children = 16;
    int64_t max_concurrent_operations = 8;
};

/// Structured assessment returned by execution guard
struct GuardVerdict {
    EscalationState state = EscalationState::kNormal;
    bool allowed = true;
    bool is_throttled = false;
    double throttle_delay_ms = 0.0;

    std::string reason;
    std::string suggested_remediation;
    std::string structured_observation; // Injected into LLM context on failure suppression

    double progress_score = 1.0;
    int64_t current_step = 0;
    int64_t no_progress_streak = 0;
    uint64_t loop_pattern_signature = 0;
};

/// Aggregated telemetry and statistics
struct GuardStats {
    uint64_t total_evaluations = 0;
    uint64_t normal_count = 0;
    uint64_t suspected_count = 0;
    uint64_t throttled_count = 0;
    uint64_t blocked_count = 0;
    uint64_t cancelled_count = 0;

    uint64_t exact_duplicates_detected = 0;
    uint64_t semantic_duplicates_detected = 0;
    uint64_t pattern_cycles_detected = 0;
    uint64_t failure_loops_suppressed = 0;
    uint64_t retrieval_loops_prevented = 0;
    uint64_t cross_agent_loops_broken = 0;
    uint64_t fan_out_explosions_capped = 0;
    uint64_t circuit_breaks_tripped = 0;

    double avg_progress_score = 1.0;
};

} // namespace strata::guard
