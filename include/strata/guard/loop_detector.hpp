// include/strata/guard/loop_detector.hpp - Pattern & Semantic Loop Detection
#pragma once

#include "strata/guard/anti_loop_types.hpp"

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::guard {

class LoopDetector {
public:
    explicit LoopDetector(const ExecutionBudget& budget = ExecutionBudget());
    ~LoopDetector();

    // Canonical normalization and fast 64-bit hashing
    static std::string normalize_text(const std::string& input);
    static std::string canonicalize_payload(const std::string& target_name, const std::string& payload);
    static uint64_t compute_hash(const std::string& text);
    static uint64_t compute_state_fingerprint(const std::vector<std::string>& state_elements);

    // Classify error category from error message
    static ErrorCategory categorize_error(const std::string& err_msg);

    // Evaluate an action record for loop patterns and progress
    GuardVerdict evaluate_action(const ActionRecord& action);

    // Record an action and its outcome to update internal pattern recognizers
    void record_action_outcome(const ActionRecord& action, bool state_changed, double progress_delta);

    // Reset detector state
    void reset();

    // Set / update budget configuration
    void set_budget(const ExecutionBudget& budget);
    const ExecutionBudget& budget() const { return budget_; }

    // Telemetry
    const GuardStats& stats() const { return stats_; }

private:
    // Sub-detectors
    bool check_exact_duplicate(const ActionRecord& action, int64_t& repeat_count);
    bool check_pattern_loop(const ActionRecord& action, int& out_period, int& out_repetitions);
    bool check_failure_loop(const ActionRecord& action, int64_t& fail_count, std::string& out_observation);
    bool check_retrieval_loop(const ActionRecord& action, int64_t& retrieval_repeat_count);

    ExecutionBudget budget_;
    mutable std::mutex mutex_;
    GuardStats stats_;

    std::deque<ActionRecord> action_history_;
    std::deque<uint64_t> hash_history_;
    std::deque<uint64_t> state_history_;

    // Failure tracking per canonical hash
    std::unordered_map<uint64_t, int64_t> failure_counts_;
    std::unordered_map<uint64_t, std::string> last_failure_message_;

    // Retrieval cache to suppress identical redundant queries
    std::unordered_map<uint64_t, int64_t> retrieval_query_counts_;
    std::unordered_map<uint64_t, std::string> retrieval_last_results_;

    int64_t consecutive_no_progress_ = 0;
    double current_progress_score_ = 1.0;
    static constexpr size_t kMaxHistoryWindow = 64;
};

} // namespace strata::guard
