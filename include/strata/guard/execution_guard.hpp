// include/strata/guard/execution_guard.hpp - Server-Side Execution Guard for Sessions and Agents
#pragma once

#include "strata/guard/anti_loop_types.hpp"
#include "strata/guard/execution_graph.hpp"
#include "strata/guard/loop_detector.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <string>

namespace strata::guard {

class ExecutionGuard {
public:
    explicit ExecutionGuard(const ExecutionBudget& budget = ExecutionBudget());
    ~ExecutionGuard();

    // Check before executing an action
    GuardVerdict check_action(const ActionRecord& action);

    // Record outcome of an action
    void record_action_outcome(const ActionRecord& action, bool state_changed,
                               double progress_delta, int64_t tokens_used = 0);

    // Check if hard resource limits have been exceeded (circuit breaker)
    bool is_circuit_broken(std::string* out_reason = nullptr) const;

    // Reset session guard
    void reset();

    // Subsystems & Accessors
    const ExecutionBudget& budget() const { return budget_; }
    void set_budget(const ExecutionBudget& budget);

    ExecutionGraph& graph() { return graph_; }
    const ExecutionGraph& graph() const { return graph_; }

    LoopDetector& detector() { return detector_; }
    const LoopDetector& detector() const { return detector_; }

    int64_t total_steps() const { return steps_; }
    int64_t total_tool_calls() const { return tool_calls_; }
    int64_t total_retrievals() const { return retrievals_; }
    int64_t total_generations() const { return generations_; }
    int64_t total_tokens() const { return tokens_used_; }
    double elapsed_time_sec() const;

private:
    ExecutionBudget budget_;
    mutable std::mutex mutex_;

    ExecutionGraph graph_;
    LoopDetector detector_;

    std::chrono::steady_clock::time_point start_time_;
    int64_t steps_ = 0;
    int64_t tool_calls_ = 0;
    int64_t retrievals_ = 0;
    int64_t generations_ = 0;
    int64_t tokens_used_ = 0;
    bool circuit_broken_ = false;
    std::string circuit_break_reason_;
};

} // namespace strata::guard
