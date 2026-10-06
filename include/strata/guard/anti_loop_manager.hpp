// include/strata/guard/anti_loop_manager.hpp - Strata Unified Server-Side Anti-Loop Manager
#pragma once

#include "strata/guard/anti_loop_types.hpp"
#include "strata/guard/execution_guard.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::guard {

class AntiLoopManager {
public:
    static AntiLoopManager& instance();

    explicit AntiLoopManager(const ExecutionBudget& default_budget = ExecutionBudget());
    ~AntiLoopManager();

    // Get or create an execution guard for a specific session/agent scope
    std::shared_ptr<ExecutionGuard> get_or_create_guard(const std::string& tenant_id,
                                                        const std::string& session_id,
                                                        const std::string& agent_id = "");

    // Check action against appropriate scope guard
    GuardVerdict evaluate_action(const ActionRecord& action);

    // Record action completion and state progress
    void record_action_outcome(const ActionRecord& action, bool state_changed,
                               double progress_delta, int64_t tokens_used = 0);

    // Cross-session feedback loop protection
    bool check_cross_session_trigger(const std::string& source_session,
                                     const std::string& target_session,
                                     const std::string& resource_key);

    // Diagnostics & Observability
    GuardStats aggregate_stats() const;
    std::string print_diagnostics() const;

    // Prune stale guards
    void prune_inactive_guards(double max_idle_sec = 3600.0);

    // Reset all guards
    void reset();

    void set_default_budget(const ExecutionBudget& budget);
    const ExecutionBudget& default_budget() const { return default_budget_; }

private:
    std::string scope_key(const std::string& tenant_id, const std::string& session_id, const std::string& agent_id) const;

    ExecutionBudget default_budget_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<ExecutionGuard>> guards_;

    // Cross-session trigger tracking: (resource_key) -> history of (session_id, timestamp)
    struct SessionTriggerRecord {
        std::string source_session;
        std::string target_session;
        double timestamp_sec;
    };
    std::unordered_map<std::string, std::vector<SessionTriggerRecord>> cross_session_triggers_;
};

} // namespace strata::guard
