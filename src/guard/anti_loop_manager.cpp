// src/guard/anti_loop_manager.cpp - Strata Unified Server-Side Anti-Loop Manager Implementation
#include "strata/guard/anti_loop_manager.hpp"

#include <chrono>
#include <iomanip>
#include <sstream>

namespace strata::guard {

AntiLoopManager& AntiLoopManager::instance() {
    static AntiLoopManager s_instance;
    return s_instance;
}

AntiLoopManager::AntiLoopManager(const ExecutionBudget& default_budget)
    : default_budget_(default_budget) {}

AntiLoopManager::~AntiLoopManager() = default;

std::string AntiLoopManager::scope_key(const std::string& tenant_id,
                                      const std::string& session_id,
                                      const std::string& agent_id) const {
    std::string t = tenant_id.empty() ? "default_tenant" : tenant_id;
    std::string s = session_id.empty() ? "global_session" : session_id;
    std::string a = agent_id.empty() ? "primary_agent" : agent_id;
    return t + "::" + s + "::" + a;
}

std::shared_ptr<ExecutionGuard> AntiLoopManager::get_or_create_guard(const std::string& tenant_id,
                                                                    const std::string& session_id,
                                                                    const std::string& agent_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = scope_key(tenant_id, session_id, agent_id);
    auto it = guards_.find(key);
    if (it != guards_.end()) {
        return it->second;
    }

    auto guard = std::make_shared<ExecutionGuard>(default_budget_);
    guards_[key] = guard;
    return guard;
}

GuardVerdict AntiLoopManager::evaluate_action(const ActionRecord& action) {
    auto guard = get_or_create_guard(action.tenant_id, action.session_id, action.agent_id);
    return guard->check_action(action);
}

void AntiLoopManager::record_action_outcome(const ActionRecord& action, bool state_changed,
                                           double progress_delta, int64_t tokens_used) {
    auto guard = get_or_create_guard(action.tenant_id, action.session_id, action.agent_id);
    guard->record_action_outcome(action, state_changed, progress_delta, tokens_used);
}

bool AntiLoopManager::check_cross_session_trigger(const std::string& source_session,
                                                  const std::string& target_session,
                                                  const std::string& resource_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (source_session.empty() || target_session.empty() || resource_key.empty()) return true;

    auto now_sec = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto& history = cross_session_triggers_[resource_key];

    // Prune history older than 60 seconds
    history.erase(std::remove_if(history.begin(), history.end(),
                                [&](const SessionTriggerRecord& r) {
                                    return (now_sec - r.timestamp_sec) > 60.0;
                                }),
                  history.end());

    // Check if target has repeatedly triggered source on this resource
    int ping_pong_count = 0;
    for (const auto& r : history) {
        if ((r.source_session == target_session && r.target_session == source_session) ||
            (r.source_session == source_session && r.target_session == target_session)) {
            ping_pong_count++;
        }
    }

    history.push_back({source_session, target_session, now_sec});

    // If ping-pong trigger count exceeds threshold, block cross-session loop
    return ping_pong_count <= 4;
}

GuardStats AntiLoopManager::aggregate_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    GuardStats total;
    double progress_sum = 0.0;

    for (const auto& kv : guards_) {
        const auto& s = kv.second->detector().stats();
        total.total_evaluations += s.total_evaluations;
        total.normal_count += s.normal_count;
        total.suspected_count += s.suspected_count;
        total.throttled_count += s.throttled_count;
        total.blocked_count += s.blocked_count;
        total.cancelled_count += s.cancelled_count;

        total.exact_duplicates_detected += s.exact_duplicates_detected;
        total.semantic_duplicates_detected += s.semantic_duplicates_detected;
        total.pattern_cycles_detected += s.pattern_cycles_detected;
        total.failure_loops_suppressed += s.failure_loops_suppressed;
        total.retrieval_loops_prevented += s.retrieval_loops_prevented;
        total.cross_agent_loops_broken += s.cross_agent_loops_broken;
        total.fan_out_explosions_capped += s.fan_out_explosions_capped;
        total.circuit_breaks_tripped += s.circuit_breaks_tripped;

        progress_sum += s.avg_progress_score;
    }

    if (!guards_.empty()) {
        total.avg_progress_score = progress_sum / guards_.size();
    }
    return total;
}

std::string AntiLoopManager::print_diagnostics() const {
    GuardStats st = aggregate_stats();
    std::ostringstream oss;
    oss << "================ STRATA ANTI-LOOP & RUNAWAY GUARD ================\n"
        << "Active Scopes:               " << guards_.size() << " guards\n"
        << "Total Evaluations:           " << st.total_evaluations << "\n"
        << "Normal / Safe Operations:    " << st.normal_count << "\n"
        << "Suspected / Throttled Steps: " << (st.suspected_count + st.throttled_count) << "\n"
        << "Blocked Redundant Actions:   " << st.blocked_count << "\n"
        << "Cancelled Runaways:          " << st.cancelled_count << "\n"
        << "Exact Duplicates Detected:   " << st.exact_duplicates_detected << "\n"
        << "Pattern Cycles Detected:     " << st.pattern_cycles_detected << "\n"
        << "Tool Failure Loops Stopped:  " << st.failure_loops_suppressed << "\n"
        << "Retrieval Loops Prevented:   " << st.retrieval_loops_prevented << "\n"
        << "Avg System Progress Score:   " << std::fixed << std::setprecision(2) << st.avg_progress_score << "\n"
        << "===================================================================";
    return oss.str();
}

void AntiLoopManager::prune_inactive_guards(double max_idle_sec) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = guards_.begin(); it != guards_.end();) {
        if (it->second->elapsed_time_sec() > max_idle_sec) {
            it = guards_.erase(it);
        } else {
            ++it;
        }
    }
}

void AntiLoopManager::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    guards_.clear();
    cross_session_triggers_.clear();
}

void AntiLoopManager::set_default_budget(const ExecutionBudget& budget) {
    std::lock_guard<std::mutex> lock(mutex_);
    default_budget_ = budget;
    for (auto& kv : guards_) {
        kv.second->set_budget(budget);
    }
}

} // namespace strata::guard
