// src/guard/execution_guard.cpp - Execution Guard Implementation
#include "strata/guard/execution_guard.hpp"

#include <iostream>

namespace strata::guard {

ExecutionGuard::ExecutionGuard(const ExecutionBudget& budget)
    : budget_(budget), detector_(budget), start_time_(std::chrono::steady_clock::now()) {}

ExecutionGuard::~ExecutionGuard() = default;

GuardVerdict ExecutionGuard::check_action(const ActionRecord& action) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 1. Check Circuit Breaker
    std::string cb_reason;
    if (is_circuit_broken(&cb_reason)) {
        GuardVerdict verdict;
        verdict.state = EscalationState::kCancelled;
        verdict.allowed = false;
        verdict.reason = "Circuit breaker tripped: " + cb_reason;
        verdict.suggested_remediation = "Operation terminated due to hard resource budget exhaustion.";
        return verdict;
    }

    // 2. Add to Graph & Check Graph Topology (Recursion Depth, Fan-out, Cycles)
    graph_.add_node(action);

    // Recursion Depth Check
    int depth = graph_.calculate_recursion_depth(action.action_id);
    if (depth > budget_.max_recursive_depth) {
        GuardVerdict verdict;
        verdict.state = EscalationState::kCancelled;
        verdict.allowed = false;
        verdict.reason = "Maximum recursive execution depth exceeded (" + std::to_string(depth) +
                         " > " + std::to_string(budget_.max_recursive_depth) + ").";
        verdict.suggested_remediation = "Flatten task execution or avoid deeply nested tool delegations.";
        return verdict;
    }

    // Fan-Out Check
    if (!action.parent_id.empty()) {
        int fan_out = graph_.calculate_fan_out(action.parent_id);
        if (fan_out > budget_.max_fan_out_children) {
            GuardVerdict verdict;
            verdict.state = EscalationState::kBlocked;
            verdict.allowed = false;
            verdict.reason = "Fan-out explosion prevented: parent operation spawned " +
                             std::to_string(fan_out) + " concurrent child requests.";
            verdict.suggested_remediation = "Batch child requests or process sequentially.";
            return verdict;
        }
    }

    // Cross-Agent Cycle Check
    if (!action.agent_id.empty()) {
        std::vector<std::string> cycle_agents;
        if (graph_.detect_cross_agent_cycle(action.agent_id, cycle_agents)) {
            GuardVerdict verdict;
            verdict.state = EscalationState::kBlocked;
            verdict.allowed = false;
            std::string agent_chain;
            for (size_t i = 0; i < cycle_agents.size(); ++i) {
                agent_chain += cycle_agents[i] + (i + 1 < cycle_agents.size() ? " -> " : "");
            }
            verdict.reason = "Cross-agent cyclic delegation loop detected: " + agent_chain;
            verdict.suggested_remediation = "Resolve mutual agent task dependency directly without reciprocal calls.";
            return verdict;
        }
    }

    // 3. Evaluate Action in Loop Detector
    GuardVerdict verdict = detector_.evaluate_action(action);

    // Budget counter checks
    if (action.kind == ActionKind::kToolCall && tool_calls_ >= budget_.max_tool_calls) {
        verdict.state = EscalationState::kBlocked;
        verdict.allowed = false;
        verdict.reason = "Exceeded max tool calls budget (" + std::to_string(budget_.max_tool_calls) + ").";
    } else if (action.kind == ActionKind::kRetrieval && retrievals_ >= budget_.max_retrievals) {
        verdict.state = EscalationState::kBlocked;
        verdict.allowed = false;
        verdict.reason = "Exceeded max retrieval budget (" + std::to_string(budget_.max_retrievals) + ").";
    }

    return verdict;
}

void ExecutionGuard::record_action_outcome(const ActionRecord& action, bool state_changed,
                                          double progress_delta, int64_t tokens_used) {
    std::lock_guard<std::mutex> lock(mutex_);
    steps_++;
    tokens_used_ += tokens_used;

    if (action.kind == ActionKind::kToolCall) tool_calls_++;
    else if (action.kind == ActionKind::kRetrieval) retrievals_++;
    else if (action.kind == ActionKind::kInference) generations_++;

    detector_.record_action_outcome(action, state_changed, progress_delta);
}

bool ExecutionGuard::is_circuit_broken(std::string* out_reason) const {
    if (circuit_broken_) {
        if (out_reason) *out_reason = circuit_break_reason_;
        return true;
    }

    double elapsed = elapsed_time_sec();
    if (elapsed > budget_.max_wall_time_sec) {
        std::string r = "Wall-clock timeout (" + std::to_string(static_cast<int>(elapsed)) +
                        "s > " + std::to_string(static_cast<int>(budget_.max_wall_time_sec)) + "s)";
        if (out_reason) *out_reason = r;
        return true;
    }

    if (tokens_used_ > budget_.max_tokens_budget) {
        std::string r = "Token budget exhausted (" + std::to_string(tokens_used_) +
                        " > " + std::to_string(budget_.max_tokens_budget) + ")";
        if (out_reason) *out_reason = r;
        return true;
    }

    if (steps_ > budget_.max_steps) {
        std::string r = "Execution step budget exhausted (" + std::to_string(steps_) +
                        " > " + std::to_string(budget_.max_steps) + ")";
        if (out_reason) *out_reason = r;
        return true;
    }

    return false;
}

double ExecutionGuard::elapsed_time_sec() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_time_).count();
}

void ExecutionGuard::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    steps_ = 0;
    tool_calls_ = 0;
    retrievals_ = 0;
    generations_ = 0;
    tokens_used_ = 0;
    circuit_broken_ = false;
    circuit_break_reason_.clear();
    start_time_ = std::chrono::steady_clock::now();
    graph_.clear();
    detector_.reset();
}

void ExecutionGuard::set_budget(const ExecutionBudget& budget) {
    std::lock_guard<std::mutex> lock(mutex_);
    budget_ = budget;
    detector_.set_budget(budget);
}

} // namespace strata::guard
