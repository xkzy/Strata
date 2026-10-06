// src/context/token_budget.cpp - Token Budget Manager Implementation
#include "strata/context/token_budget.hpp"

#include <algorithm>
#include <iostream>

namespace strata::context {

TokenBudgetManager::TokenBudgetManager(const TokenBudgetConfig& config)
    : config_(config) {}

bool TokenBudgetManager::needs_compaction(int64_t current_active_tokens, double threshold_pct) const {
    int64_t available_for_input = max_input_tokens();
    return current_active_tokens >= static_cast<int64_t>(available_for_input * threshold_pct);
}

TokenAllocation TokenBudgetManager::compute_allocation(int64_t requested_system,
                                                       int64_t requested_active_conv,
                                                       int64_t requested_working_memory,
                                                       int64_t requested_retrieval,
                                                       int64_t requested_summary) const {
    TokenAllocation alloc;
    alloc.generation_reserve = config_.generation_reserve;

    int64_t budget_for_input = max_input_tokens();
    if (budget_for_input <= 0) {
        return alloc;
    }

    // 1. System Prompt gets highest priority up to its ratio
    int64_t max_system = static_cast<int64_t>(budget_for_input * config_.system_ratio);
    alloc.system_tokens = std::min(requested_system, max_system);
    int64_t remaining = budget_for_input - alloc.system_tokens;

    // 2. Working memory (scratchpad state)
    int64_t max_wm = static_cast<int64_t>(budget_for_input * config_.working_memory_ratio);
    alloc.working_memory_tokens = std::min(requested_working_memory, std::min(max_wm, remaining));
    remaining -= alloc.working_memory_tokens;

    // 3. Active recent conversation turns
    int64_t max_conv = static_cast<int64_t>(budget_for_input * config_.active_conv_ratio);
    alloc.active_conv_tokens = std::min(requested_active_conv, std::min(max_conv, remaining));
    remaining -= alloc.active_conv_tokens;

    // 4. Retrieved historical context
    int64_t max_retrieval = static_cast<int64_t>(budget_for_input * config_.retrieval_ratio);
    alloc.retrieval_tokens = std::min(requested_retrieval, std::min(max_retrieval, remaining));
    remaining -= alloc.retrieval_tokens;

    // 5. High-level summaries
    alloc.summary_tokens = std::min(requested_summary, remaining);
    remaining -= alloc.summary_tokens;

    // If leftover budget exists, distribute surplus to active conversation or retrieval
    if (remaining > 0 && requested_active_conv > alloc.active_conv_tokens) {
        int64_t extra = std::min(remaining, requested_active_conv - alloc.active_conv_tokens);
        alloc.active_conv_tokens += extra;
        remaining -= extra;
    }
    if (remaining > 0 && requested_retrieval > alloc.retrieval_tokens) {
        int64_t extra = std::min(remaining, requested_retrieval - alloc.retrieval_tokens);
        alloc.retrieval_tokens += extra;
        remaining -= extra;
    }

    return alloc;
}

} // namespace strata::context
