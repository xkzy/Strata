// include/strata/context/token_budget.hpp - Dynamic Physical Context Token Budgeting
//
// Dynamically allocates available physical context among system instructions, user query,
// active conversation, working memory, retrieved information, and generation reserve.
#pragma once

#include <cstdint>
#include <string>

namespace strata::context {

struct TokenBudgetConfig {
    int64_t physical_context_limit = 8192; // 8K, 16K, 32K physical model context
    int64_t generation_reserve = 1024;      // Tokens reserved for model output

    // Dynamic targets / ratios
    double system_ratio = 0.15;            // Max ~15% for system prompt
    double active_conv_ratio = 0.35;       // ~35% for recent conversation turns
    double working_memory_ratio = 0.20;    // ~20% for working scratchpad / active state
    double retrieval_ratio = 0.20;         // ~20% for retrieved historical context
    double summary_ratio = 0.10;           // ~10% for high-level summaries
};

struct TokenAllocation {
    int64_t system_tokens = 0;
    int64_t active_conv_tokens = 0;
    int64_t working_memory_tokens = 0;
    int64_t retrieval_tokens = 0;
    int64_t summary_tokens = 0;
    int64_t generation_reserve = 0;

    int64_t total_allocated() const {
        return system_tokens + active_conv_tokens + working_memory_tokens +
               retrieval_tokens + summary_tokens + generation_reserve;
    }
};

class TokenBudgetManager {
public:
    explicit TokenBudgetManager(const TokenBudgetConfig& config = TokenBudgetConfig());

    const TokenBudgetConfig& config() const { return config_; }
    void set_config(const TokenBudgetConfig& cfg) { config_ = cfg; }

    // Computes dynamic budget breakdown based on actual demand
    TokenAllocation compute_allocation(int64_t requested_system,
                                       int64_t requested_active_conv,
                                       int64_t requested_working_memory,
                                       int64_t requested_retrieval,
                                       int64_t requested_summary) const;

    // Checks if physical context usage exceeds compaction threshold (e.g. 85%)
    bool needs_compaction(int64_t current_active_tokens, double threshold_pct = 0.85) const;

    // Max usable context for input (physical limit minus generation reserve)
    int64_t max_input_tokens() const {
        return config_.physical_context_limit - config_.generation_reserve;
    }

private:
    TokenBudgetConfig config_;
};

} // namespace strata::context
