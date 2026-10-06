// src/context/context_compactor.cpp - Automatic Context Compactor Implementation
#include "strata/context/context_compactor.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>

namespace strata::context {

ContextCompactor::ContextCompactor(std::shared_ptr<ImportanceScorer> scorer,
                                   std::shared_ptr<RetrievalIndex> index,
                                   std::shared_ptr<HierarchicalMemoryManager> memory_mgr)
    : scorer_(std::move(scorer)), index_(std::move(index)), memory_mgr_(std::move(memory_mgr)) {}

void ContextCompactor::classify_and_process(ContextItem& item, double score) {
    // If item was already active, decide transition based on score
    if (item.type() == ContextItemType::kSystemPrompt) {
        // System prompt is never compressed or evicted
        item.set_state(ResidencyState::kActive);
        return;
    }

    if (score >= 0.70) {
        // High importance: keep fully active in physical context
        item.set_state(ResidencyState::kActive);
    } else if (score >= 0.40) {
        // Medium importance: compress in active context, index full content in RAG
        if (index_) {
            index_->index_item(item);
        }
        item.set_state(ResidencyState::kCompressed);
        // Reduce effective token count to summary size
        int64_t compressed_tokens = std::max<int64_t>(15, item.token_count() / 5);
        item.set_token_count(compressed_tokens);
    } else {
        // Low importance / stale: move to searchable index / archive, remove from active context
        if (index_) {
            index_->index_item(item);
        }
        item.set_state(ResidencyState::kIndexed);
        item.set_token_count(0);
    }
}

CompactionResult ContextCompactor::compact_if_needed(
    std::vector<std::shared_ptr<ContextItem>>& active_items,
    const TokenBudgetManager& budget_mgr,
    const std::string& current_query,
    double current_time_sec) {

    int64_t current_active_tokens = 0;
    for (const auto& it : active_items) {
        if (it && it->state() == ResidencyState::kActive) {
            current_active_tokens += it->token_count();
        }
    }

    if (!budget_mgr.needs_compaction(current_active_tokens, 0.85)) {
        CompactionResult res;
        res.triggered = false;
        res.initial_active_tokens = current_active_tokens;
        res.compacted_active_tokens = current_active_tokens;
        return res;
    }

    int64_t target_budget = static_cast<int64_t>(budget_mgr.max_input_tokens() * 0.70); // Compact down to 70%
    return compact_to_target(active_items, target_budget, current_query, current_time_sec);
}

CompactionResult ContextCompactor::compact_to_target(
    std::vector<std::shared_ptr<ContextItem>>& active_items,
    int64_t target_max_tokens,
    const std::string& current_query,
    double current_time_sec) {

    auto t0 = std::chrono::high_resolution_clock::now();
    CompactionResult res;
    res.triggered = true;

    for (const auto& it : active_items) {
        if (it) res.initial_active_tokens += it->token_count();
    }

    // Score all items
    std::vector<std::pair<std::shared_ptr<ContextItem>, double>> scored_items;
    scored_items.reserve(active_items.size());

    for (const auto& it : active_items) {
        if (!it) continue;
        double s = scorer_ ? scorer_->score(*it, current_query, current_time_sec) : 0.5;
        it->set_importance_score(s);
        scored_items.push_back({it, s});
    }

    // Sort ascending by score (lowest score compacted first)
    std::sort(scored_items.begin(), scored_items.end(),
              [](const auto& a, const auto& b) {
                  // Keep system prompt always last to compact
                  if (a.first->type() == ContextItemType::kSystemPrompt) return false;
                  if (b.first->type() == ContextItemType::kSystemPrompt) return true;
                  return a.second < b.second;
              });

    int64_t current_tokens = res.initial_active_tokens;
    std::vector<std::shared_ptr<ContextItem>> surviving_active;

    for (auto& pair : scored_items) {
        auto& item = pair.first;
        double score = pair.second;

        if (current_tokens > target_max_tokens && item->type() != ContextItemType::kSystemPrompt) {
            int64_t before_tokens = item->token_count();
            classify_and_process(*item, score);
            int64_t after_tokens = item->token_count();
            current_tokens -= (before_tokens - after_tokens);

            if (item->state() == ResidencyState::kCompressed) {
                res.items_compressed++;
                surviving_active.push_back(item);
            } else if (item->state() == ResidencyState::kIndexed) {
                res.items_indexed++;
            }
        } else {
            item->set_state(ResidencyState::kActive);
            surviving_active.push_back(item);
        }
    }

    // Reorder surviving active items chronologically
    std::sort(surviving_active.begin(), surviving_active.end(),
              [](const auto& a, const auto& b) { return a->timestamp() < b->timestamp(); });

    active_items = std::move(surviving_active);

    // Trigger hierarchical memory update
    if (memory_mgr_) {
        memory_mgr_->update_hierarchy_incremental();
    }

    res.compacted_active_tokens = current_tokens;
    res.tokens_freed = res.initial_active_tokens - res.compacted_active_tokens;

    auto t1 = std::chrono::high_resolution_clock::now();
    res.duration_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return res;
}

} // namespace strata::context
