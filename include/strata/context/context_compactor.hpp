// include/strata/context/context_compactor.hpp - Automatic Incremental Context Compaction
//
// Classifies context items into preserve, compress, index, and archive states,
// shrinking active physical footprint while indexing details into searchable memory.
#pragma once

#include "strata/context/context_item.hpp"
#include "strata/context/hierarchical_memory.hpp"
#include "strata/context/importance_scorer.hpp"
#include "strata/context/retrieval_index.hpp"
#include "strata/context/token_budget.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace strata::context {

struct CompactionResult {
    bool triggered = false;
    int64_t initial_active_tokens = 0;
    int64_t compacted_active_tokens = 0;
    int64_t tokens_freed = 0;
    size_t items_compressed = 0;
    size_t items_indexed = 0;
    size_t items_archived = 0;
    double duration_ms = 0.0;
};

class ContextCompactor {
public:
    ContextCompactor(std::shared_ptr<ImportanceScorer> scorer,
                     std::shared_ptr<RetrievalIndex> index,
                     std::shared_ptr<HierarchicalMemoryManager> memory_mgr);

    // Analyze active context items and compact them if token budget threshold is exceeded
    CompactionResult compact_if_needed(std::vector<std::shared_ptr<ContextItem>>& active_items,
                                       const TokenBudgetManager& budget_mgr,
                                       const std::string& current_query,
                                       double current_time_sec);

    // Force compaction to fit within a specific target token budget
    CompactionResult compact_to_target(std::vector<std::shared_ptr<ContextItem>>& active_items,
                                       int64_t target_max_tokens,
                                       const std::string& current_query,
                                       double current_time_sec);

private:
    std::shared_ptr<ImportanceScorer> scorer_;
    std::shared_ptr<RetrievalIndex> index_;
    std::shared_ptr<HierarchicalMemoryManager> memory_mgr_;

    void classify_and_process(ContextItem& item, double score);
};

} // namespace strata::context
