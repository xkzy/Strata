// include/strata/context/virtual_context.hpp - Virtual Context Engine
//
// Bridges massive 1M-2M+ token virtual context with small 8K-32K physical LLM contexts
// via dynamic paging, hierarchical memory, automatic compaction, and on-demand retrieval.
#pragma once

#include "strata/context/context_compactor.hpp"
#include "strata/context/context_item.hpp"
#include "strata/context/hierarchical_memory.hpp"
#include "strata/context/importance_scorer.hpp"
#include "strata/context/retrieval_index.hpp"
#include "strata/context/token_budget.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::context {

struct VirtualContextStats {
    uint64_t virtual_context_total_tokens = 0;
    uint64_t physical_active_tokens = 0;
    uint64_t working_memory_tokens = 0;
    uint64_t compressed_tokens = 0;
    uint64_t indexed_tokens = 0;
    uint64_t archived_tokens = 0;
    uint64_t total_items_tracked = 0;

    uint64_t compaction_events = 0;
    uint64_t total_tokens_freed_by_compaction = 0;

    uint64_t retrieval_queries = 0;
    uint64_t retrieval_hits = 0;
    double last_retrieval_latency_ms = 0.0;
    double last_compaction_latency_ms = 0.0;
};

struct PromptPayload {
    std::string assembled_prompt;
    int64_t total_prompt_tokens = 0;
    int64_t active_items_included = 0;
    int64_t retrieved_chunks_included = 0;
    bool has_summary = false;
    bool has_working_memory = false;
};

class VirtualContextManager {
public:
    explicit VirtualContextManager(int64_t physical_context_limit = 8192);
    ~VirtualContextManager();

    // Ingest new context items (messages, tool outputs, files, code diffs, logs)
    int64_t append_user_message(const std::string& content, int64_t approx_tokens = 0);
    int64_t append_assistant_message(const std::string& content, int64_t approx_tokens = 0);
    int64_t append_system_prompt(const std::string& content, int64_t approx_tokens = 0);

    // Tool & coding agent context objects
    int64_t append_tool_result(const std::string& tool_name, const std::string& command_or_call,
                               const std::string& output, int exit_code = 0, int64_t approx_tokens = 0);
    int64_t append_file_content(const std::string& filename, const std::string& content,
                                const std::vector<std::string>& symbols = {}, int64_t approx_tokens = 0);
    int64_t append_code_diff(const std::string& filename, const std::string& diff,
                             const std::string& commit_hash = "", int64_t approx_tokens = 0);

    // Working memory update
    void update_working_memory(const std::string& working_state, int64_t approx_tokens = 0);

    // Context Selection: builds optimized prompt within physical budget for the given query
    PromptPayload select_and_assemble_context(const std::string& current_query,
                                              const std::string& system_override = "");

    // Retrieve historical context on demand
    std::vector<SearchResult> retrieve_relevant(const std::string& query, size_t top_k = 5);

    // Manual or background compaction trigger
    CompactionResult trigger_compaction(const std::string& current_query = "");

    // Diagnostics and metrics
    VirtualContextStats get_stats() const;
    std::string print_diagnostics() const;

    // Direct access to components
    TokenBudgetManager& budget_manager() { return budget_mgr_; }
    RetrievalIndex& retrieval_index() { return *index_; }
    HierarchicalMemoryManager& memory_manager() { return *memory_mgr_; }

private:
    int64_t next_item_id_ = 1;
    TokenBudgetManager budget_mgr_;
    std::shared_ptr<ImportanceScorer> scorer_;
    std::shared_ptr<RetrievalIndex> index_;
    std::shared_ptr<HierarchicalMemoryManager> memory_mgr_;
    std::shared_ptr<ContextCompactor> compactor_;

    std::vector<std::shared_ptr<ContextItem>> active_items_;
    std::shared_ptr<ContextItem> system_prompt_item_;
    std::shared_ptr<ContextItem> working_memory_item_;

    mutable VirtualContextStats stats_;

    int64_t estimate_tokens(const std::string& text) const;
    double current_time_seconds() const;
};

} // namespace strata::context
