// include/strata/context/hierarchical_memory.hpp - Multi-Tier Hierarchical Context Memory
//
// Maintains L0 (immutable raw events) to L4 (global state) summaries, preventing
// recursive summary degradation by always grounding representations in authoritative source data.
#pragma once

#include "strata/context/context_item.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::context {

struct HierarchyNode {
    int64_t node_id;
    int level; // 0 = Raw, 1 = Chunk, 2 = Section, 3 = Session, 4 = Global
    std::string summary_text;
    int64_t token_count;
    std::vector<int64_t> child_item_ids;
    double created_timestamp;
};

class HierarchicalMemoryManager {
public:
    HierarchicalMemoryManager();
    ~HierarchicalMemoryManager();

    // Register an immutable L0 raw item
    void add_raw_item(std::shared_ptr<ContextItem> item);

    // Generate or update hierarchical summaries incrementally (L1, L2, L3)
    void update_hierarchy_incremental(int64_t chunk_token_threshold = 1024,
                                      int64_t section_token_threshold = 8192);

    // Retrieve original source item by ID (never destroyed)
    std::shared_ptr<ContextItem> get_source_item(int64_t item_id) const;

    // Get current high-level session / global summaries for prompt injection
    std::string get_session_summary() const;
    std::string get_global_state() const;
    void set_global_state(const std::string& state);

    // Query hierarchy nodes
    std::vector<HierarchyNode> get_level_nodes(int level) const;

    // Statistics
    uint64_t total_raw_tokens() const { return total_raw_tokens_; }
    uint64_t total_summary_tokens() const { return total_summary_tokens_; }
    size_t total_raw_items() const { return raw_items_.size(); }

private:
    mutable std::mutex mutex_;
    std::unordered_map<int64_t, std::shared_ptr<ContextItem>> raw_items_;
    std::vector<int64_t> chronological_order_;

    std::vector<HierarchyNode> l1_chunks_;
    std::vector<HierarchyNode> l2_sections_;
    HierarchyNode l3_session_;
    std::string l4_global_state_;

    uint64_t total_raw_tokens_ = 0;
    uint64_t total_summary_tokens_ = 0;
    int64_t next_node_id_ = 1;

    std::string create_extractive_summary(const std::vector<std::shared_ptr<ContextItem>>& items,
                                          size_t max_sentences = 3) const;
};

} // namespace strata::context
