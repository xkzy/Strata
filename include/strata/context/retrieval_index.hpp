// include/strata/context/retrieval_index.hpp - Built-In Hierarchical Retrieval Engine
//
// Model-independent, in-engine hierarchical index supporting exact, metadata,
// semantic, and section-to-chunk multi-tier retrieval over 1M-2M+ virtual tokens.
#pragma once

#include "strata/context/context_item.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace strata::context {

struct RetrievalQuery {
    std::string text;
    std::vector<std::string> tags;
    std::string filename_filter;
    std::string symbol_filter;
    double min_timestamp = 0.0;
    size_t top_k = 5;
    bool hierarchical_expand = true; // Extract original L0 chunk context when summary matches
};

struct SearchResult {
    int64_t item_id = -1;
    double relevance_score = 0.0;
    std::string matched_content; // Snippet or full text
    ContextItemType type;
    int64_t token_count = 0;
    std::string filename;
    std::string section_name;
    bool is_expanded_from_summary = false;
};

class RetrievalIndex {
public:
    virtual ~RetrievalIndex() = default;

    virtual void index_item(const ContextItem& item) = 0;
    virtual void remove_item(int64_t item_id) = 0;
    virtual void clear() = 0;

    virtual std::vector<SearchResult> search(const RetrievalQuery& query) const = 0;
    virtual size_t total_indexed_items() const = 0;
    virtual uint64_t total_indexed_tokens() const = 0;
};

// High-speed memory-efficient Hierarchical Inverted BM25 Index
class HierarchicalBM25Index : public RetrievalIndex {
public:
    HierarchicalBM25Index();
    ~HierarchicalBM25Index() override = default;

    void index_item(const ContextItem& item) override;
    void remove_item(int64_t item_id) override;
    void clear() override;

    std::vector<SearchResult> search(const RetrievalQuery& query) const override;
    size_t total_indexed_items() const override { return item_count_; }
    uint64_t total_indexed_tokens() const override { return total_tokens_; }

private:
    struct DocPosting {
        int64_t item_id;
        uint32_t term_freq;
        double score_weight;
    };

    struct DocumentRecord {
        int64_t item_id;
        ContextItemType type;
        int64_t token_count;
        double timestamp;
        std::string filename;
        std::string symbol_name;
        std::vector<std::string> tags;
        std::string content_snippet;
        std::string raw_content;
        int doc_length;
    };

    size_t item_count_ = 0;
    uint64_t total_tokens_ = 0;
    double avg_doc_length_ = 0.0;

    // Tokenized terms -> list of document postings
    std::unordered_map<std::string, std::vector<DocPosting>> inverted_index_;
    std::unordered_map<int64_t, DocumentRecord> doc_records_;

    // Tag and metadata indexes for fast hierarchical filtering
    std::unordered_map<std::string, std::unordered_set<int64_t>> tag_index_;
    std::unordered_map<std::string, std::unordered_set<int64_t>> file_index_;
    std::unordered_map<std::string, std::unordered_set<int64_t>> symbol_index_;

    std::vector<std::string> tokenize(const std::string& text) const;
    double compute_bm25_term_weight(double tf, double df, int doc_len) const;
};

} // namespace strata::context
