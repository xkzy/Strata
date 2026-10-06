// src/context/retrieval_index.cpp - Hierarchical Retrieval Index Implementation
#include "strata/context/retrieval_index.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <sstream>

namespace strata::context {

namespace {
struct AsciiLut {
    bool is_token_char[256]{};
    char tolower_char[256]{};
    AsciiLut() {
        for (int i = 0; i < 256; ++i) {
            unsigned char c = static_cast<unsigned char>(i);
            if (std::isalnum(c) || c == '_') {
                is_token_char[i] = true;
            }
            tolower_char[i] = static_cast<char>(std::tolower(c));
        }
    }
};
const AsciiLut& get_ascii_lut() {
    static const AsciiLut lut;
    return lut;
}
} // anonymous namespace

HierarchicalBM25Index::HierarchicalBM25Index() = default;

std::vector<std::string> HierarchicalBM25Index::tokenize(const std::string& text) const {
    const auto& lut = get_ascii_lut();
    std::vector<std::string> tokens;
    tokens.reserve(text.size() / 6 + 4);

    size_t start = 0;
    const size_t len = text.size();
    while (start < len) {
        while (start < len && !lut.is_token_char[static_cast<unsigned char>(text[start])]) {
            ++start;
        }
        if (start >= len) break;

        size_t end = start;
        while (end < len && lut.is_token_char[static_cast<unsigned char>(text[end])]) {
            ++end;
        }

        if (end - start >= 2) {
            std::string tok(end - start, '\0');
            for (size_t i = 0; i < end - start; ++i) {
                tok[i] = lut.tolower_char[static_cast<unsigned char>(text[start + i])];
            }
            tokens.push_back(std::move(tok));
        }
        start = end;
    }
    return tokens;
}

double HierarchicalBM25Index::compute_bm25_term_weight(double tf, double df, int doc_len) const {
    const double k1 = 1.2;
    const double b = 0.75;
    double N = static_cast<double>(item_count_ > 0 ? item_count_ : 1);
    double idf = std::log((N - df + 0.5) / (df + 0.5) + 1.0);
    double avgdl = avg_doc_length_ > 0 ? avg_doc_length_ : 50.0;
    double num = tf * (k1 + 1.0);
    double den = tf + k1 * (1.0 - b + b * (doc_len / avgdl));
    return idf * (num / den);
}

void HierarchicalBM25Index::index_item(const ContextItem& item) {
    int64_t id = item.id();
    std::string text_to_index = item.raw_content();
    if (!item.summary_l1().empty()) {
        text_to_index += " " + item.summary_l1();
    }

    auto tokens = tokenize(text_to_index);
    int doc_len = static_cast<int>(tokens.size());

    // Term frequencies
    std::unordered_map<std::string, uint32_t> tf_map;
    for (const auto& t : tokens) {
        tf_map[t]++;
    }

    // Add postings to inverted index
    for (const auto& kv : tf_map) {
        inverted_index_[kv.first].push_back(DocPosting{id, kv.second, 0.0});
    }

    // Save Document Record
    DocumentRecord rec;
    rec.item_id = id;
    rec.type = item.type();
    rec.token_count = item.token_count();
    rec.timestamp = item.timestamp();
    rec.filename = item.metadata().filename;
    rec.symbol_name = item.metadata().symbol_name;
    rec.tags = item.metadata().tags;
    rec.content_snippet = item.raw_content().substr(0, std::min<size_t>(item.raw_content().size(), 300));
    rec.raw_content = item.raw_content();
    rec.doc_length = doc_len;

    doc_records_[id] = rec;

    // Index metadata
    if (!rec.filename.empty()) file_index_[rec.filename].insert(id);
    if (!rec.symbol_name.empty()) symbol_index_[rec.symbol_name].insert(id);
    for (const auto& tag : rec.tags) {
        tag_index_[tag].insert(id);
    }

    item_count_++;
    total_tokens_ += item.token_count();
    avg_doc_length_ = (avg_doc_length_ * (item_count_ - 1) + doc_len) / item_count_;
}

void HierarchicalBM25Index::remove_item(int64_t item_id) {
    auto it = doc_records_.find(item_id);
    if (it == doc_records_.end()) return;

    total_tokens_ -= it->second.token_count;
    doc_records_.erase(it);
    item_count_ = doc_records_.size();

    // Clean metadata indices
    for (auto& kv : file_index_) kv.second.erase(item_id);
    for (auto& kv : symbol_index_) kv.second.erase(item_id);
    for (auto& kv : tag_index_) kv.second.erase(item_id);
}

void HierarchicalBM25Index::clear() {
    inverted_index_.clear();
    doc_records_.clear();
    file_index_.clear();
    symbol_index_.clear();
    tag_index_.clear();
    item_count_ = 0;
    total_tokens_ = 0;
    avg_doc_length_ = 0.0;
}

std::vector<SearchResult> HierarchicalBM25Index::search(const RetrievalQuery& query) const {
    std::vector<SearchResult> results;
    if (query.text.empty() && query.tags.empty() && query.filename_filter.empty()) {
        return results;
    }

    auto query_tokens = tokenize(query.text);
    std::unordered_map<int64_t, double> doc_scores;

    // 1. BM25 text match
    for (const auto& qterm : query_tokens) {
        auto it = inverted_index_.find(qterm);
        if (it != inverted_index_.end()) {
            double df = static_cast<double>(it->second.size());
            for (const auto& posting : it->second) {
                auto doc_it = doc_records_.find(posting.item_id);
                if (doc_it != doc_records_.end()) {
                    double weight = compute_bm25_term_weight(posting.term_freq, df, doc_it->second.doc_length);
                    doc_scores[posting.item_id] += weight;
                }
            }
        }
    }

    // 2. Metadata filtering & boosting
    if (!query.filename_filter.empty()) {
        auto f_it = file_index_.find(query.filename_filter);
        if (f_it != file_index_.end()) {
            for (int64_t id : f_it->second) {
                doc_scores[id] += 5.0; // Significant boost for exact file match
            }
        }
    }

    if (!query.symbol_filter.empty()) {
        auto s_it = symbol_index_.find(query.symbol_filter);
        if (s_it != symbol_index_.end()) {
            for (int64_t id : s_it->second) {
                doc_scores[id] += 4.0;
            }
        }
    }

    for (const auto& tag : query.tags) {
        auto t_it = tag_index_.find(tag);
        if (t_it != tag_index_.end()) {
            for (int64_t id : t_it->second) {
                doc_scores[id] += 2.0;
            }
        }
    }

    // Rank candidate documents
    std::vector<std::pair<int64_t, double>> ranked;
    for (const auto& kv : doc_scores) {
        if (kv.second > 0.0) {
            ranked.push_back(kv);
        }
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    size_t limit = std::min(query.top_k, ranked.size());
    for (size_t i = 0; i < limit; ++i) {
        int64_t doc_id = ranked[i].first;
        const auto& rec = doc_records_.at(doc_id);

        SearchResult res;
        res.item_id = doc_id;
        res.relevance_score = ranked[i].second;
        res.type = rec.type;
        res.token_count = rec.token_count;
        res.filename = rec.filename;
        res.section_name = rec.symbol_name;
        res.matched_content = query.hierarchical_expand ? rec.raw_content : rec.content_snippet;
        res.is_expanded_from_summary = query.hierarchical_expand;

        results.push_back(res);
    }

    return results;
}

} // namespace strata::context
