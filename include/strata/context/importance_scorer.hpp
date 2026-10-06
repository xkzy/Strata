// include/strata/context/importance_scorer.hpp - Extensible Importance Scoring
//
// Computes multi-signal importance metrics (recency, reference count, query relevance,
// task state, code dependencies) to guide compaction and context paging.
#pragma once

#include "strata/context/context_item.hpp"

#include <memory>
#include <string>
#include <vector>

namespace strata::context {

struct ScoringWeights {
    double recency_weight = 0.25;
    double frequency_weight = 0.20;
    double relevance_weight = 0.30;
    double dependency_weight = 0.15;
    double type_priority_weight = 0.10;
    double half_life_seconds = 3600.0; // 1 hour half-life for recency decay
};

class ImportanceScorer {
public:
    virtual ~ImportanceScorer() = default;

    virtual double score(const ContextItem& item, const std::string& current_query,
                         double current_time_sec) const = 0;
};

class CompositeImportanceScorer : public ImportanceScorer {
public:
    explicit CompositeImportanceScorer(const ScoringWeights& weights = ScoringWeights());

    double score(const ContextItem& item, const std::string& current_query,
                 double current_time_sec) const override;

    const ScoringWeights& weights() const { return weights_; }
    void set_weights(const ScoringWeights& w) { weights_ = w; }

private:
    ScoringWeights weights_;
    double compute_type_priority(ContextItemType type) const;
    double compute_query_relevance(const std::string& text, const std::string& query) const;
};

class CodeAwareImportanceScorer : public ImportanceScorer {
public:
    explicit CodeAwareImportanceScorer(const ScoringWeights& weights = ScoringWeights());

    double score(const ContextItem& item, const std::string& current_query,
                 double current_time_sec) const override;

    void set_active_files(const std::vector<std::string>& files) { active_files_ = files; }
    void set_active_symbols(const std::vector<std::string>& symbols) { active_symbols_ = symbols; }

private:
    CompositeImportanceScorer base_scorer_;
    std::vector<std::string> active_files_;
    std::vector<std::string> active_symbols_;
};

} // namespace strata::context
