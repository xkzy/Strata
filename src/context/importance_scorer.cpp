// src/context/importance_scorer.cpp - Importance Scorer Implementation
#include "strata/context/importance_scorer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <unordered_set>

namespace strata::context {

namespace {

std::unordered_set<std::string> extract_words(const std::string& text) {
    std::unordered_set<std::string> words;
    std::string word;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
            word += std::tolower(static_cast<unsigned char>(c));
        } else if (!word.empty()) {
            if (word.size() > 2) {
                words.insert(word);
            }
            word.clear();
        }
    }
    if (!word.empty() && word.size() > 2) {
        words.insert(word);
    }
    return words;
}

} // anonymous namespace

CompositeImportanceScorer::CompositeImportanceScorer(const ScoringWeights& weights)
    : weights_(weights) {}

double CompositeImportanceScorer::compute_type_priority(ContextItemType type) const {
    switch (type) {
        case ContextItemType::kSystemPrompt:
            return 1.0;
        case ContextItemType::kWorkingMemory:
            return 0.95;
        case ContextItemType::kUserMessage:
            return 0.85;
        case ContextItemType::kAssistantMessage:
            return 0.75;
        case ContextItemType::kErrorLog:
        case ContextItemType::kTestResult:
            return 0.70;
        case ContextItemType::kCodeChange:
            return 0.65;
        case ContextItemType::kToolCall:
        case ContextItemType::kToolOutput:
            return 0.50;
        case ContextItemType::kFileContent:
            return 0.45;
        case ContextItemType::kCommandOutput:
            return 0.40;
        case ContextItemType::kSummaryL1:
        case ContextItemType::kSummaryL2:
        case ContextItemType::kSummaryL3:
            return 0.60;
    }
    return 0.5;
}

double CompositeImportanceScorer::compute_query_relevance(const std::string& text,
                                                          const std::string& query) const {
    if (query.empty() || text.empty()) return 0.5;

    auto query_words = extract_words(query);
    if (query_words.empty()) return 0.5;

    auto text_words = extract_words(text);
    size_t matches = 0;
    for (const auto& w : query_words) {
        if (text_words.find(w) != text_words.end()) {
            matches++;
        }
    }
    return static_cast<double>(matches) / static_cast<double>(query_words.size());
}

double CompositeImportanceScorer::score(const ContextItem& item, const std::string& current_query,
                                        double current_time_sec) const {
    // 1. Recency Decay
    double dt = std::max(0.0, current_time_sec - item.timestamp());
    double recency_score = std::exp(-dt / weights_.half_life_seconds);

    // 2. Frequency of Reference
    double freq_score = std::min(1.0, std::log1p(static_cast<double>(item.access_count())) / 3.0);

    // 3. Query Relevance
    double relevance_score = compute_query_relevance(item.raw_content(), current_query);

    // 4. Dependency / Reference links
    double dep_score = std::min(1.0, static_cast<double>(item.references().size()) * 0.25);

    // 5. Type Priority
    double type_score = compute_type_priority(item.type());

    double total = (weights_.recency_weight * recency_score) +
                   (weights_.frequency_weight * freq_score) +
                   (weights_.relevance_weight * relevance_score) +
                   (weights_.dependency_weight * dep_score) +
                   (weights_.type_priority_weight * type_score);

    return std::clamp(total, 0.0, 1.0);
}

CodeAwareImportanceScorer::CodeAwareImportanceScorer(const ScoringWeights& weights)
    : base_scorer_(weights) {}

double CodeAwareImportanceScorer::score(const ContextItem& item, const std::string& current_query,
                                        double current_time_sec) const {
    double base = base_scorer_.score(item, current_query, current_time_sec);

    // Boost if item matches active coding files
    if (!item.metadata().filename.empty()) {
        for (const auto& f : active_files_) {
            if (item.metadata().filename == f || item.metadata().filename.find(f) != std::string::npos) {
                base += 0.25;
                break;
            }
        }
    }

    // Boost if item references active symbols
    if (!item.metadata().symbol_name.empty()) {
        for (const auto& s : active_symbols_) {
            if (item.metadata().symbol_name == s) {
                base += 0.20;
                break;
            }
        }
    }

    // Boost errors and failed test results
    if (item.type() == ContextItemType::kErrorLog ||
        (item.type() == ContextItemType::kTestResult && item.metadata().exit_code != 0)) {
        base += 0.15;
    }

    return std::clamp(base, 0.0, 1.0);
}

} // namespace strata::context
