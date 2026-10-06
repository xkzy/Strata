// src/generation/generation_loop_detector.cpp - Online Generation Loop Detector Implementation
#include "strata/generation/generation_loop_detector.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace strata::generation {

GenerationLoopDetector::GenerationLoopDetector(const GenerationLoopConfig& config)
    : config_(config) {}

GenerationLoopDetector::~GenerationLoopDetector() = default;

uint64_t GenerationLoopDetector::hash_ngram(size_t start_idx, size_t n) const {
    // 64-bit FNV-1a hash
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < n && (start_idx + i) < token_window_.size(); ++i) {
        int32_t tok = token_window_[start_idx + i];
        hash ^= static_cast<uint64_t>(tok & 0xFF);
        hash *= 1099511628211ULL;
        hash ^= static_cast<uint64_t>((tok >> 8) & 0xFF);
        hash *= 1099511628211ULL;
        hash ^= static_cast<uint64_t>((tok >> 16) & 0xFF);
        hash *= 1099511628211ULL;
        hash ^= static_cast<uint64_t>((tok >> 24) & 0xFF);
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool GenerationLoopDetector::is_code_or_structured_token(const std::string& text) const {
    if (text.empty()) return false;
    // Check indentation, braces, JSON formatting, programming keywords
    if (text.find("    ") != std::string::npos || text.find("\t") != std::string::npos) return true;
    for (char c : text) {
        if (c == '{' || c == '}' || c == '[' || c == ']' || c == '|' || c == ';' || c == ',') return true;
    }
    if (text == "for" || text == "while" || text == "def" || text == "function" ||
        text == "return" || text == "const" || text == "let" || text == "var" ||
        text == "import" || text == "from" || text == "class") {
        return true;
    }
    return false;
}

void GenerationLoopDetector::update_diversity_and_entropy() {
    if (token_window_.empty()) {
        current_diversity_ = 1.0;
        current_entropy_ = 1.0;
        return;
    }

    size_t total = token_window_.size();
    size_t unique = token_counts_.size();
    current_diversity_ = static_cast<double>(unique) / static_cast<double>(total);

    double entropy = 0.0;
    for (const auto& kv : token_counts_) {
        double p = static_cast<double>(kv.second) / static_cast<double>(total);
        if (p > 0.0) {
            entropy -= p * std::log2(p);
        }
    }

    // Normalized entropy: 0.0 (all identical) to 1.0 (uniform distribution)
    double max_entropy = total > 1 ? std::log2(static_cast<double>(std::min(total, unique))) : 1.0;
    current_entropy_ = max_entropy > 0.0 ? std::min(1.0, entropy / max_entropy) : 0.0;
}

GenerationLoopVerdict GenerationLoopDetector::feed_token(int32_t token_id, const std::string& token_text) {
    total_tokens_seen_++;
    GenerationLoopVerdict verdict;
    verdict.should_stop = false;
    verdict.confidence = LoopConfidence::kNormal;
    verdict.loop_type = LoopType::kNone;

    // 1. Level 0: Single Token Repetition
    if (token_id == last_token_) {
        single_token_run_++;
    } else {
        last_token_ = token_id;
        single_token_run_ = 1;
    }

    size_t max_single = config_.max_single_token_repeat;
    // Indentation and formatting characters in structured output can legitimately repeat
    if (config_.structured_output_protection &&
        (token_text == " " || token_text == "  " || token_text == "    " || token_text == "\t" || token_text == "\n")) {
        max_single *= 4;
    }

    if (single_token_run_ >= max_single) {
        verdict.should_stop = true;
        verdict.confidence = LoopConfidence::kConfirmedLoop;
        verdict.loop_type = LoopType::kSingleToken;
        verdict.period = 1;
        verdict.repetitions = single_token_run_;
        verdict.trim_token_count = single_token_run_ - 1;
        verdict.reason = "Single token repeat loop: token " + std::to_string(token_id) + " repeated " +
                         std::to_string(single_token_run_) + " times continuously.";
        verdict.finish_reason = "generation_loop";
        tokens_saved_ += single_token_run_;
        return verdict;
    }

    // Add to sliding window
    token_window_.push_back(token_id);
    text_window_.push_back(token_text);
    token_counts_[token_id]++;

    if (is_code_or_structured_token(token_text)) {
        code_like_tokens_in_window_++;
    }

    if (token_window_.size() > config_.window_size) {
        int32_t old_tok = token_window_.front();
        std::string old_text = text_window_.front();
        token_window_.pop_front();
        text_window_.pop_front();

        auto it = token_counts_.find(old_tok);
        if (it != token_counts_.end()) {
            if (it->second <= 1) token_counts_.erase(it);
            else it->second--;
        }

        if (is_code_or_structured_token(old_text) && code_like_tokens_in_window_ > 0) {
            code_like_tokens_in_window_--;
        }
    }

    update_diversity_and_entropy();
    verdict.diversity_ratio = current_diversity_;
    verdict.entropy = current_entropy_;

    size_t n_tokens = token_window_.size();
    if (n_tokens < 4) {
        return verdict;
    }

    // Determine structured protection allowance
    double code_ratio = static_cast<double>(code_like_tokens_in_window_) / static_cast<double>(n_tokens);
    bool in_structured_mode = config_.code_protection_enabled && (code_ratio > 0.35);

    // 2. Level 1 & Level 3: N-gram & Periodic Pattern Cycles (period p from 2 up to max_periodic_period)
    for (size_t p = 2; p <= std::min(config_.max_periodic_period, n_tokens / 2); ++p) {
        size_t reps = 1;
        bool match = true;

        for (size_t rep = 1; rep < n_tokens / p; ++rep) {
            for (size_t i = 0; i < p; ++i) {
                size_t curr_idx = n_tokens - 1 - i;
                size_t prev_idx = n_tokens - 1 - i - (rep * p);
                if (token_window_[curr_idx] != token_window_[prev_idx]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                reps++;
            } else {
                break;
            }
        }

        bool is_ngram = (p <= 4 || std::find(config_.ngram_sizes.begin(), config_.ngram_sizes.end(), p) != config_.ngram_sizes.end());
        size_t base_limit = is_ngram ? config_.max_ngram_repetitions : config_.max_periodic_repetitions;
        size_t threshold = in_structured_mode ? (base_limit + 2) : base_limit;

        if (reps >= threshold) {
            verdict.should_stop = true;
            verdict.confidence = LoopConfidence::kConfirmedLoop;
            verdict.loop_type = (p <= 4) ? LoopType::kNgram : LoopType::kPeriodicCycle;
            verdict.period = p;
            verdict.repetitions = reps;
            verdict.trim_token_count = p * (reps - 1);
            verdict.reason = "Periodic generation loop detected (period " + std::to_string(p) +
                             " repeated " + std::to_string(reps) + " times).";
            verdict.finish_reason = "generation_loop";
            tokens_saved_ += (p * reps);
            return verdict;
        } else if (reps >= 2) {
            verdict.confidence = LoopConfidence::kSuspicious;
            verdict.period = p;
            verdict.repetitions = reps;
        }
    }

    // 3. Level 2: Repeating Spans via rolling hashes
    for (size_t span_len : config_.ngram_sizes) {
        if (span_len >= config_.min_span_length && n_tokens >= span_len * 2) {
            uint64_t curr_hash = hash_ngram(n_tokens - span_len, span_len);
            size_t span_reps = 1;

            for (size_t k = 1; k < n_tokens / span_len; ++k) {
                uint64_t prev_hash = hash_ngram(n_tokens - (k + 1) * span_len, span_len);
                if (curr_hash == prev_hash) {
                    span_reps++;
                } else {
                    break;
                }
            }

            size_t span_thresh = in_structured_mode ? (config_.max_span_repetitions + 1)
                                                   : config_.max_span_repetitions;

            if (span_reps >= span_thresh) {
                verdict.should_stop = true;
                verdict.confidence = LoopConfidence::kConfirmedLoop;
                verdict.loop_type = LoopType::kRepeatingSpan;
                verdict.period = span_len;
                verdict.repetitions = span_reps;
                verdict.trim_token_count = span_len * (span_reps - 1);
                verdict.reason = "Repeating text span loop detected (length " + std::to_string(span_len) +
                                 " tokens repeated " + std::to_string(span_reps) + " times).";
                verdict.finish_reason = "generation_loop";
                tokens_saved_ += (span_len * span_reps);
                return verdict;
            }
        }
    }

    // 4. Diversity Collapse Check
    if (n_tokens >= 32 && !in_structured_mode) {
        if (current_diversity_ < config_.min_diversity_ratio && current_entropy_ < config_.entropy_threshold) {
            consecutive_suspicious_++;
            if (consecutive_suspicious_ >= 12) {
                verdict.should_stop = true;
                verdict.confidence = LoopConfidence::kConfirmedLoop;
                verdict.loop_type = LoopType::kDiversityCollapse;
                verdict.reason = "Diversity and entropy collapsed in generation stream.";
                verdict.finish_reason = "generation_loop";
                return verdict;
            } else {
                verdict.confidence = LoopConfidence::kProbableLoop;
            }
        } else {
            if (consecutive_suspicious_ > 0) consecutive_suspicious_--;
        }
    }

    return verdict;
}

void GenerationLoopDetector::reset() {
    token_window_.clear();
    text_window_.clear();
    token_counts_.clear();
    ngram_frequency_.clear();
    last_token_ = -1;
    single_token_run_ = 0;
    code_like_tokens_in_window_ = 0;
    current_diversity_ = 1.0;
    current_entropy_ = 1.0;
    total_tokens_seen_ = 0;
    tokens_saved_ = 0;
    consecutive_suspicious_ = 0;
}

} // namespace strata::generation
