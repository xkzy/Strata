// include/strata/generation/generation_loop_detector.hpp - Online Generation Loop Detector
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::generation {

enum class LoopConfidence {
    kNormal,
    kSuspicious,
    kProbableLoop,
    kConfirmedLoop
};

enum class LoopType {
    kNone,
    kSingleToken,
    kNgram,
    kRepeatingSpan,
    kPeriodicCycle,
    kDiversityCollapse
};

struct GenerationLoopConfig {
    size_t max_single_token_repeat = 32;
    std::vector<size_t> ngram_sizes = {2, 3, 4, 8, 16};
    size_t max_ngram_repetitions = 5;
    size_t max_span_repetitions = 4;
    size_t min_span_length = 12;
    size_t window_size = 128;
    double entropy_threshold = 0.20;
    double min_diversity_ratio = 0.12;
    bool code_protection_enabled = true;
    bool structured_output_protection = true;
    size_t max_periodic_period = 32;
    size_t max_periodic_repetitions = 4;
};

struct GenerationLoopVerdict {
    bool should_stop = false;
    LoopConfidence confidence = LoopConfidence::kNormal;
    LoopType loop_type = LoopType::kNone;
    size_t period = 0;
    size_t repetitions = 0;
    double diversity_ratio = 1.0;
    double entropy = 1.0;
    size_t trim_token_count = 0;
    std::string reason;
    std::string finish_reason;
};

class GenerationLoopDetector {
public:
    explicit GenerationLoopDetector(const GenerationLoopConfig& config = GenerationLoopConfig());
    ~GenerationLoopDetector();

    // Online per-token evaluation: call for every token generated
    GenerationLoopVerdict feed_token(int32_t token_id, const std::string& token_text = "");

    // Reset detector state for a new request
    void reset();

    // Diagnostics & telemetry
    size_t total_tokens_seen() const { return total_tokens_seen_; }
    size_t tokens_saved() const { return tokens_saved_; }
    double current_diversity() const { return current_diversity_; }
    double current_entropy() const { return current_entropy_; }
    const GenerationLoopConfig& config() const { return config_; }
    void set_config(const GenerationLoopConfig& config) { config_ = config; }

private:
    uint64_t hash_ngram(size_t start_idx, size_t n) const;
    void update_diversity_and_entropy();
    bool is_code_or_structured_token(const std::string& text) const;

    GenerationLoopConfig config_;
    std::deque<int32_t> token_window_;
    std::deque<std::string> text_window_;
    std::unordered_map<int32_t, size_t> token_counts_;
    
    // Single-token repeat tracker
    int32_t last_token_ = -1;
    size_t single_token_run_ = 0;

    // N-gram and periodic tracking
    std::unordered_map<uint64_t, size_t> ngram_frequency_;

    // Structured & code awareness
    size_t code_like_tokens_in_window_ = 0;

    // Running metrics
    double current_diversity_ = 1.0;
    double current_entropy_ = 1.0;
    size_t total_tokens_seen_ = 0;
    size_t tokens_saved_ = 0;
    size_t consecutive_suspicious_ = 0;
};

} // namespace strata::generation
