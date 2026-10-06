// src/guard/loop_detector.cpp - Loop Detector Implementation
#include "strata/guard/loop_detector.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace strata::guard {

LoopDetector::LoopDetector(const ExecutionBudget& budget)
    : budget_(budget) {}

LoopDetector::~LoopDetector() = default;

std::string LoopDetector::normalize_text(const std::string& input) {
    std::string result;
    result.reserve(input.size());

    bool in_whitespace = false;
    for (char c : input) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!in_whitespace) {
                result.push_back(' ');
                in_whitespace = true;
            }
        } else {
            // Lowercase standard alphanumeric characters
            result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            in_whitespace = false;
        }
    }

    // Trim trailing space
    while (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    // Trim leading space
    size_t start = 0;
    while (start < result.size() && result[start] == ' ') {
        start++;
    }
    return start > 0 ? result.substr(start) : result;
}

std::string LoopDetector::canonicalize_payload(const std::string& target_name, const std::string& payload) {
    std::string norm_target = normalize_text(target_name);
    std::string norm_payload = normalize_text(payload);

    // Normalize common path separators
    for (char& c : norm_payload) {
        if (c == '\\') c = '/';
    }

    return norm_target + "::" + norm_payload;
}

uint64_t LoopDetector::compute_hash(const std::string& text) {
    // 64-bit FNV-1a hash
    uint64_t hash = 14695981039346656037ULL;
    for (char c : text) {
        hash ^= static_cast<uint8_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t LoopDetector::compute_state_fingerprint(const std::vector<std::string>& state_elements) {
    std::string combined;
    for (const auto& elem : state_elements) {
        combined += normalize_text(elem) + "|";
    }
    return compute_hash(combined);
}

ErrorCategory LoopDetector::categorize_error(const std::string& err_msg) {
    std::string lower = normalize_text(err_msg);
    if (lower.find("timeout") != std::string::npos || lower.find("timed out") != std::string::npos ||
        lower.find("connection reset") != std::string::npos || lower.find("econnreset") != std::string::npos) {
        return ErrorCategory::kTransientNetwork;
    }
    if (lower.find("rate limit") != std::string::npos || lower.find("429") != std::string::npos ||
        lower.find("too many requests") != std::string::npos) {
        return ErrorCategory::kRateLimited;
    }
    if (lower.find("syntax") != std::string::npos || lower.find("parse error") != std::string::npos ||
        lower.find("invalid json") != std::string::npos || lower.find("invalid argument") != std::string::npos ||
        lower.find("unexpected token") != std::string::npos || lower.find("unknown parameter") != std::string::npos) {
        return ErrorCategory::kSyntaxOrSchema;
    }
    if (lower.find("not found") != std::string::npos || lower.find("no such file") != std::string::npos ||
        lower.find("enoent") != std::string::npos || lower.find("404") != std::string::npos) {
        return ErrorCategory::kNotFoundOrPath;
    }
    if (lower.find("permission") != std::string::npos || lower.find("access denied") != std::string::npos ||
        lower.find("eacces") != std::string::npos || lower.find("unauthorized") != std::string::npos) {
        return ErrorCategory::kPermissionDenied;
    }
    if (lower.find("panic") != std::string::npos || lower.find("segmentation fault") != std::string::npos ||
        lower.find("abort") != std::string::npos) {
        return ErrorCategory::kExecutionPanic;
    }
    return ErrorCategory::kUnknown;
}

GuardVerdict LoopDetector::evaluate_action(const ActionRecord& action) {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.total_evaluations++;

    GuardVerdict verdict;
    verdict.current_step = static_cast<int64_t>(action_history_.size()) + 1;
    verdict.no_progress_streak = consecutive_no_progress_;
    verdict.progress_score = current_progress_score_;
    verdict.state = EscalationState::kNormal;
    verdict.allowed = true;

    // 1. Check Tool Failure Loop (prioritized for structured observation feedback to agent)
    int64_t fail_count = 0;
    std::string failure_obs;
    if (check_failure_loop(action, fail_count, failure_obs)) {
        verdict.state = EscalationState::kBlocked;
        verdict.allowed = false;
        verdict.reason = "Repeated tool failure loop: failed " + std::to_string(fail_count) + " times identically.";
        verdict.suggested_remediation = "Select an alternative tool, parameters, or approach.";
        verdict.structured_observation = failure_obs;
        return verdict;
    }

    // 2. Check Pattern Cycles (e.g. A->B->A->B or A->B->C->A->B->C)
    int period = 0, repetitions = 0;
    if (check_pattern_loop(action, period, repetitions)) {
        stats_.pattern_cycles_detected++;
        if (repetitions >= budget_.max_pattern_repetitions) {
            verdict.state = EscalationState::kBlocked;
            verdict.allowed = false;
            verdict.reason = "Cyclic loop pattern detected (period " + std::to_string(period) +
                             " repeated " + std::to_string(repetitions) + " times).";
            verdict.suggested_remediation = "Break cyclic dependency with a different problem solving strategy.";
            stats_.blocked_count++;
            return verdict;
        } else if (repetitions >= 2) {
            verdict.state = EscalationState::kThrottled;
            verdict.is_throttled = true;
            verdict.throttle_delay_ms = 100.0;
            verdict.reason = "Potential alternating loop pattern identified.";
            stats_.throttled_count++;
        }
    }

    // 3. Check Exact Duplicate Operations
    int64_t repeat_count = 0;
    if (check_exact_duplicate(action, repeat_count)) {
        stats_.exact_duplicates_detected++;
        if (repeat_count >= budget_.max_identical_tool_calls) {
            verdict.state = EscalationState::kBlocked;
            verdict.allowed = false;
            verdict.reason = "Identical operation repeated " + std::to_string(repeat_count) + " times without progress.";
            verdict.suggested_remediation = "Change tool arguments or choose an alternative action.";
            stats_.blocked_count++;
            return verdict;
        } else if (repeat_count >= 2) {
            verdict.state = EscalationState::kSuspected;
            verdict.is_throttled = true;
            verdict.throttle_delay_ms = 50.0 * repeat_count;
            verdict.reason = "Duplicate operation detected (" + std::to_string(repeat_count) + "x).";
            stats_.suspected_count++;
        }
    }

    // 4. Check Retrieval Redundancy Loop
    if (action.kind == ActionKind::kRetrieval) {
        int64_t ret_count = 0;
        if (check_retrieval_loop(action, ret_count)) {
            stats_.retrieval_loops_prevented++;
            if (ret_count >= budget_.max_identical_retrievals) {
                verdict.state = EscalationState::kBlocked;
                verdict.allowed = false;
                verdict.reason = "Redundant retrieval loop: identical query executed " +
                                 std::to_string(ret_count) + " times with unchanged result.";
                verdict.suggested_remediation = "Use existing retrieved context without querying the same query again.";
                stats_.blocked_count++;
                return verdict;
            }
        }
    }

    // 5. Check No-Progress Budget
    if (consecutive_no_progress_ >= budget_.max_no_progress_steps) {
        verdict.state = EscalationState::kCancelled;
        verdict.allowed = false;
        verdict.reason = "Runaway execution cancelled: " + std::to_string(consecutive_no_progress_) +
                         " consecutive steps without state progress.";
        verdict.suggested_remediation = "Re-evaluate plan and produce meaningful state transitions.";
        stats_.cancelled_count++;
        return verdict;
    }

    if (verdict.state == EscalationState::kNormal) {
        stats_.normal_count++;
    }
    return verdict;
}

void LoopDetector::record_action_outcome(const ActionRecord& action, bool state_changed, double progress_delta) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Update histories
    action_history_.push_back(action);
    hash_history_.push_back(action.canonical_hash);
    state_history_.push_back(action.state_fingerprint);

    while (action_history_.size() > kMaxHistoryWindow) {
        action_history_.pop_front();
        hash_history_.pop_front();
        state_history_.pop_front();
    }

    // Track failures
    if (action.is_error) {
        failure_counts_[action.canonical_hash]++;
        last_failure_message_[action.canonical_hash] = action.error_message;
    } else {
        failure_counts_[action.canonical_hash] = 0;
    }

    // Track retrievals
    if (action.kind == ActionKind::kRetrieval) {
        retrieval_query_counts_[action.canonical_hash]++;
        retrieval_last_results_[action.canonical_hash] = action.result_summary;
    }

    // Update progress tracking
    if (state_changed || progress_delta > 0.0) {
        consecutive_no_progress_ = 0;
        current_progress_score_ = std::min(1.0, current_progress_score_ + std::max(0.1, progress_delta));
    } else {
        consecutive_no_progress_++;
        current_progress_score_ = std::max(0.0, current_progress_score_ - 0.2);
    }
    stats_.avg_progress_score = current_progress_score_;
}

bool LoopDetector::check_exact_duplicate(const ActionRecord& action, int64_t& repeat_count) {
    repeat_count = 1;
    if (hash_history_.empty()) return false;

    // Scan recent history in reverse
    for (auto it = hash_history_.rbegin(); it != hash_history_.rend(); ++it) {
        if (*it == action.canonical_hash) {
            repeat_count++;
        } else {
            break;
        }
    }
    return repeat_count > 1;
}

bool LoopDetector::check_pattern_loop(const ActionRecord& action, int& out_period, int& out_repetitions) {
    if (hash_history_.size() + 1 < 4) return false;

    // Build temporary sequence including the candidate action
    std::vector<uint64_t> seq(hash_history_.begin(), hash_history_.end());
    seq.push_back(action.canonical_hash);

    size_t n = seq.size();
    for (size_t p = 1; p <= std::min<size_t>(6, n / 2); ++p) {
        int reps = 1;
        bool match = true;

        for (size_t rep = 1; rep < n / p; ++rep) {
            for (size_t i = 0; i < p; ++i) {
                size_t curr_idx = n - 1 - i;
                size_t prev_idx = n - 1 - i - (rep * p);
                if (seq[curr_idx] != seq[prev_idx]) {
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

        if (reps >= 2) {
            out_period = static_cast<int>(p);
            out_repetitions = reps;
            return true;
        }
    }
    return false;
}

bool LoopDetector::check_failure_loop(const ActionRecord& action, int64_t& fail_count, std::string& out_observation) {
    auto it = failure_counts_.find(action.canonical_hash);
    if (it != failure_counts_.end() && it->second >= budget_.max_identical_failures) {
        fail_count = it->second;
        std::ostringstream oss;
        oss << "Repeated failure detected.\n"
            << "Tool: " << action.target_name << "\n"
            << "Attempts: " << fail_count << "\n"
            << "Error: " << last_failure_message_[action.canonical_hash] << "\n"
            << "Further identical execution suppressed. Please select an alternative strategy.";
        out_observation = oss.str();
        stats_.failure_loops_suppressed++;
        return true;
    }
    return false;
}

bool LoopDetector::check_retrieval_loop(const ActionRecord& action, int64_t& retrieval_repeat_count) {
    auto it = retrieval_query_counts_.find(action.canonical_hash);
    if (it != retrieval_query_counts_.end()) {
        retrieval_repeat_count = it->second;
        return retrieval_repeat_count >= budget_.max_identical_retrievals;
    }
    return false;
}

void LoopDetector::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    action_history_.clear();
    hash_history_.clear();
    state_history_.clear();
    failure_counts_.clear();
    last_failure_message_.clear();
    retrieval_query_counts_.clear();
    retrieval_last_results_.clear();
    consecutive_no_progress_ = 0;
    current_progress_score_ = 1.0;
}

void LoopDetector::set_budget(const ExecutionBudget& budget) {
    std::lock_guard<std::mutex> lock(mutex_);
    budget_ = budget;
}

} // namespace strata::guard
