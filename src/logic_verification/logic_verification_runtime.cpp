// src/logic_verification/logic_verification_runtime.cpp - Logic Verification Runtime Implementation
#include "strata/logic_verification/logic_verification_runtime.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace strata::logic {

LogicVerificationRuntime::LogicVerificationRuntime(std::shared_ptr<context::VirtualContextManager> vctx,
                                                   std::shared_ptr<math::MathRuntime> math_runtime)
    : vctx_(std::move(vctx)),
      math_runtime_(std::move(math_runtime)),
      extractor_(std::make_unique<ClaimExtractor>()),
      planner_(std::make_unique<VerificationPlanner>(math_runtime_)),
      cache_(std::make_unique<VerificationCache>(10000)) {}

LogicVerificationRuntime::~LogicVerificationRuntime() = default;

VerificationResult LogicVerificationRuntime::verify(const VerificationClaim& claim) {
    auto start = std::chrono::steady_clock::now();
    VerificationResult res = planner_->verify_claim(claim, cache_.get());
    auto end = std::chrono::steady_clock::now();
    res.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Update metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.total_verifications++;
        if (res.status == VerificationStatus::kPass) metrics_.pass_count++;
        else if (res.status == VerificationStatus::kFail) metrics_.fail_count++;
        else if (res.status == VerificationStatus::kPartial) metrics_.partial_count++;
        else metrics_.unknown_count++;

        if (res.cache_hit) metrics_.cache_hits++;
        else metrics_.cache_misses++;

        metrics_.total_execution_time_ms += res.execution_time_ms;
        metrics_.backend_distribution[res.backend_used]++;
    }

    // External State Store Integration (Virtual Context)
    if (vctx_ && (res.status == VerificationStatus::kPass || res.status == VerificationStatus::kFail)) {
        std::string store_content = "[Verification: " + std::string(verification_status_to_string(res.status)) +
                                    " | Backend: " + res.backend_used + " | Claim: " + claim.expression +
                                    " | Evidence: " + res.evidence + "]";
        vctx_->append_tool_result("verifier", claim.expression, store_content,
                                 res.status == VerificationStatus::kPass ? 0 : 1, 30);
    }

    return res;
}

std::vector<VerificationResult> LogicVerificationRuntime::verify_text(const std::string& text,
                                                                      const std::string& tenant_id,
                                                                      const std::string& session_id) {
    auto claims = extractor_->extract_claims(text, tenant_id, session_id);
    std::vector<VerificationResult> results;
    results.reserve(claims.size());

    for (const auto& claim : claims) {
        results.push_back(verify(claim));
    }
    return results;
}

std::string LogicVerificationRuntime::intercept_and_verify(const std::string& generation_text,
                                                           std::vector<VerificationResult>& out_results,
                                                           const std::string& tenant_id,
                                                           const std::string& session_id) {
    out_results = verify_text(generation_text, tenant_id, session_id);
    if (out_results.empty()) {
        return generation_text;
    }

    std::string augmented = generation_text;
    for (const auto& res : out_results) {
        if (!res.compact_observation.empty()) {
            augmented += "\n[" + res.compact_observation + "]";
        }
    }
    return augmented;
}

VerificationMetrics LogicVerificationRuntime::get_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void LogicVerificationRuntime::reset_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_ = VerificationMetrics();
}

std::string LogicVerificationRuntime::print_diagnostics() const {
    auto m = get_metrics();
    std::ostringstream ss;
    ss << "=== Strata Deterministic Logic Verifier Diagnostics ===\n";
    ss << "Total Verifications: " << m.total_verifications << "\n";
    ss << "Pass: " << m.pass_count << " (" << std::fixed << std::setprecision(1) << (m.pass_rate() * 100.0) << "%)\n";
    ss << "Fail: " << m.fail_count << "\n";
    ss << "Unknown: " << m.unknown_count << "\n";
    ss << "Partial: " << m.partial_count << "\n";
    ss << "Cache Hits: " << m.cache_hits << " (Hit Rate: " << std::fixed << std::setprecision(1) << (m.cache_hit_rate() * 100.0) << "%)\n";
    ss << "Avg Latency: " << std::fixed << std::setprecision(2) << m.avg_latency_ms() << " ms\n";
    ss << "Cached Claims: " << cache_->size() << " / " << cache_->max_entries() << "\n";
    ss << "Backends Active:\n";
    for (const auto& pair : m.backend_distribution) {
        ss << "  - " << pair.first << ": " << pair.second << " calls\n";
    }
    return ss.str();
}

} // namespace strata::logic
