// include/strata/logic_verification/logic_verification_runtime.hpp - Logic Verification Runtime
//
// Top-level deterministic verification runtime coordinating claim extraction,
// planner routing, result caching, virtual context audit trails, and metrics.
#pragma once

#include "strata/context/virtual_context.hpp"
#include "strata/logic_verification/claim_extractor.hpp"
#include "strata/logic_verification/verification_cache.hpp"
#include "strata/logic_verification/verification_planner.hpp"
#include "strata/logic_verification/verification_types.hpp"
#include "strata/math/math_runtime.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::logic {

class LogicVerificationRuntime {
public:
    explicit LogicVerificationRuntime(std::shared_ptr<context::VirtualContextManager> vctx = nullptr,
                                      std::shared_ptr<math::MathRuntime> math_runtime = nullptr);
    ~LogicVerificationRuntime();

    // Verify a single structured claim
    VerificationResult verify(const VerificationClaim& claim);

    // Extract claims from text and verify all identifiable claims
    std::vector<VerificationResult> verify_text(const std::string& text,
                                                const std::string& tenant_id = "default",
                                                const std::string& session_id = "default");

    // Intercept claims in generation text and augment with deterministic verification observations
    std::string intercept_and_verify(const std::string& generation_text,
                                     std::vector<VerificationResult>& out_results,
                                     const std::string& tenant_id = "default",
                                     const std::string& session_id = "default");

    // Metrics & Diagnostics
    VerificationMetrics get_metrics() const;
    void reset_metrics();
    std::string print_diagnostics() const;

    // Subsystem accessors
    const ClaimExtractor& extractor() const { return *extractor_; }
    const VerificationPlanner& planner() const { return *planner_; }
    VerificationCache& cache() { return *cache_; }

private:
    std::shared_ptr<context::VirtualContextManager> vctx_;
    std::shared_ptr<math::MathRuntime> math_runtime_;
    std::unique_ptr<ClaimExtractor> extractor_;
    std::unique_ptr<VerificationPlanner> planner_;
    std::unique_ptr<VerificationCache> cache_;

    mutable std::mutex metrics_mutex_;
    VerificationMetrics metrics_;
};

} // namespace strata::logic
