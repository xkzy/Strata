// include/strata/logic_verification/verification_planner.hpp - Deterministic Verification Planner
//
// Routes verification requests to the cheapest and most reliable deterministic backend
// following the strict hierarchy: Cache -> Fast Numeric -> Rule -> Constraint -> Unit -> Schema -> Mathics -> Code.
#pragma once

#include "strata/logic_verification/claim_extractor.hpp"
#include "strata/logic_verification/constraint_engine.hpp"
#include "strata/logic_verification/rule_engine.hpp"
#include "strata/logic_verification/schema_verifier.hpp"
#include "strata/logic_verification/unit_verifier.hpp"
#include "strata/logic_verification/verification_cache.hpp"
#include "strata/logic_verification/verification_types.hpp"
#include "strata/math/math_runtime.hpp"

#include <memory>

namespace strata::logic {

class VerificationPlanner {
public:
    VerificationPlanner(std::shared_ptr<math::MathRuntime> math_runtime = nullptr);
    ~VerificationPlanner();

    // Route and verify a single claim
    VerificationResult verify_claim(const VerificationClaim& claim,
                                    VerificationCache* cache = nullptr) const;

    // Verify arithmetic expression with tolerance
    VerificationResult verify_arithmetic(const VerificationClaim& claim) const;

    // Verify symbolic identity / calculus claim
    VerificationResult verify_symbolic(const VerificationClaim& claim) const;

    // Verify algebraic equation solution
    VerificationResult verify_equation(const VerificationClaim& claim) const;

private:
    std::shared_ptr<math::MathRuntime> math_runtime_;
    std::unique_ptr<RuleEngine> rule_engine_;
    std::unique_ptr<ConstraintEngine> constraint_engine_;
    std::unique_ptr<UnitVerifier> unit_verifier_;
    std::unique_ptr<SchemaVerifier> schema_verifier_;
};

} // namespace strata::logic
