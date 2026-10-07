// src/rt/recovery.cpp
#include "strata/rt/recovery.hpp"

namespace strata::rt {

RecoveryDecision decide_recovery(const RecoveryInput& in, const RecoveryPolicy& policy) {
    RecoveryDecision d;

    // 1. generation loops: their own signal, their own remedy (dynamic temperature), independent of verification
    if (in.signals.generation_loop_state == LoopState::kConfirmedLoop) {
        if (in.loop_recoveries_used < policy.max_loop_recoveries) {
            d.action = RecoveryAction::kRegenerate;
            d.reason = "confirmed generation loop: resume from the last good text at a higher temperature";
            d.adjust_temperature = true;
        } else {
            d.action = RecoveryAction::kStopSafely;
            d.reason = "generation loop persisted through every recovery attempt";
        }
        return d;
    }
    if (in.signals.generation_loop_state == LoopState::kProbableLoop || in.signals.generation_loop_state == LoopState::kSuspicious) {
        d.action = RecoveryAction::kContinue;       // the live temperature controller handles it token by token
        d.reason = "possible loop: dynamic temperature";
        d.adjust_temperature = true;
        if (!in.verdict) return d;
    }

    // 2. claims: never touch the temperature for a factual problem
    if (!in.verdict) return d;
    const ClaimVerdict& v = *in.verdict;
    d.adjust_temperature = false;
    switch (v.state) {
        case VerificationState::kVerified:
        case VerificationState::kSupported:
        case VerificationState::kPartiallySupported:
            d.action = RecoveryAction::kContinue;
            d.reason = "claim holds";
            return d;
        case VerificationState::kContradicted:
            if (in.claim_retractable && in.regenerations_used < policy.max_regenerations) {
                d.action = RecoveryAction::kRegenerate;
                d.reason = "contradicted by " + v.deciding_verifier + ": " + v.explanation;
            } else if (in.claim_retractable) {
                d.action = RecoveryAction::kHedgeClaim;
                d.reason = "contradicted and regeneration budget is spent";
            } else {
                d.action = RecoveryAction::kContinue;   // already delivered: nothing can retract it (recorded as a late contradiction)
                d.reason = "contradicted after delivery";
            }
            return d;
        case VerificationState::kUnsupported:
        case VerificationState::kUnknown:
            if (v.verifier_failed || v.conflicting_evidence) {
                // a failed verifier is not evidence of falsehood
                d.action = policy.strict && in.claim_retractable ? RecoveryAction::kHedgeClaim : RecoveryAction::kContinue;
                d.reason = v.verifier_failed ? "verifier unavailable: UNKNOWN" : "evidence conflicts: UNKNOWN";
                return d;
            }
            if (!in.retrieval_tried && v.claim.risk == ClaimRisk::kHigh) {
                d.action = RecoveryAction::kRetrieveInternally;
                d.reason = "high-risk claim without support: look for evidence";
                return d;
            }
            if (policy.strict && v.claim.risk == ClaimRisk::kHigh && in.claim_retractable) {
                d.action = RecoveryAction::kHedgeClaim;
                d.reason = "strict mode: unverified high-risk claim";
                return d;
            }
            d.action = RecoveryAction::kContinue;
            d.reason = "UNKNOWN is allowed to reach the response";
            return d;
    }
    return d;
}

} // namespace strata::rt
