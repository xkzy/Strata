// include/strata/rt/recovery.hpp - what to do when the signals say something is wrong
#pragma once

#include "strata/rt/types.hpp"
#include "strata/rt/verifier.hpp"

#include <string>

namespace strata::rt {

enum class RecoveryAction {
    kContinue,             // nothing to do (UNKNOWN is allowed to reach the response)
    kRetrieveInternally,   // look for evidence, then verify again
    kRegenerate,           // discard the unreleased text and generate again with what is now known
    kHedgeClaim,           // replace the claim by an honest "could not verify" statement
    kStopSafely            // end the generation cleanly
};

inline const char* to_string(RecoveryAction a) {
    switch (a) {
        case RecoveryAction::kContinue: return "continue";
        case RecoveryAction::kRetrieveInternally: return "retrieve";
        case RecoveryAction::kRegenerate: return "regenerate";
        case RecoveryAction::kHedgeClaim: return "hedge";
        case RecoveryAction::kStopSafely: return "stop_safely";
    }
    return "continue";
}

struct RecoveryPolicy {
    int max_regenerations = 2;     // factual regenerations per request
    int max_loop_recoveries = 3;   // loop recoveries per request
    bool strict = false;           // STRICT: an unverifiable high-risk claim is never delivered as if it were fine
};

struct RecoveryInput {
    RuntimeSignals signals;
    const ClaimVerdict* verdict = nullptr;   // the claim being judged (null when only the loop signal changed)
    bool retrieval_tried = false;
    bool claim_retractable = true;           // the claim's value has not reached the caller yet
    int regenerations_used = 0;
    int loop_recoveries_used = 0;
};

struct RecoveryDecision {
    RecoveryAction action = RecoveryAction::kContinue;
    std::string reason;
    // Only a generation loop may raise the temperature. This is false for every factual decision.
    bool adjust_temperature = false;
};

RecoveryDecision decide_recovery(const RecoveryInput& in, const RecoveryPolicy& policy);

} // namespace strata::rt
