// include/strata/rt/types.hpp - shared vocabulary of the transparent inference runtime
//
// Everything in strata::rt is internal to the inference runtime. The public surface stays "messages in, text out";
// none of these types is a tool or an API the caller has to know about.
#pragma once

#include "strata/context/multi_tenant_context.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::rt {

// ---- the four independent signals (never merged into one score) ----
enum class LoopState { kNormal, kSuspicious, kProbableLoop, kConfirmedLoop };

enum class VerificationState {
    kVerified,            // proven by a deterministic verifier or authoritative exact state
    kSupported,           // evidence agrees
    kPartiallySupported,  // some components confirmed
    kContradicted,        // evidence or a deterministic computation disagrees
    kUnsupported,         // evidence exists but does not back the claim
    kUnknown              // nothing could be established (including: the verifier failed)
};

enum class EvidenceState { kNone, kAvailable, kConflicting };
enum class ResourceState { kNormal, kConstrained, kExhausted };

inline const char* to_string(LoopState s) {
    switch (s) {
        case LoopState::kNormal: return "NORMAL";
        case LoopState::kSuspicious: return "SUSPICIOUS";
        case LoopState::kProbableLoop: return "PROBABLE_LOOP";
        case LoopState::kConfirmedLoop: return "CONFIRMED_LOOP";
    }
    return "NORMAL";
}
inline const char* to_string(VerificationState s) {
    switch (s) {
        case VerificationState::kVerified: return "VERIFIED";
        case VerificationState::kSupported: return "SUPPORTED";
        case VerificationState::kPartiallySupported: return "PARTIALLY_SUPPORTED";
        case VerificationState::kContradicted: return "CONTRADICTED";
        case VerificationState::kUnsupported: return "UNSUPPORTED";
        case VerificationState::kUnknown: return "UNKNOWN";
    }
    return "UNKNOWN";
}
inline const char* to_string(EvidenceState s) {
    switch (s) {
        case EvidenceState::kNone: return "NONE";
        case EvidenceState::kAvailable: return "AVAILABLE";
        case EvidenceState::kConflicting: return "CONFLICTING";
    }
    return "NONE";
}
inline const char* to_string(ResourceState s) {
    switch (s) {
        case ResourceState::kNormal: return "NORMAL";
        case ResourceState::kConstrained: return "CONSTRAINED";
        case ResourceState::kExhausted: return "EXHAUSTED";
    }
    return "NORMAL";
}

struct RuntimeSignals {
    LoopState generation_loop_state = LoopState::kNormal;
    VerificationState verification_state = VerificationState::kUnknown;
    EvidenceState evidence_state = EvidenceState::kNone;
    ResourceState resource_state = ResourceState::kNormal;
};

// ---- request identity: every mutable state is scoped by all six ids ----
struct RequestScope {
    context::SecurityScope security;   // tenant / user / workspace(project) / agent / session
    std::string request_id;
    std::string key() const { return security.to_string() + "/" + request_id; }
};

// ---- what the caller sends: a normal chat/completion request ----
struct Message {
    std::string role;      // system | user | assistant | tool
    std::string content;
    std::string name;      // optional (tool name)
};

struct Sampling {
    double temperature = 0.7;   // the caller's baseline; the runtime only moves it for generation loops
    double top_p = 1.0;
    int top_k = 0;
    int64_t seed = -1;
    int max_tokens = 1024;
    std::vector<std::string> stop;
};

enum class FinishReason { kStop, kLength, kCancelled };
inline const char* to_string(FinishReason r) {
    switch (r) {
        case FinishReason::kStop: return "stop";
        case FinishReason::kLength: return "length";
        case FinishReason::kCancelled: return "cancelled";
    }
    return "stop";
}

// ---- stable 128-bit hashing for dependency-aware keys ----
std::string hash128(const std::string& s);
// order-sensitive combination of several parts (length-prefixed so ("ab","c") != ("a","bc"))
std::string hash_parts(const std::vector<std::string>& parts);

} // namespace strata::rt
