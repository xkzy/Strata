// include/strata/rt/claim.hpp - internal claim representation and the detector that finds claims in generated text
#pragma once

#include "strata/logic_verification/verification_types.hpp"
#include "strata/rt/types.hpp"

#include <string>
#include <vector>

namespace strata::rt {

enum class ClaimKind {
    kArithmetic,     // 37 * 19 = 703
    kEquation,       // x = 5 solves 2x + 5 = 15
    kSymbolic,       // (x+1)^2 = x^2 + 2x + 1
    kUnit,           // 10 m / 2 s = 5 m/s
    kNumber,         // an exact figure about something: "the server uses 8080 workers"
    kConfigValue,    // "the port is 8080", "timeout = 30"
    kDate,           // 2024-03-05, March 5 2024
    kFilePath,       // /etc/nginx/nginx.conf
    kApiSignature,   // `connect(host, port)`
    kIdentifier,     // `ClaimDetector`, `read_file`
    kCitation,       // [3], RFC 7231, et al., URLs
    kSpecification,  // supports up to 64 GB
    kCodeBehavior,   // a code block claimed to do something
    kSchema,         // JSON / schema
    kProposition     // logical statements
};

enum class ClaimRisk { kLow, kMedium, kHigh };

inline const char* to_string(ClaimKind k) {
    switch (k) {
        case ClaimKind::kArithmetic: return "arithmetic";
        case ClaimKind::kEquation: return "equation";
        case ClaimKind::kSymbolic: return "symbolic";
        case ClaimKind::kUnit: return "unit";
        case ClaimKind::kNumber: return "number";
        case ClaimKind::kConfigValue: return "config_value";
        case ClaimKind::kDate: return "date";
        case ClaimKind::kFilePath: return "file_path";
        case ClaimKind::kApiSignature: return "api_signature";
        case ClaimKind::kIdentifier: return "identifier";
        case ClaimKind::kCitation: return "citation";
        case ClaimKind::kSpecification: return "specification";
        case ClaimKind::kCodeBehavior: return "code_behavior";
        case ClaimKind::kSchema: return "schema";
        case ClaimKind::kProposition: return "proposition";
    }
    return "number";
}
inline const char* to_string(ClaimRisk r) {
    return r == ClaimRisk::kHigh ? "high" : r == ClaimRisk::kMedium ? "medium" : "low";
}

struct Claim {
    std::string id;
    ClaimKind kind = ClaimKind::kNumber;
    ClaimRisk risk = ClaimRisk::kMedium;
    std::string text;         // the sentence (or span) the claim lives in
    size_t begin = 0;         // byte offsets into the generated response
    size_t end = 0;
    std::string subject;      // what the value is about: "port", "timeout", "37*19"
    std::string value;        // the claimed value: "8080", "713"
    std::string normalized;   // canonical form used for cache keys and matching
    // set for kinds a deterministic verifier can compute (arithmetic, equation, symbolic, unit, schema)
    bool has_formal = false;
    logic::VerificationClaim formal;
};

struct DetectorConfig {
    // Claims below this risk are not reported at all (Low-risk prose costs one cheap scan and nothing else).
    ClaimRisk min_risk = ClaimRisk::kMedium;
    size_t max_claims_per_sentence = 8;
};

// Finds verifiable claims in generated text. Incremental: `scan` is called with the whole text so far and
// only inspects complete sentences that start at or after `from`, so cost grows with new text, not total text.
class ClaimDetector {
public:
    explicit ClaimDetector(const DetectorConfig& cfg = DetectorConfig());
    ~ClaimDetector();

    // Cheap pre-filter: can this text contain anything risky at all? (digits, slashes, backticks, brackets...)
    static bool may_contain_claims(const std::string& text);

    // End of the last complete sentence in text[from, ...); `from` if none is complete yet.
    static size_t complete_sentence_end(const std::string& text, size_t from);

    std::vector<Claim> scan(const std::string& text, size_t from, size_t to,
                            const RequestScope& scope) const;

    // Overall risk of a span of text (for deciding how much verification effort it deserves).
    ClaimRisk classify(const std::string& text) const;

private:
    struct Impl;
    Impl* impl_;
    DetectorConfig cfg_;
};

} // namespace strata::rt
