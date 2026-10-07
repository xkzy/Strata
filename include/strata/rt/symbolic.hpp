// include/strata/rt/symbolic.hpp - equation and calculus claims stated in prose, checked on the native CAS
//
// "The derivative of x^2 is 2x."  "(x+1)^2 expands to x^2 + 2x + 1."  "The solution of 2x + 5 = 15 is x = 5."
// Only phrasings that make an unambiguous mathematical statement are recognized, and only when both sides parse as
// expressions: "the line is y = 2x + 1" is a definition, not an identity, and is never checked.
#pragma once

#include "strata/rt/verifier.hpp"

#include <string>
#include <vector>

namespace strata::rt {

struct SymbolicSpec {
    std::string kind;    // derivative | integral | expand | simplify | factor | equals | solve
    std::string f;       // the expression the statement is about (for solve: the equation "lhs = rhs")
    std::string g;       // the claimed result (for solve: "x = 2; x = 3")
    std::string var;     // variable (derivative / integral / solve), may be empty
    std::string shown;   // the statement as it appears
};

// Finds the statements of this kind in one sentence.
std::vector<SymbolicSpec> find_symbolic_claims(const std::string& sentence);
// The text a claim carries in Claim::formal.expression, and back.
std::string encode_symbolic(const SymbolicSpec& s);
bool decode_symbolic(const std::string& text, SymbolicSpec& out);

class SymbolicVerifier : public IClaimVerifier {
public:
    std::string name() const override { return "symbolic_cas"; }
    std::string version() const override { return "1"; }
    std::string backend_version() const override { return "cas-0.2.0"; }
    Tier tier() const override { return Tier::kSpecialized; }
    bool handles(const Claim& c) const override { return c.kind == ClaimKind::kSymbolic && c.has_formal; }
    VerifyOutcome verify(const Claim& claim, const EvidenceSet& evidence, Clock::time_point deadline) override;
};

} // namespace strata::rt
