// include/strata/logic_verification/claim_extractor.hpp - Claim Extraction Engine
//
// Extracts verifiable claims (arithmetic, equations, symbolic calculus, constraints,
// propositions, units, JSON schemas, SQL queries) from LLM generation text or structured payloads.
#pragma once

#include "strata/logic_verification/verification_types.hpp"

#include <string>
#include <vector>

namespace strata::logic {

class ClaimExtractor {
public:
    ClaimExtractor();
    ~ClaimExtractor();

    // Extract all identifiable verifiable claims from unstructured or semi-structured LLM text
    std::vector<VerificationClaim> extract_claims(const std::string& text,
                                                  const std::string& tenant_id = "default",
                                                  const std::string& session_id = "default") const;

    // Parse explicit JSON claim descriptor
    bool parse_json_claim(const std::string& json_str, VerificationClaim& out_claim, std::string& err) const;

    // Fast heuristics for intent / claim presence
    bool has_verifiable_content(const std::string& text) const;

private:
    void extract_arithmetic_claims(const std::string& text, std::vector<VerificationClaim>& claims,
                                   const std::string& tenant_id, const std::string& session_id) const;

    void extract_equation_claims(const std::string& text, std::vector<VerificationClaim>& claims,
                                 const std::string& tenant_id, const std::string& session_id) const;

    void extract_proposition_claims(const std::string& text, std::vector<VerificationClaim>& claims,
                                    const std::string& tenant_id, const std::string& session_id) const;

    void extract_constraint_claims(const std::string& text, std::vector<VerificationClaim>& claims,
                                   const std::string& tenant_id, const std::string& session_id) const;

    void extract_unit_claims(const std::string& text, std::vector<VerificationClaim>& claims,
                             const std::string& tenant_id, const std::string& session_id) const;

    void extract_json_and_code_claims(const std::string& text, std::vector<VerificationClaim>& claims,
                                      const std::string& tenant_id, const std::string& session_id) const;

    void extract_sql_claims(const std::string& text, std::vector<VerificationClaim>& claims,
                            const std::string& tenant_id, const std::string& session_id) const;
};

} // namespace strata::logic
