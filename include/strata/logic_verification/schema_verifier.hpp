// include/strata/logic_verification/schema_verifier.hpp - Structured JSON & Type Schema Verifier
//
// Verifies JSON syntax, schema definitions, required properties, types, ranges,
// and structural invariants deterministically.
#pragma once

#include "strata/logic_verification/verification_types.hpp"

#include <string>
#include <vector>

namespace strata::logic {

class SchemaVerifier {
public:
    SchemaVerifier();
    ~SchemaVerifier();

    // Verify raw JSON string against schema definition
    VerificationResult verify_schema(const std::string& json_payload,
                                     const std::string& schema_definition,
                                     const std::string& claim_id = "schema_1") const;

    // Fast check for syntactically valid JSON
    bool is_valid_json(const std::string& json_str, std::string& err) const;
};

} // namespace strata::logic
