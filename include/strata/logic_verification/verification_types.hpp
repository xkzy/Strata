// include/strata/logic_verification/verification_types.hpp - Logic Verification Type System
//
// Core abstractions, tri-state verification result model (PASS / FAIL / UNKNOWN / PARTIAL),
// Intermediate Representation (IR) for claims, tolerances, and metrics.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::logic {

enum class VerificationStatus {
    kPass = 0,      // Deterministic evidence confirms the claim
    kFail,          // Deterministic evidence contradicts the claim
    kUnknown,       // Cannot establish correctness deterministically (NEVER converted to PASS)
    kPartial        // Multi-component claim partially verified
};

inline const char* verification_status_to_string(VerificationStatus s) {
    switch (s) {
        case VerificationStatus::kPass: return "PASS";
        case VerificationStatus::kFail: return "FAIL";
        case VerificationStatus::kUnknown: return "UNKNOWN";
        case VerificationStatus::kPartial: return "PARTIAL";
    }
    return "UNKNOWN";
}

inline VerificationStatus string_to_verification_status(const std::string& s) {
    if (s == "PASS" || s == "pass") return VerificationStatus::kPass;
    if (s == "FAIL" || s == "fail") return VerificationStatus::kFail;
    if (s == "PARTIAL" || s == "partial") return VerificationStatus::kPartial;
    return VerificationStatus::kUnknown;
}

enum class ClaimType {
    kArithmetic = 0,    // Pure arithmetic (exact integer, rational, floating point)
    kSymbolic,          // Symbolic mathematical identity, calculus, expansion
    kEquation,          // Algebraic equation solutions (e.g. 2x+5=15, x=5)
    kProposition,       // Propositional logic, modus ponens, implications, contradictions
    kConstraint,        // Linear inequality & variable interval bounds (e.g. x>0, x<10)
    kConsistency,       // State collision, duplicate incompatible assignments
    kUnitDimension,     // Dimensional analysis and SI unit conversions
    kSchemaType,        // JSON Schema, field existence, type constraints
    kCodeSyntax,        // Code syntax, compilation, unit test assertions
    kSQL,               // SQL syntax, read-only safety, table/column schema validity
    kComposite,         // Composite multi-claim bundle
    kUnknown            // Unstructured natural language claim
};

inline const char* claim_type_to_string(ClaimType t) {
    switch (t) {
        case ClaimType::kArithmetic: return "arithmetic";
        case ClaimType::kSymbolic: return "symbolic";
        case ClaimType::kEquation: return "equation";
        case ClaimType::kProposition: return "proposition";
        case ClaimType::kConstraint: return "constraint";
        case ClaimType::kConsistency: return "consistency";
        case ClaimType::kUnitDimension: return "unit_dimension";
        case ClaimType::kSchemaType: return "schema_type";
        case ClaimType::kCodeSyntax: return "code_syntax";
        case ClaimType::kSQL: return "sql";
        case ClaimType::kComposite: return "composite";
        case ClaimType::kUnknown: return "unknown";
    }
    return "unknown";
}

struct NumericalTolerance {
    double abs_tol = 1e-6;
    double rel_tol = 1e-5;
    bool exact_only = false;
};

struct VerificationClaim {
    std::string claim_id;
    ClaimType type = ClaimType::kUnknown;
    std::string raw_statement;      // Original extracted text / statement
    std::string expression;         // Formal expression to evaluate / check
    std::string claimed_value;      // What LLM claims the result / state is

    // Contextual & relational components
    std::vector<std::string> premises;      // For logical deductions (A, A->B)
    std::vector<std::string> constraints;   // For constraint systems (x > 0, x < 10)
    std::string schema_json;                // For schema checks
    std::string code_snippet;               // For code checks
    std::string language = "generic";
    std::string sql_query;                  // For SQL checks
    std::string unit_expression;            // For unit dimension checks (e.g. "10 m / 2 s")
    std::string claimed_unit;               // e.g. "m/s" or "kg"

    // Sub-claims for composite verification
    std::vector<VerificationClaim> sub_claims;

    // Multi-tenant isolation
    std::string tenant_id = "default";
    std::string session_id = "default";
    std::string user_id = "default";

    NumericalTolerance tolerance;
    std::unordered_map<std::string, std::string> metadata;
};

struct VerificationResult {
    std::string claim_id;
    VerificationStatus status = VerificationStatus::kUnknown;
    ClaimType type = ClaimType::kUnknown;
    std::string backend_used;               // e.g. "fast_numeric", "mathics", "rule_engine", "constraint_engine"
    std::string evidence;                   // Deterministic proof or contradiction explanation
    std::string failure_reason;
    std::string expected_value;
    std::string actual_value;

    // Structured sub-component verification (for PARTIAL / composite)
    std::vector<std::string> verified_components;
    std::vector<std::string> failed_components;
    std::vector<std::string> unknown_components;

    // Compact LLM-facing observation (prevents context window pollution)
    std::string compact_observation;
    int64_t observation_tokens = 0;

    double execution_time_ms = 0.0;
    bool cache_hit = false;
    std::string canonical_hash;
    uint64_t timestamp_ns = 0;
};

struct VerificationMetrics {
    uint64_t total_verifications = 0;
    uint64_t pass_count = 0;
    uint64_t fail_count = 0;
    uint64_t unknown_count = 0;
    uint64_t partial_count = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t retries_attempted = 0;
    uint64_t retries_succeeded = 0;
    double total_execution_time_ms = 0.0;
    std::unordered_map<std::string, uint64_t> backend_distribution;

    double pass_rate() const {
        return total_verifications > 0 ? static_cast<double>(pass_count) / total_verifications : 0.0;
    }
    double cache_hit_rate() const {
        uint64_t total_lookups = cache_hits + cache_misses;
        return total_lookups > 0 ? static_cast<double>(cache_hits) / total_lookups : 0.0;
    }
    double avg_latency_ms() const {
        return total_verifications > 0 ? total_execution_time_ms / total_verifications : 0.0;
    }
};

} // namespace strata::logic
