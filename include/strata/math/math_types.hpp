// include/strata/math/math_types.hpp - Mathematical Backing Engine & Runtime Types
//
// Defines structured mathematical execution types, operations, modes,
// backends, and security parameters for deterministic math evaluation in Strata.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::math {

enum class MathOperation {
    kEvaluate = 0,         // General evaluation
    kSimplify,             // Algebraic simplification
    kExpand,               // Polynomial / algebraic expansion
    kFactor,               // Polynomial / integer factorization
    kSolve,                // Algebraic / transcendental equation solving
    kDifferentiate,        // Symbolic derivative
    kIntegrate,            // Symbolic / definite integral
    kLimit,                // Analytical limit
    kSeries,               // Taylor / power series expansion
    kNumericEvaluate,      // High precision numerical evaluation (N[expr])
    kMatrixOp,             // Matrix multiplication, inverse, transpose
    kDeterminant,          // Matrix determinant
    kProbability           // Permutations, combinations, distributions
};

enum class MathMode {
    kExact = 0,            // Exact rational / symbolic arithmetic by default
    kNumeric,              // Floating point / arbitrary precision numerical
    kSymbolic              // Pure symbolic transformation
};

enum class MathBackendType {
    kFastNumeric = 0,      // Lightweight fast path for arithmetic / rational math
    kMathics,              // Mathics3 CAS (Wolfram Language compatible kernel)
    kSageMath,             // SageMath CAS (Python / Sage open-source mathematics system)
    kSymPy,                // Python SymPy CAS backend
    kCustom                // Custom or future CAS backend
};

enum class MathStatus {
    kSuccess = 0,
    kTimeout,
    kInvalidExpression,
    kExecutionError,
    kCancelled,
    kResourceLimitExceeded,
    kUnsupportedOperation
};

struct MathSecurityLimits {
    size_t max_expression_length = 4096;
    uint32_t max_complexity_score = 10000;
    double timeout_ms = 5000.0;
    uint32_t max_recursion_depth = 50;
    uint32_t max_memory_mb = 512;
    uint32_t max_concurrent_evaluations = 8;
};

struct MathRequest {
    int64_t request_id = 0;
    std::string tenant_id = "default_tenant";
    std::string user_id = "default_user";
    std::string project_id = "default_project";
    std::string session_id = "default_session";
    std::string agent_id = "default_agent";

    MathOperation operation = MathOperation::kEvaluate;
    std::string expression;
    std::string variable;           // e.g. "x" for calculus/solve
    std::string point;              // e.g. "0" for limits/series
    int order = 1;                  // e.g. derivative order or series expansion order
    std::string assumptions;        // e.g. "x > 0", "integers"
    MathMode mode = MathMode::kExact;
    int precision_digits = 15;
    double timeout_ms = 5000.0;
};

// Where a result came from: enough to reproduce it and to decide whether a cached copy may be reused.
struct MathProvenance {
    std::string expression_hash;        // content hash of normalized expression + operation + assumptions + precision
    std::string normalized_expression;
    std::string operation;
    std::string backend;
    std::string backend_version;
    std::string algorithm;              // e.g. "exact rational CAS evaluation", "int64 fast path (overflow-checked)"
    std::string assumptions;
    int precision_digits = 0;           // 0 = exact
    std::string rounding_mode = "n/a";  // numeric results are rounded to precision_digits by printf %g (round-half-even on the binary value)
    bool exact = true;
    std::string ir_version;
    std::vector<std::string> input_hashes;
    int64_t timestamp_ms = 0;
};

struct MathResult {
    int64_t result_id = 0;
    int64_t request_id = 0;
    MathStatus status = MathStatus::kSuccess;

    MathBackendType backend_type = MathBackendType::kFastNumeric;
    std::string backend_name;
    std::string backend_version;

    std::string canonical_expression;
    std::string exact_result;       // Exact rational / symbolic representation
    std::string numeric_result;     // Numerical approximation if requested or available
    std::string raw_result;         // Raw CAS output string
    std::string compact_observation;// Compact result string suitable for LLM injection

    double execution_time_ms = 0.0;
    uint32_t complexity_score = 0;
    bool cache_hit = false;
    int64_t observation_tokens = 0;
    std::string error_message;

    MathProvenance provenance;

    std::unordered_map<std::string, std::string> metadata;
};

struct MathVerificationResult {
    bool matches = false;
    std::string llm_claimed_result;
    std::string ground_truth_result;
    std::string discrepancy_details;
    double relative_error = 0.0;
    double confidence = 1.0;
};

struct MathRuntimeStats {
    uint64_t total_calculations = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t fast_path_count = 0;
    uint64_t mathics_count = 0;
    uint64_t verification_count = 0;
    uint64_t verification_failures = 0;
    uint64_t timeouts = 0;
    uint64_t security_rejections = 0;
    double total_execution_time_ms = 0.0;

    double cache_hit_rate() const {
        if (total_calculations == 0) return 0.0;
        return static_cast<double>(cache_hits) / static_cast<double>(total_calculations);
    }

    double avg_execution_time_ms() const {
        if (total_calculations == 0) return 0.0;
        return total_execution_time_ms / static_cast<double>(total_calculations);
    }
};

inline const char* math_operation_to_string(MathOperation op) {
    switch (op) {
        case MathOperation::kEvaluate: return "evaluate";
        case MathOperation::kSimplify: return "simplify";
        case MathOperation::kExpand: return "expand";
        case MathOperation::kFactor: return "factor";
        case MathOperation::kSolve: return "solve";
        case MathOperation::kDifferentiate: return "differentiate";
        case MathOperation::kIntegrate: return "integrate";
        case MathOperation::kLimit: return "limit";
        case MathOperation::kSeries: return "series";
        case MathOperation::kNumericEvaluate: return "numeric_evaluate";
        case MathOperation::kMatrixOp: return "matrix_op";
        case MathOperation::kDeterminant: return "determinant";
        case MathOperation::kProbability: return "probability";
        default: return "unknown";
    }
}

inline const char* math_mode_to_string(MathMode mode) {
    switch (mode) {
        case MathMode::kExact: return "exact";
        case MathMode::kNumeric: return "numeric";
        case MathMode::kSymbolic: return "symbolic";
        default: return "unknown";
    }
}

inline const char* math_backend_to_string(MathBackendType backend) {
    switch (backend) {
        case MathBackendType::kFastNumeric: return "FastNumeric";
        case MathBackendType::kMathics: return "Mathics3";
        case MathBackendType::kSymPy: return "SymPy";
        case MathBackendType::kCustom: return "Custom";
        default: return "Unknown";
    }
}

} // namespace strata::math
