// include/strata/math/expression_parser.hpp - Mathematical Expression Parser & Validator
//
// Parses mathematical strings, performs safety validation, computes complexity scores,
// and extracts calculation intents from model prompt / response streams.
#pragma once

#include "strata/math/math_types.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace strata::math {

struct ParsedCalculationIntent {
    MathOperation operation = MathOperation::kEvaluate;
    std::string expression;
    std::string variable = "x";
    std::string point = "0";
    int order = 1;
    MathMode mode = MathMode::kExact;
    std::string raw_match;
    size_t start_pos = 0;
    size_t length = 0;
    double confidence = 1.0;
};

class ExpressionValidator {
public:
    explicit ExpressionValidator(const MathSecurityLimits& limits = MathSecurityLimits());

    // Validates that an expression is safe to evaluate (no command injection, within length & depth limits)
    bool validate(const std::string& expression, std::string& error_msg) const;

    // Computes heuristic complexity score based on operators, nesting depth, variable count, and functions
    uint32_t compute_complexity(const std::string& expression) const;

    // Checks whether expression is purely numerical / rational arithmetic (eligible for FastNumericBackend)
    bool is_pure_arithmetic(const std::string& expression) const;

    const MathSecurityLimits& limits() const { return limits_; }

private:
    MathSecurityLimits limits_;
};

class ExpressionParser {
public:
    ExpressionParser();

    // Canonicalizes mathematical expression for cache lookups (strips unnecessary whitespace, standardizes multiplication)
    static std::string canonicalize(const std::string& expr);

    // Formats compact LLM observation from exact and numeric result
    static std::string format_compact_observation(const std::string& expr,
                                                 const std::string& exact_res,
                                                 const std::string& numeric_res,
                                                 MathOperation op);

    // Scans text for explicit mathematical calculations (e.g. "2,384 * 7,291", "integrate x^2 dx", "solve x^2 + 5x + 6 == 0")
    std::vector<ParsedCalculationIntent> detect_calculation_intents(const std::string& text) const;

    // Parses structured JSON math request payload
    bool parse_structured_payload(const std::string& json_str, MathRequest& request, std::string& error_msg) const;
};

} // namespace strata::math
