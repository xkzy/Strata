// src/math/expression_parser.cpp - Mathematical Expression Parser & Validator Implementation
#include "strata/math/expression_parser.hpp"

#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>

namespace strata::math {

// ---------------------------------------------------------------------------
// ExpressionValidator Implementation
// ---------------------------------------------------------------------------

ExpressionValidator::ExpressionValidator(const MathSecurityLimits& limits)
    : limits_(limits) {}

bool ExpressionValidator::validate(const std::string& expression, std::string& error_msg) const {
    if (expression.empty()) {
        error_msg = "Expression is empty";
        return false;
    }

    if (expression.length() > limits_.max_expression_length) {
        error_msg = "Expression exceeds maximum allowed length (" +
                    std::to_string(limits_.max_expression_length) + " characters)";
        return false;
    }

    // Check maximum nesting depth of parentheses, brackets, braces
    int depth = 0;
    int max_depth = 0;
    for (char c : expression) {
        if (c == '(' || c == '[' || c == '{') {
            depth++;
            if (depth > max_depth) max_depth = depth;
        } else if (c == ')' || c == ']' || c == '}') {
            depth--;
            if (depth < 0) {
                error_msg = "Mismatched closing bracket in expression";
                return false;
            }
        }
    }

    if (depth != 0) {
        error_msg = "Unclosed opening bracket in expression";
        return false;
    }

    if (static_cast<uint32_t>(max_depth) > limits_.max_recursion_depth) {
        error_msg = "Expression nesting depth (" + std::to_string(max_depth) +
                    ") exceeds recursion limit (" + std::to_string(limits_.max_recursion_depth) + ")";
        return false;
    }

    // Security check: blacklist dangerous system / execution tokens
    static const std::vector<std::string> blacklist = {
        "import", "exec", "eval", "__", "system", "os.", "subprocess", "open(", "file(",
        "globals", "locals", "compile", "builtins", "getattr", "setattr", "delattr",
        "lambda", "while", "for", "def", "class", "socket", "http", "curl", "wget",
        "rm ", "sh", "bash", "powershell", "cmd.exe"
    };

    std::string lower_expr = expression;
    std::transform(lower_expr.begin(), lower_expr.end(), lower_expr.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    for (const auto& token : blacklist) {
        if (lower_expr.find(token) != std::string::npos) {
            error_msg = "Expression contains forbidden or non-mathematical token: '" + token + "'";
            return false;
        }
    }

    // Check computed complexity score
    uint32_t complexity = compute_complexity(expression);
    if (complexity > limits_.max_complexity_score) {
        error_msg = "Expression complexity score (" + std::to_string(complexity) +
                    ") exceeds maximum budget (" + std::to_string(limits_.max_complexity_score) + ")";
        return false;
    }

    return true;
}

uint32_t ExpressionValidator::compute_complexity(const std::string& expression) const {
    uint32_t score = 1;
    int nesting = 0;

    for (size_t i = 0; i < expression.length(); ++i) {
        char c = expression[i];
        if (c == '(' || c == '[' || c == '{') {
            nesting++;
            score += 2 * (nesting + 1);
        } else if (c == ')' || c == ']' || c == '}') {
            if (nesting > 0) nesting--;
        } else if (c == '+' || c == '-') {
            score += 1;
        } else if (c == '*' || c == '/') {
            score += 2;
        } else if (c == '^') {
            score += 5;
        } else if (std::isalpha(static_cast<unsigned char>(c))) {
            // Function call or variable
            score += 3;
        }
    }

    return score;
}

bool ExpressionValidator::is_pure_arithmetic(const std::string& expression) const {
    // Check if the expression contains only numbers, rational operators, and basic arithmetic
    for (char c : expression) {
        if (std::isalpha(static_cast<unsigned char>(c))) {
            return false;
        }
        if (c != '0' && c != '1' && c != '2' && c != '3' && c != '4' &&
            c != '5' && c != '6' && c != '7' && c != '8' && c != '9' &&
            c != '+' && c != '-' && c != '*' && c != '/' && c != '%' &&
            c != '^' && c != '(' && c != ')' && c != '.' && c != ',' &&
            c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// ExpressionParser Implementation
// ---------------------------------------------------------------------------

ExpressionParser::ExpressionParser() = default;

// Normalizes spelling only; it must never change what the text means. (An earlier version removed every space and every
// comma-before-three-digits: "gcd(12,345)" became "gcd(12345)" and "2 3" (a product) became "23". Both collided with
// different expressions in the result cache.)
std::string ExpressionParser::canonicalize(const std::string& expr) {
    std::string clean;
    clean.reserve(expr.length());
    int depth = 0;   // inside (), [] or {} a comma separates arguments and is never a thousands separator
    auto is_operand_end = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == ')' || c == ']' || c == '}' || c == '_' || c == '.' || c == '!'; };
    auto is_operand_start = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '(' || c == '[' || c == '{' || c == '_' || c == '.'; };
    bool pending_space = false;

    for (size_t i = 0; i < expr.length(); ++i) {
        const unsigned char c = static_cast<unsigned char>(expr[i]);
        if (c == '(' || c == '[' || c == '{') ++depth;
        else if ((c == ')' || c == ']' || c == '}') && depth > 0) --depth;

        // thousands separator in a plain number at top level only: 2,384 -> 2384 (but never gcd(12,345) or {1,234})
        if (c == ',' && depth == 0 && i > 0 && std::isdigit(static_cast<unsigned char>(expr[i - 1])) && i + 3 < expr.length() &&
            std::isdigit(static_cast<unsigned char>(expr[i + 1])) && std::isdigit(static_cast<unsigned char>(expr[i + 2])) &&
            std::isdigit(static_cast<unsigned char>(expr[i + 3])) && (i + 4 >= expr.length() || !std::isdigit(static_cast<unsigned char>(expr[i + 4])))) {
            size_t b = i;
            while (b > 0 && (std::isdigit(static_cast<unsigned char>(expr[b - 1])) || expr[b - 1] == ',')) --b;   // the number's digit groups so far
            const bool leading_group_ok = (i - b) <= 3 || expr[i - 4] == ',';
            if (leading_group_ok) continue;
        }
        // unicode operators: x (C3 97) -> *, ÷ (C3 B7) -> /, . (C2 B7) -> *, minus (E2 88 92) -> -
        if (c == 0xC3 && i + 1 < expr.length() && static_cast<unsigned char>(expr[i + 1]) == 0x97) { clean.push_back('*'); ++i; pending_space = false; continue; }
        if (c == 0xC3 && i + 1 < expr.length() && static_cast<unsigned char>(expr[i + 1]) == 0xB7) { clean.push_back('/'); ++i; pending_space = false; continue; }
        if (c == 0xC2 && i + 1 < expr.length() && static_cast<unsigned char>(expr[i + 1]) == 0xB7) { clean.push_back('*'); ++i; pending_space = false; continue; }
        if (c == 0xE2 && i + 2 < expr.length() && static_cast<unsigned char>(expr[i + 1]) == 0x88 && static_cast<unsigned char>(expr[i + 2]) == 0x92) { clean.push_back('-'); i += 2; pending_space = false; continue; }

        if (std::isspace(c)) { pending_space = true; continue; }
        // a space between two operands is meaning (implicit multiplication: "2 3", "x y"); around operators it is not
        if (pending_space && !clean.empty() && is_operand_end(clean.back()) && is_operand_start(static_cast<char>(c))) clean.push_back(' ');
        pending_space = false;
        clean.push_back(static_cast<char>(c));
    }
    return clean;
}

std::string ExpressionParser::format_compact_observation(const std::string& expr,
                                                        const std::string& exact_res,
                                                        const std::string& numeric_res,
                                                        MathOperation op) {
    std::string obs;
    obs.reserve(128);

    obs += "[MathResult: ";
    obs += math_operation_to_string(op);
    obs += "(";
    obs += expr;
    obs += ") = ";

    if (!exact_res.empty()) {
        obs += exact_res;
        if (!numeric_res.empty() && numeric_res != exact_res) {
            obs += " (approx " + numeric_res + ")";
        }
    } else if (!numeric_res.empty()) {
        obs += numeric_res;
    } else {
        obs += "null";
    }

    obs += "]";
    return obs;
}

std::vector<ParsedCalculationIntent> ExpressionParser::detect_calculation_intents(const std::string& text) const {
    std::vector<ParsedCalculationIntent> results;
    // std::regex matches recursively in the length of the input: a very long text overflows the stack. Prompts that
    // ask for a calculation are short; a longer text is not scanned.
    if (text.size() > 4096) return results;

    // Pattern 1: Explicit arithmetic calculation like "2384 * 7291" or "2,384 x 7,291"
    std::regex arith_regex(R"((?:calculate|eval|compute)?\s*([0-9]+(?:\.[0-9]+)?\s*[\+\-\*\/\^×÷]\s*[0-9]+(?:\.[0-9]+)?(?:\s*[\+\-\*\/\^×÷]\s*[0-9]+(?:\.[0-9]+)?)*))",
                           std::regex::icase);

    // Pattern 2: Calculus derivatives: "differentiate sin(x^2)" or "derivative of x^3"
    std::regex diff_regex(R"((?:differentiate|derivative(?:\s+of)?|d\/dx)\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s]+?)(?:\s+with\s+respect\s+to\s+([a-zA-Z]))?$)",
                          std::regex::icase);

    // Pattern 3: Calculus integrals: "integrate x^2 sin(x)" or "integral of x^2"
    std::regex int_regex(R"((?:integrate|integral(?:\s+of)?)\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s]+?)(?:\s+d([a-zA-Z]))?$)",
                         std::regex::icase);

    // Pattern 4: Equations: "solve x^2 + 5x + 6 == 0"
    std::regex solve_regex(R"((?:solve|find\s+roots\s+of)\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s\=\<\>]+?)(?:\s+for\s+([a-zA-Z]))?$)",
                           std::regex::icase);

    // Pattern 5: Roots / Functions: "sqrt(123456789)"
    std::regex sqrt_regex(R"((sqrt\([0-9\.]+\)))", std::regex::icase);

    // Pattern 6: Simplification / Expansion / Factorization: "simplify x^2 - y^2", "factor x^2 + 5x + 6"
    std::regex simplify_regex(R"((?:simplify)\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s]+))", std::regex::icase);
    std::regex factor_regex(R"((?:factor)\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s]+))", std::regex::icase);
    std::regex expand_regex(R"((?:expand)\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s]+))", std::regex::icase);

    std::smatch m;
    if (std::regex_search(text, m, diff_regex)) {
        ParsedCalculationIntent intent;
        intent.operation = MathOperation::kDifferentiate;
        intent.expression = m[1].str();
        if (m.size() > 2 && m[2].matched) {
            intent.variable = m[2].str();
        } else {
            intent.variable = "x";
        }
        intent.raw_match = m[0].str();
        intent.start_pos = m.position(0);
        intent.length = m.length(0);
        results.push_back(intent);
    } else if (std::regex_search(text, m, int_regex)) {
        ParsedCalculationIntent intent;
        intent.operation = MathOperation::kIntegrate;
        intent.expression = m[1].str();
        if (m.size() > 2 && m[2].matched) {
            intent.variable = m[2].str();
        } else {
            intent.variable = "x";
        }
        intent.raw_match = m[0].str();
        intent.start_pos = m.position(0);
        intent.length = m.length(0);
        results.push_back(intent);
    } else if (std::regex_search(text, m, solve_regex)) {
        ParsedCalculationIntent intent;
        intent.operation = MathOperation::kSolve;
        intent.expression = m[1].str();
        if (m.size() > 2 && m[2].matched) {
            intent.variable = m[2].str();
        } else {
            intent.variable = "x";
        }
        intent.raw_match = m[0].str();
        intent.start_pos = m.position(0);
        intent.length = m.length(0);
        results.push_back(intent);
    } else if (std::regex_search(text, m, simplify_regex)) {
        ParsedCalculationIntent intent;
        intent.operation = MathOperation::kSimplify;
        intent.expression = m[1].str();
        intent.raw_match = m[0].str();
        intent.start_pos = m.position(0);
        intent.length = m.length(0);
        results.push_back(intent);
    } else if (std::regex_search(text, m, factor_regex)) {
        ParsedCalculationIntent intent;
        intent.operation = MathOperation::kFactor;
        intent.expression = m[1].str();
        intent.raw_match = m[0].str();
        intent.start_pos = m.position(0);
        intent.length = m.length(0);
        results.push_back(intent);
    } else if (std::regex_search(text, m, expand_regex)) {
        ParsedCalculationIntent intent;
        intent.operation = MathOperation::kExpand;
        intent.expression = m[1].str();
        intent.raw_match = m[0].str();
        intent.start_pos = m.position(0);
        intent.length = m.length(0);
        results.push_back(intent);
    } else if (std::regex_search(text, m, sqrt_regex)) {
        ParsedCalculationIntent intent;
        intent.operation = MathOperation::kEvaluate;
        intent.expression = m[1].str();
        intent.raw_match = m[0].str();
        intent.start_pos = m.position(0);
        intent.length = m.length(0);
        results.push_back(intent);
    } else if (std::regex_search(text, m, arith_regex)) {
        ParsedCalculationIntent intent;
        intent.operation = MathOperation::kEvaluate;
        intent.expression = m[1].str();
        intent.raw_match = m[0].str();
        intent.start_pos = m.position(0);
        intent.length = m.length(0);
        results.push_back(intent);
    }

    return results;
}

bool ExpressionParser::parse_structured_payload(const std::string& json_str, MathRequest& request, std::string& error_msg) const {
    if (json_str.empty()) {
        error_msg = "Payload string is empty";
        return false;
    }

    // Lightweight structured JSON parser
    auto extract_field = [](const std::string& json, const std::string& key) -> std::string {
        std::string pattern = "\"" + key + "\"\\s*:\\s*\"([^\"]*)\"";
        std::regex re(pattern);
        std::smatch m;
        if (std::regex_search(json, m, re)) {
            return m[1].str();
        }
        return "";
    };

    std::string op_str = extract_field(json_str, "operation");
    std::string expr = extract_field(json_str, "expression");
    std::string mode_str = extract_field(json_str, "mode");
    std::string var_str = extract_field(json_str, "variable");

    if (expr.empty()) {
        // Check if expression was non-quoted integer or value
        std::regex num_expr_re(R"lit("expression"\s*:\s*([0-9\+\-\*\/\^\(\)\.\s]+))lit");
        std::smatch nm;
        if (std::regex_search(json_str, nm, num_expr_re)) {
            expr = nm[1].str();
        }
    }

    if (expr.empty()) {
        error_msg = "Missing 'expression' field in structured JSON payload";
        return false;
    }

    request.expression = expr;

    if (!op_str.empty()) {
        if (op_str == "simplify") request.operation = MathOperation::kSimplify;
        else if (op_str == "expand") request.operation = MathOperation::kExpand;
        else if (op_str == "factor") request.operation = MathOperation::kFactor;
        else if (op_str == "solve") request.operation = MathOperation::kSolve;
        else if (op_str == "differentiate") request.operation = MathOperation::kDifferentiate;
        else if (op_str == "integrate") request.operation = MathOperation::kIntegrate;
        else if (op_str == "limit") request.operation = MathOperation::kLimit;
        else if (op_str == "series") request.operation = MathOperation::kSeries;
        else if (op_str == "numeric_evaluate") request.operation = MathOperation::kNumericEvaluate;
        else if (op_str == "determinant") request.operation = MathOperation::kDeterminant;
        else if (op_str == "probability") request.operation = MathOperation::kProbability;
        else request.operation = MathOperation::kEvaluate;
    }

    if (!mode_str.empty()) {
        if (mode_str == "numeric") request.mode = MathMode::kNumeric;
        else if (mode_str == "symbolic") request.mode = MathMode::kSymbolic;
        else request.mode = MathMode::kExact;
    }

    if (!var_str.empty()) {
        request.variable = var_str;
    }

    return true;
}

} // namespace strata::math
