// src/math/math_backend.cpp - Mathematical Execution Backend Implementations
#include "strata/math/math_backend.hpp"
#include "strata/math/expression_parser.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <regex>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <vector>

namespace strata::math {

// ---------------------------------------------------------------------------
// FastNumericBackend::Rational Helper Implementation
// ---------------------------------------------------------------------------


// The fast path works in int64. Every operation is checked: a result it cannot represent exactly is reported (and the
// runtime re-runs the request on the arbitrary-precision CAS) instead of wrapping around into a wrong "exact" answer.
struct FastPathLimit : std::runtime_error {
    using std::runtime_error::runtime_error;
};
static int64_t mul_ck(int64_t a, int64_t b) {
#if defined(__GNUC__) || defined(__clang__)
    int64_t r;
    if (__builtin_mul_overflow(a, b, &r)) throw FastPathLimit("int64 overflow");
    return r;
#else
    if (std::fabs(static_cast<double>(a) * static_cast<double>(b)) >= 9.2e18) throw FastPathLimit("int64 overflow");
    return a * b;
#endif
}
static int64_t add_ck(int64_t a, int64_t b) {
#if defined(__GNUC__) || defined(__clang__)
    int64_t r;
    if (__builtin_add_overflow(a, b, &r)) throw FastPathLimit("int64 overflow");
    return r;
#else
    if (std::fabs(static_cast<double>(a) + static_cast<double>(b)) >= 9.2e18) throw FastPathLimit("int64 overflow");
    return a + b;
#endif
}
static int64_t sub_ck(int64_t a, int64_t b) { return add_ck(a, b == INT64_MIN ? throw FastPathLimit("int64 overflow") : -b); }

static int64_t compute_gcd(int64_t a, int64_t b) {
    a = std::abs(a);
    b = std::abs(b);
    while (b != 0) {
        int64_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

FastNumericBackend::Rational::Rational(int64_t n, int64_t d) : num(n), den(d) {
    reduce();
}

void FastNumericBackend::Rational::reduce() {
    if (den < 0) {
        num = -num;
        den = -den;
    }
    if (den == 0) {
        // Leave as is or division by zero
        return;
    }
    int64_t g = compute_gcd(num, den);
    if (g > 1) {
        num /= g;
        den /= g;
    }
}

FastNumericBackend::Rational FastNumericBackend::Rational::operator+(const Rational& o) const {
    int64_t common_den = mul_ck(den, o.den);
    int64_t new_num = add_ck(mul_ck(num, o.den), mul_ck(o.num, den));
    return Rational(new_num, common_den);
}

FastNumericBackend::Rational FastNumericBackend::Rational::operator-(const Rational& o) const {
    int64_t common_den = mul_ck(den, o.den);
    int64_t new_num = sub_ck(mul_ck(num, o.den), mul_ck(o.num, den));
    return Rational(new_num, common_den);
}

FastNumericBackend::Rational FastNumericBackend::Rational::operator*(const Rational& o) const {
    return Rational(mul_ck(num, o.num), mul_ck(den, o.den));
}

FastNumericBackend::Rational FastNumericBackend::Rational::operator/(const Rational& o) const {
    return Rational(mul_ck(num, o.den), mul_ck(den, o.num));
}

std::string FastNumericBackend::Rational::to_string() const {
    if (den == 1) {
        return std::to_string(num);
    }
    return std::to_string(num) + "/" + std::to_string(den);
}

double FastNumericBackend::Rational::to_double() const {
    if (den == 0) return 0.0;
    return static_cast<double>(num) / static_cast<double>(den);
}

// ---------------------------------------------------------------------------
// FastNumericBackend Implementation
// ---------------------------------------------------------------------------

FastNumericBackend::FastNumericBackend() = default;

bool FastNumericBackend::supports_operation(MathOperation op, MathMode mode) const {
    switch (op) {
        case MathOperation::kEvaluate:
        case MathOperation::kNumericEvaluate:
        case MathOperation::kDeterminant:
        case MathOperation::kProbability:
            return true;
        case MathOperation::kSimplify:
        case MathOperation::kFactor:
        case MathOperation::kExpand:
            // Fast numeric supports pure arithmetic simplification / expansion
            return (mode != MathMode::kSymbolic);
        default:
            return false;
    }
}

static int get_precedence(char op) {
    if (op == '+' || op == '-') return 1;
    if (op == '*' || op == '/' || op == '%') return 2;
    if (op == '^') return 3;
    return 0;
}

bool FastNumericBackend::evaluate_arithmetic_exact(const std::string& raw_expr, Rational& out_rat, double& out_dbl, std::string& err) {
    std::string expr = ExpressionParser::canonicalize(raw_expr);
    if (expr.empty()) {
        err = "Empty expression";
        return false;
    }

    // Check for single integer or float
    // Tokenize arithmetic expression with Shunting-yard algorithm
    std::stack<Rational> values;
    std::stack<char> ops;

    auto apply_op = [&values](char op) -> bool {
        if (values.size() < 2) return false;
        Rational val2 = values.top(); values.pop();
        Rational val1 = values.top(); values.pop();

        if (op == '+') values.push(val1 + val2);
        else if (op == '-') values.push(val1 - val2);
        else if (op == '*') values.push(val1 * val2);
        else if (op == '/') {
            if (val2.num == 0) return false;
            values.push(val1 / val2);
        } else if (op == '^') {
            if (val2.den != 1) throw FastPathLimit("non-integer power cannot be represented exactly");
            const bool negative = val2.num < 0;
            const int64_t p = negative ? -val2.num : val2.num;
            int64_t n = 1, d = 1;
            if (val1.den == 1 && (val1.num == 1 || val1.num == 0 || val1.num == -1)) {
                n = (p == 0) ? 1 : (val1.num == -1 ? (p % 2 ? -1 : 1) : val1.num);
            } else {
                if (p > 64) throw FastPathLimit("int64 overflow");
                for (int64_t i = 0; i < p; ++i) {
                    n = mul_ck(n, val1.num);
                    d = mul_ck(d, val1.den);
                }
            }
            if (negative) {
                if (n == 0) return false;   // 0 to a negative power
                values.push(Rational(d, n));
            } else {
                values.push(Rational(n, d));
            }
        }
        return true;
    };

    // Check for special functions like sqrt(...)
    if (expr.find("sqrt(") == 0 && expr.back() == ')') {
        std::string inner = expr.substr(5, expr.length() - 6);
        Rational inner_rat;
        double inner_dbl = 0.0;
        if (!evaluate_arithmetic_exact(inner, inner_rat, inner_dbl, err)) return false;
        double s = std::sqrt(inner_rat.to_double());
        out_dbl = s;
        int64_t is = static_cast<int64_t>(std::round(s));
        if (is * is == inner_rat.num && inner_rat.den == 1) {
            out_rat = Rational(is, 1);
        } else {
            out_rat = Rational(static_cast<int64_t>(s * 1000000.0), 1000000);
        }
        return true;
    }

    {   // grammar check: this evaluator used to skip anything it did not recognize ("2 3" evaluated to 3). Refuse instead; the CAS takes over.
        bool want_operand = true;
        int depth = 0;
        for (size_t i = 0; i < expr.length(); ++i) {
            const char c = expr[i];
            if (c == ' ') {
                // a space between two operands is implicit multiplication, which this evaluator does not do
                size_t j = i;
                while (j < expr.length() && expr[j] == ' ') ++j;
                if (!want_operand && j < expr.length() && (std::isdigit(static_cast<unsigned char>(expr[j])) || expr[j] == '(')) { err = "implicit multiplication"; return false; }
                i = j - 1;
            } else if (std::isdigit(static_cast<unsigned char>(c)) || (c == '.' && i + 1 < expr.length() && std::isdigit(static_cast<unsigned char>(expr[i + 1])))) {
                if (!want_operand) { err = "missing operator"; return false; }
                while (i + 1 < expr.length() && (std::isdigit(static_cast<unsigned char>(expr[i + 1])) || expr[i + 1] == '.')) ++i;
                want_operand = false;
            } else if (c == '(') {
                if (!want_operand) { err = "implicit multiplication"; return false; }
                ++depth;
            } else if (c == ')') {
                if (want_operand || depth == 0) { err = "unbalanced parentheses"; return false; }
                --depth;
            } else if (c == '+' || c == '-') {
                want_operand = true;   // binary or unary
            } else if (c == '*' || c == '/' || c == '^') {
                if (want_operand) { err = "operator without an operand"; return false; }
                want_operand = true;
            } else {
                err = std::string("unsupported character '") + c + "'";
                return false;
            }
        }
        if (want_operand || depth != 0) { err = "incomplete expression"; return false; }
    }

    for (size_t i = 0; i < expr.length(); ++i) {
        if (expr[i] == ' ') continue;

        if (expr[i] == '(') {
            ops.push(expr[i]);
        } else if (std::isdigit(static_cast<unsigned char>(expr[i]))) {
            int64_t val = 0;
            while (i < expr.length() && std::isdigit(static_cast<unsigned char>(expr[i]))) {
                val = add_ck(mul_ck(val, 10), expr[i] - '0');
                i++;
            }
            if (i < expr.length() && expr[i] == '.') {
                i++;
                // decimal literal: exact fraction digits / 10^k (at most 9 digits), never a rounded double
                int64_t frac = 0, scale = 1;
                int digits = 0;
                while (i < expr.length() && std::isdigit(static_cast<unsigned char>(expr[i]))) {
                    if (++digits > 9) throw FastPathLimit("decimal literal too long for the fast path");
                    frac = frac * 10 + (expr[i] - '0');
                    scale *= 10;
                    i++;
                }
                values.push(Rational(add_ck(mul_ck(val, scale), frac), scale));
            } else {
                values.push(Rational(val, 1));
            }
            i--; // Step back for outer loop increment
        } else if (expr[i] == ')') {
            while (!ops.empty() && ops.top() != '(') {
                if (!apply_op(ops.top())) {
                    err = "Evaluation error inside parentheses";
                    return false;
                }
                ops.pop();
            }
            if (!ops.empty()) ops.pop(); // pop '('
        } else if (expr[i] == '+' || expr[i] == '-' || expr[i] == '*' || expr[i] == '/' || expr[i] == '^') {
            // Handle unary minus
            if (expr[i] == '-' && (i == 0 || expr[i - 1] == '(' || expr[i - 1] == '+' || expr[i - 1] == '-' || expr[i - 1] == '*' || expr[i - 1] == '/')) {
                values.push(Rational(0, 1));
            }
            while (!ops.empty() && get_precedence(ops.top()) >= get_precedence(expr[i])) {
                if (!apply_op(ops.top())) {
                    err = "Operator evaluation error";
                    return false;
                }
                ops.pop();
            }
            ops.push(expr[i]);
        }
    }

    while (!ops.empty()) {
        if (!apply_op(ops.top())) {
            err = "Final evaluation error";
            return false;
        }
        ops.pop();
    }

    if (values.size() != 1) {
        err = "malformed expression";
        return false;
    }

    out_rat = values.top();
    out_dbl = out_rat.to_double();
    return true;
}

bool FastNumericBackend::evaluate_determinant(const std::string& expr, double& out_det, std::string& err) {
    // Basic 2x2 or 3x3 determinant: det([[a,b],[c,d]])
    std::regex det2_re(R"lit(\[\[([0-9\.\-]+),([0-9\.\-]+)\],\[([0-9\.\-]+),([0-9\.\-]+)\]\])lit");
    std::smatch m;
    if (std::regex_search(expr, m, det2_re)) {
        double a = std::stod(m[1].str());
        double b = std::stod(m[2].str());
        double c = std::stod(m[3].str());
        double d = std::stod(m[4].str());
        out_det = a * d - b * c;
        return true;
    }
    err = "Matrix determinant syntax not recognized (expected [[a,b],[c,d]])";
    return false;
}

bool FastNumericBackend::evaluate_combinatorics(const std::string& expr, int64_t& out_val, std::string& err) {
    // Combinations C(n, k) or nCk or factorial n!
    std::regex fact_re(R"lit(([0-9]+)!)lit");
    std::regex comb_re(R"lit((?:C|nCr|comb)\(([0-9]+),\s*([0-9]+)\))lit");
    std::regex perm_re(R"lit((?:P|nPr|perm)\(([0-9]+),\s*([0-9]+)\))lit");

    std::smatch m;
    if (std::regex_search(expr, m, fact_re)) {
        int n = std::stoi(m[1].str());
        if (n < 0 || n > 20) {
            throw FastPathLimit("factorial argument out of 64-bit range");
        }
        int64_t res = 1;
        for (int i = 2; i <= n; ++i) res = mul_ck(res, i);
        out_val = res;
        return true;
    } else if (std::regex_search(expr, m, comb_re)) {
        int n = std::stoi(m[1].str());
        int k = std::stoi(m[2].str());
        if (k < 0 || k > n || n > 60) {
            throw FastPathLimit("combination arguments out of fast-path range");
        }
        if (k > n - k) k = n - k;
        int64_t res = 1;
        for (int i = 1; i <= k; ++i) {
            res = mul_ck(res, n - i + 1) / i;
        }
        out_val = res;
        return true;
    } else if (std::regex_search(expr, m, perm_re)) {
        int n = std::stoi(m[1].str());
        int k = std::stoi(m[2].str());
        if (k < 0 || k > n || n > 20) {
            throw FastPathLimit("permutation arguments out of fast-path range");
        }
        int64_t res = 1;
        for (int i = 0; i < k; ++i) {
            res = mul_ck(res, n - i);
        }
        out_val = res;
        return true;
    }

    err = "Combinatorics syntax not matched";
    return false;
}

MathResult FastNumericBackend::execute(const MathRequest& request) {
    try {
        return execute_checked(request);
    } catch (const FastPathLimit& e) {
        MathResult res;
        res.request_id = request.request_id;
        res.backend_type = MathBackendType::kFastNumeric;
        res.backend_name = name();
        res.backend_version = version();
        res.status = MathStatus::kResourceLimitExceeded;
        res.error_message = std::string("fast path: ") + e.what();
        return res;
    }
}

MathResult FastNumericBackend::execute_checked(const MathRequest& request) {
    auto start_time = std::chrono::steady_clock::now();
    MathResult res;
    res.request_id = request.request_id;
    res.backend_type = MathBackendType::kFastNumeric;
    res.backend_name = name();
    res.backend_version = version();

    std::string clean_expr = ExpressionParser::canonicalize(request.expression);
    res.canonical_expression = clean_expr;

    if (request.operation == MathOperation::kDeterminant) {
        double det = 0.0;
        std::string err;
        if (evaluate_determinant(clean_expr, det, err)) {
            res.status = MathStatus::kSuccess;
            res.exact_result = std::to_string(det);
            res.numeric_result = std::to_string(det);
            res.raw_result = res.exact_result;
        } else {
            res.status = MathStatus::kExecutionError;
            res.error_message = err;
        }
    } else if (request.operation == MathOperation::kProbability) {
        int64_t val = 0;
        std::string err;
        if (evaluate_combinatorics(clean_expr, val, err)) {
            res.status = MathStatus::kSuccess;
            res.exact_result = std::to_string(val);
            res.numeric_result = std::to_string(val);
            res.raw_result = res.exact_result;
        } else {
            res.status = MathStatus::kExecutionError;
            res.error_message = err;
        }
    } else {
        Rational rat;
        double dbl = 0.0;
        std::string err;
        if (evaluate_arithmetic_exact(clean_expr, rat, dbl, err)) {
            res.status = MathStatus::kSuccess;
            res.exact_result = rat.to_string();
            std::ostringstream ss;
            ss.precision(request.precision_digits);
            ss << dbl;
            res.numeric_result = ss.str();
            res.raw_result = (request.mode == MathMode::kNumeric) ? res.numeric_result : res.exact_result;
        } else {
            res.status = MathStatus::kExecutionError;
            res.error_message = err;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    res.compact_observation = ExpressionParser::format_compact_observation(
        request.expression, res.exact_result, res.numeric_result, request.operation);
    return res;
}

} // namespace strata::math
