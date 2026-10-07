// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/sage_backend.cpp - SageMath Execution Backend
//
// Implements SageMath syntax translation, operations, and evaluation on top of
// the native CAS engine with Python/Sage semantics.
#include "strata/math/sage_backend.hpp"
#include "strata/math/cas/engine.hpp"
#include "strata/math/expression_parser.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <regex>
#include <sstream>

namespace strata::math {

namespace {

std::string format_double(double v, int digits) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*g", digits < 1 ? 15 : (digits > 17 ? 17 : digits), v);
    return buf;
}

std::string pick_variable(const MathRequest& request, const cas::Expr& e) {
    if (!request.variable.empty()) return request.variable;
    std::vector<std::string> syms;
    cas::free_symbols(e, syms);
    if (syms.empty()) return "x";
    for (const auto& s : syms) if (s == "x") return "x";
    if (syms.size() == 1) return syms[0];
    return "x";
}

// {{x -> 1}, {x -> 2}} (what roots / Solve return)
bool is_solution_set(const cas::Expr& e) {
    if (!e->has_head("List") || e->args.empty()) return false;
    for (const auto& set : e->args) {
        if (!set->has_head("List") || set->args.empty()) return false;
        for (const auto& r : set->args) if (!r->has_head("Rule", 2)) return false;
    }
    return true;
}

// Format result in SageMath Python style
std::string format_sage_result(const cas::Expr& out, MathOperation op) {
    if (out->has_head("Symbol")) {
        if (out->name == "True") return "True";
        if (out->name == "False") return "False";
    }

    if ((op == MathOperation::kSolve || is_solution_set(out)) && out->has_head("List")) {
        // Format as [x == a, x == b]
        std::string s = "[";
        for (size_t i = 0; i < out->args.size(); ++i) {
            const auto& set = out->args[i];
            if (set->has_head("List") && !set->args.empty()) {
                for (size_t j = 0; j < set->args.size(); ++j) {
                    const auto& rule = set->args[j];
                    if (rule->has_head("Rule") && rule->args.size() == 2) {
                        if (i > 0 || j > 0) s += ", ";
                        s += cas::to_string(rule->args[0]) + " == " + cas::to_string(rule->args[1]);
                    } else {
                        if (i > 0 || j > 0) s += ", ";
                        s += cas::to_string(rule);
                    }
                }
            } else if (set->has_head("Rule") && set->args.size() == 2) {
                if (i > 0) s += ", ";
                s += cas::to_string(set->args[0]) + " == " + cas::to_string(set->args[1]);
            } else {
                if (i > 0) s += ", ";
                s += cas::to_string(set);
            }
        }
        return s + "]";
    }

    if (out->has_head("List")) {
        // Convert list {a, b} to [a, b]
        std::string s = "[";
        for (size_t i = 0; i < out->args.size(); ++i) {
            if (i > 0) s += ", ";
            s += format_sage_result(out->args[i], op);
        }
        return s + "]";
    }

    std::string str = cas::to_string(out);
    return str;
}


// Sage calls whose argument shape differs from the CAS's: rewritten by a small bracket-aware pass (no regex, bounded).
constexpr size_t kMaxSageInput = 4096;   // like the expression parser: translation is not meant for documents
constexpr int kMaxRewrites = 64;

bool ident_char(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

// Splits "a, b(c, d), [e, f]" at top-level commas. False when the brackets do not balance.
bool split_args(const std::string& s, std::vector<std::string>& out) {
    int depth = 0;
    std::string cur;
    for (char c : s) {
        if (c == '(' || c == '[' || c == '{') ++depth;
        else if (c == ')' || c == ']' || c == '}') { if (--depth < 0) return false; }
        if (c == ',' && depth == 0) { out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    if (depth != 0) return false;
    if (!cur.empty() || !out.empty()) out.push_back(cur);
    for (auto& a : out) {
        a.erase(0, a.find_first_not_of(" \t\n\r"));
        a.erase(a.find_last_not_of(" \t\n\r") + 1);
    }
    return true;
}

std::string rewrite_one(const std::string& name, const std::vector<std::string>& a, bool& changed) {
    auto call = [&](const std::string& n, const std::string& args) { changed = true; return n + "(" + args + ")"; };
    if ((name == "integral" || name == "integrate") && a.size() == 4)          // integral(f, x, a, b)
        return call("Integrate", a[0] + ", {" + a[1] + ", " + a[2] + ", " + a[3] + "}");
    if (name == "limit" && a.size() == 2) {                                    // limit(f, x=a); a one-sided limit is not rewritten
        const size_t eq = a[1].find('=');                                      // (the CAS form here is two-sided; never answer a different question)
        if (eq != std::string::npos && eq + 1 < a[1].size() && a[1][eq + 1] != '=')
            return call("Limit", a[0] + ", " + a[1].substr(0, eq) + " -> " + a[1].substr(eq + 1));
    }
    if (name == "taylor" && a.size() == 4) return call("Series", a[0] + ", {" + a[1] + ", " + a[2] + ", " + a[3] + "}");   // taylor(f, x, a, n)
    if (name == "sigma" && (a.size() == 1 || a.size() == 2))                  // Sage sigma(n, k); the CAS takes (k, n)
        return call("DivisorSigma", (a.size() == 2 ? a[1] : std::string("1")) + ", " + a[0]);
    return "";
}

// Rewrites the outermost matching call first, then the calls inside its arguments.
std::string rewrite_calls(const std::string& text, int& budget, int depth = 0) {
    if (depth > 16) return text;
    std::string out;
    size_t i = 0;
    while (i < text.size()) {
        if (!ident_char(text[i]) || (i > 0 && (ident_char(text[i - 1]) || text[i - 1] == '.'))) { out.push_back(text[i++]); continue; }
        size_t j = i;
        while (j < text.size() && ident_char(text[j])) ++j;
        const std::string name = text.substr(i, j - i);
        if (j < text.size() && text[j] == '(' && budget > 0) {
            int d = 0;
            size_t k = j;
            for (; k < text.size(); ++k) {
                if (text[k] == '(') ++d;
                else if (text[k] == ')' && --d == 0) break;
            }
            if (k < text.size()) {
                std::vector<std::string> args;
                if (split_args(text.substr(j + 1, k - j - 1), args)) {
                    for (auto& a : args) a = rewrite_calls(a, budget, depth + 1);
                    bool changed = false;
                    std::string r = rewrite_one(name, args, changed);
                    --budget;
                    if (changed) { out += r; i = k + 1; continue; }
                    std::string joined;
                    for (size_t n = 0; n < args.size(); ++n) joined += (n ? ", " : "") + args[n];
                    out += name + "(" + joined + ")";
                    i = k + 1;
                    continue;
                }
            }
        }
        out += name;
        i = j;
    }
    return out;
}

} // namespace

SageBackend::SageBackend(bool mock_mode) : mock_mode_(mock_mode) {}

bool SageBackend::supports_operation(MathOperation op, MathMode mode) const {
    (void)mode;
    switch (op) {
        case MathOperation::kEvaluate:
        case MathOperation::kNumericEvaluate:
        case MathOperation::kSimplify:
        case MathOperation::kExpand:
        case MathOperation::kDifferentiate:
        case MathOperation::kFactor:
        case MathOperation::kSolve:
        case MathOperation::kIntegrate:
        case MathOperation::kLimit:
        case MathOperation::kSeries:
        case MathOperation::kMatrixOp:
        case MathOperation::kDeterminant:
        case MathOperation::kProbability:
            return true;
        default:
            return false;
    }
}

std::string SageBackend::translate_sage_syntax(const std::string& sage_expr) {
    if (sage_expr.empty()) return "";
    if (sage_expr.size() > kMaxSageInput) throw cas::CasParseError("expression too long for SageMath translation (" + std::to_string(kMaxSageInput) + " characters at most)");

    std::string text = sage_expr;

    // 1. Strip leading variable definitions like "var('x')", "x = var('x')", "x, y = var('x y')", "var('x, y')"
    static const std::regex var_decl_regex(
        R"((?:(?:[a-zA-Z_][a-zA-Z0-9_,\s]*=)?\s*var\s*\(\s*['"][^'"]*['"]\s*\)\s*(?:;|\n)?))",
        std::regex::optimize);
    text = std::regex_replace(text, var_decl_regex, "");

    // 2. Strip PolynomialRing declarations like "R.<x> = PolynomialRing(QQ)" or "R.<x, y> = QQ[]"
    static const std::regex ring_decl_regex(
        R"((?:[a-zA-Z_][a-zA-Z0-9_]*\.<[^>]+>\s*=\s*(?:PolynomialRing\([^)]+\)|[a-zA-Z0-9_]+\[[^\]]+\])\s*(?:;|\n)?))",
        std::regex::optimize);
    text = std::regex_replace(text, ring_decl_regex, "");

    // Trim whitespace
    text.erase(0, text.find_first_not_of(" \t\n\r"));
    text.erase(text.find_last_not_of(" \t\n\r") + 1);

    // 3. Translate dot-method syntax:
    // "expr.diff(x)" -> "diff(expr, x)"
    // "expr.factor()" -> "factor(expr)"
    // "expr.integrate(x)" -> "integrate(expr, x)"
    // "expr.expand()" -> "expand(expr)"
    // "expr.simplify()" -> "simplify(expr)"
    // "expr.roots(x)" -> "solve(expr == 0, x)"
    // "M.det()" -> "det(M)"
    // "M.inverse()" -> "inverse(M)"
    // "M.transpose()" -> "transpose(M)"
    static const std::regex dot_method_regex(
        R"(\(?([a-zA-Z0-9_\+\-\*\/\^\s\(\)]+?)\)?\.(diff|derivative|integrate|integral|factor|expand|simplify|roots|det|inverse|transpose)\s*\(([^)]*)\))",
        std::regex::optimize);

    std::smatch match;
    for (int rounds = 0; std::regex_search(text, match, dot_method_regex); ++rounds) {
        if (rounds >= kMaxRewrites) throw cas::CasParseError("too many chained SageMath method calls");
        std::string target = match[1].str();
        std::string method = match[2].str();
        std::string args = match[3].str();

        std::string replacement;
        if (method == "roots") {
            std::string var = args.empty() ? "x" : args;
            replacement = "solve(" + target + " == 0, " + var + ")";
        } else if (args.empty()) {
            replacement = method + "(" + target + ")";
        } else {
            replacement = method + "(" + target + ", " + args + ")";
        }
        text.replace(match.position(0), match.length(0), replacement);
    }

    // 4. Translate "matrix([[...]])" to "[[...]]"
    static const std::regex matrix_regex(
        R"(\bmatrix\s*\(\s*(\[\[[\s\S]*?\]\])\s*\))",
        std::regex::optimize);
    text = std::regex_replace(text, matrix_regex, "$1");

    int budget = kMaxRewrites;
    return rewrite_calls(text, budget);
}

MathResult SageBackend::execute(const MathRequest& request) {
    auto start_time = std::chrono::steady_clock::now();
    MathResult res;
    res.request_id = request.request_id;
    res.backend_type = MathBackendType::kSageMath;
    res.backend_name = name();
    res.backend_version = version();

    std::string translated;
    try {
        translated = translate_sage_syntax(request.expression);
        if (translated.empty()) translated = request.expression;
        if (translated.size() > kMaxSageInput) throw cas::CasParseError("expression too long for SageMath translation");
        res.canonical_expression = ExpressionParser::canonicalize(translated);

        cas::Budget budget;
        if (request.timeout_ms > 0) budget.timeout_ms = request.timeout_ms;
        cas::Engine engine(budget);
        cas::Expr e = cas::parse(translated);

        cas::Expr out;
        switch (request.operation) {
            case MathOperation::kEvaluate:
            case MathOperation::kNumericEvaluate:
                out = engine.eval(e);
                break;
            case MathOperation::kSimplify:
                out = engine.simplify(e);
                break;
            case MathOperation::kExpand:
                out = engine.expand(e);
                break;
            case MathOperation::kDifferentiate: {
                std::string var = pick_variable(request, e);
                out = e;
                for (int i = 0; i < (request.order < 1 ? 1 : request.order); ++i) out = engine.diff(out, var);
                break;
            }
            case MathOperation::kFactor:
                out = engine.factor(e);
                break;
            case MathOperation::kIntegrate: {
                std::string var = pick_variable(request, e);
                size_t comma = request.point.find(',');
                if (comma != std::string::npos) {
                    out = engine.integrate_definite(e, var, cas::parse(request.point.substr(0, comma)), cas::parse(request.point.substr(comma + 1)));
                } else {
                    out = engine.integrate(e, var);
                }
                break;
            }
            case MathOperation::kLimit: {
                std::string var = pick_variable(request, e);
                std::string pt = request.point.empty() ? "0" : request.point;
                while (!pt.empty() && std::isspace(static_cast<unsigned char>(pt.back()))) pt.pop_back();
                int dir = 0;
                if (pt.size() > 1 && (pt.back() == '+' || pt.back() == '-')) { dir = pt.back() == '+' ? 1 : -1; pt.pop_back(); }
                out = engine.limit(e, var, cas::parse(pt), dir);
                break;
            }
            case MathOperation::kSeries: {
                std::string var = pick_variable(request, e);
                int n = request.order >= 1 ? request.order : 5;
                out = engine.series(e, var, cas::parse(request.point.empty() ? "0" : request.point), n);
                break;
            }
            case MathOperation::kMatrixOp:
                out = engine.eval(e);
                break;
            case MathOperation::kDeterminant:
                out = e->has_head("List") ? engine.eval(cas::app("Det", {e})) : engine.eval(e);
                break;
            case MathOperation::kProbability:
                out = engine.eval(e);
                break;
            case MathOperation::kSolve: {
                std::vector<std::string> vars;
                if (!request.variable.empty()) {
                    std::string cur;
                    for (char c : request.variable + ",") {
                        if (c == ',' || c == ' ' || c == ';') { if (!cur.empty()) vars.push_back(cur); cur.clear(); }
                        else cur.push_back(c);
                    }
                }
                out = engine.solve(e, vars);
                break;
            }
            default:
                throw cas::CasUnsupported(std::string("operation '") + math_operation_to_string(request.operation) +
                                          "' is not implemented in SageMath backend");
        }

        res.exact_result = format_sage_result(out, request.operation);
        res.raw_result = res.exact_result;
        double v = 0.0;
        if (engine.numeric_value(out, v)) res.numeric_result = format_double(v, request.precision_digits);
        res.status = MathStatus::kSuccess;
    } catch (const cas::CasLimitError& ex) {
        res.status = MathStatus::kResourceLimitExceeded;
        res.error_message = ex.what();
    } catch (const cas::CasParseError& ex) {
        res.status = MathStatus::kInvalidExpression;
        res.error_message = ex.what();
    } catch (const cas::CasUnsupported& ex) {
        res.status = MathStatus::kUnsupportedOperation;
        res.error_message = ex.what();
    } catch (const cas::CasError& ex) {
        res.status = MathStatus::kExecutionError;
        res.error_message = ex.what();
    }

    auto end_time = std::chrono::steady_clock::now();
    res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    if (res.status == MathStatus::kSuccess) {
        res.compact_observation = ExpressionParser::format_compact_observation(
            request.expression, res.exact_result, res.numeric_result, request.operation);
    } else {
        res.compact_observation = "[SageMathError: " + res.error_message + "]";
    }
    return res;
}

} // namespace strata::math
