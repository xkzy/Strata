// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/mathics_backend.cpp - MathicsBackend on top of the native CAS (src/math/cas)
#include "strata/math/cas/engine.hpp"
#include "strata/math/expression_parser.hpp"
#include "strata/math/math_backend.hpp"

#include <cctype>
#include <chrono>
#include <cstdio>

namespace strata::math {

namespace {

std::string format_double(double v, int digits) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*g", digits < 1 ? 15 : (digits > 17 ? 17 : digits), v);
    return buf;
}

// The variable a calculus/solve request refers to: the explicit one, else x if present, else the only symbol.
std::string pick_variable(const MathRequest& request, const cas::Expr& e) {
    if (!request.variable.empty()) return request.variable;
    std::vector<std::string> syms;
    cas::free_symbols(e, syms);
    if (syms.empty()) return "x";
    for (const auto& s : syms) if (s == "x") return "x";
    if (syms.size() == 1) return syms[0];
    throw cas::CasUnsupported("ambiguous variable: specify which of " + std::to_string(syms.size()) + " symbols to use");
}

// Solve results are a list of solution sets. One variable prints compactly as {x -> -3, x -> -2}; several variables
// print as {{x -> 1, y -> 2}}.
std::string format_result(const cas::Expr& out, MathOperation op) {
    if (op == MathOperation::kSolve && out->has_head("List")) {
        bool single = true;
        for (const auto& set : out->args) if (!set->has_head("List") || set->args.size() != 1) single = false;
        if (single && !out->args.empty()) {
            std::string s = "{";
            for (size_t i = 0; i < out->args.size(); ++i) s += (i ? ", " : "") + cas::to_string(out->args[i]->args[0]);
            return s + "}";
        }
    }
    return cas::to_string(out);
}

} // namespace

MathicsBackend::MathicsBackend(bool mock_mode) : mock_mode_(mock_mode) {}

bool MathicsBackend::supports_operation(MathOperation op, MathMode mode) const {
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

MathResult MathicsBackend::execute(const MathRequest& request) {
    auto start_time = std::chrono::steady_clock::now();
    MathResult res;
    res.request_id = request.request_id;
    res.backend_type = MathBackendType::kMathics;
    res.backend_name = name();
    res.backend_version = version();
    res.canonical_expression = ExpressionParser::canonicalize(request.expression);

    try {
        cas::Budget budget;
        if (request.timeout_ms > 0) budget.timeout_ms = request.timeout_ms;
        cas::Engine engine(budget);
        cas::Expr e = cas::parse(request.expression);

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
                // definite integral when `point` holds "a,b"
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
                // a bare matrix {{..},{..}} means its determinant
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
                                          "' is not implemented in the native CAS");
        }

        res.exact_result = format_result(out, request.operation);
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
        res.compact_observation = "[MathError: " + res.error_message + "]";
    }
    return res;
}

} // namespace strata::math
