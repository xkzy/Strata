// src/math/math_runtime.cpp - Unified Mathematical Runtime Implementation
#include "strata/math/math_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>

namespace strata::math {

MathRuntime::MathRuntime(std::shared_ptr<context::VirtualContextManager> vctx,
                         const MathSecurityLimits& limits)
    : vctx_(std::move(vctx)),
      limits_(limits),
      parser_(std::make_unique<ExpressionParser>()),
      validator_(std::make_unique<ExpressionValidator>(limits)),
      cache_(std::make_unique<MathResultCache>(10000)),
      fast_backend_(std::make_unique<FastNumericBackend>()),
      mathics_backend_(std::make_unique<MathicsBackend>()),
      sage_backend_(std::make_unique<SageBackend>()) {}

MathRuntime::~MathRuntime() = default;

IMathBackend* MathRuntime::route_backend(const MathRequest& request, uint32_t complexity) const {
    // Calculator Router:
    // 1. If SageMath syntax detected (var, dot-methods, matrix brackets, roots, etc.) -> SageBackend
    // 2. If pure arithmetic or simple numeric evaluation with low complexity -> FastNumericBackend
    // 3. If symbolic calculus, equation solving, series, or high complexity -> MathicsBackend / SageBackend
    const std::string& expr = request.expression;
    bool has_sage_syntax = (expr.find("var(") != std::string::npos ||
                            expr.find(".diff(") != std::string::npos ||
                            expr.find(".derivative(") != std::string::npos ||
                            expr.find(".integrate(") != std::string::npos ||
                            expr.find(".integral(") != std::string::npos ||
                            expr.find(".factor(") != std::string::npos ||
                            expr.find(".expand(") != std::string::npos ||
                            expr.find(".simplify(") != std::string::npos ||
                            expr.find(".roots(") != std::string::npos ||
                            expr.find(".det(") != std::string::npos ||
                            expr.find(".inverse(") != std::string::npos ||
                            expr.find(".transpose(") != std::string::npos ||
                            expr.find("matrix([") != std::string::npos ||
                            expr.find("is_prime(") != std::string::npos ||
                            expr.find("euler_phi(") != std::string::npos ||
                            expr.find("fibonacci(") != std::string::npos ||
                            expr.find("power_mod(") != std::string::npos ||
                            expr.find("xgcd(") != std::string::npos);

    if (has_sage_syntax && sage_backend_->is_available() && sage_backend_->supports_operation(request.operation, request.mode)) {
        return sage_backend_.get();
    }

    bool is_arith = validator_->is_pure_arithmetic(request.expression);

    if (request.operation == MathOperation::kEvaluate ||
        request.operation == MathOperation::kNumericEvaluate ||
        request.operation == MathOperation::kDeterminant ||
        request.operation == MathOperation::kProbability) {
        if (is_arith || request.operation == MathOperation::kDeterminant || request.operation == MathOperation::kProbability) {
            if (fast_backend_->supports_operation(request.operation, request.mode)) {
                return fast_backend_.get();
            }
        }
    }

    if (mathics_backend_->is_available() && mathics_backend_->supports_operation(request.operation, request.mode)) {
        return mathics_backend_.get();
    }

    if (sage_backend_->is_available() && sage_backend_->supports_operation(request.operation, request.mode)) {
        return sage_backend_.get();
    }

    return fast_backend_.get();
}

int64_t MathRuntime::estimate_tokens(const std::string& text) const {
    if (text.empty()) return 0;
    // Rough token heuristic: ~4 chars per token
    return std::max<int64_t>(1, static_cast<int64_t>(text.length() + 3) / 4);
}

void MathRuntime::normalize_result(MathResult& result, const MathRequest& req) const {
    result.observation_tokens = estimate_tokens(result.compact_observation);

    if (result.status == MathStatus::kSuccess) {
        if (req.mode == MathMode::kExact && !result.exact_result.empty()) {
            result.raw_result = result.exact_result;
        } else if (req.mode == MathMode::kNumeric && !result.numeric_result.empty()) {
            result.raw_result = result.numeric_result;
        }
    }
}

MathResult MathRuntime::process_request(const MathRequest& request) {
    stats_.total_calculations++;

    // Step 1: Validate Expression
    std::string val_err;
    if (!validator_->validate(request.expression, val_err)) {
        stats_.security_rejections++;
        MathResult res;
        res.request_id = request.request_id;
        res.status = MathStatus::kInvalidExpression;
        res.error_message = val_err;
        res.compact_observation = "[MathError: Invalid expression: " + val_err + "]";
        return res;
    }

    uint32_t complexity = validator_->compute_complexity(request.expression);

    // Step 2: Cache Lookup
    MathResult cached_res;
    if (cache_->get(request, cached_res)) {
        stats_.cache_hits++;
        cached_res.complexity_score = complexity;
        return cached_res;
    }
    stats_.cache_misses++;

    // Step 3: Calculation Routing
    IMathBackend* backend = route_backend(request, complexity);
    if (!backend) {
        MathResult res;
        res.request_id = request.request_id;
        res.status = MathStatus::kUnsupportedOperation;
        res.error_message = "No suitable mathematical backend available for request";
        return res;
    }

    if (backend->backend_type() == MathBackendType::kFastNumeric) {
        stats_.fast_path_count++;
    } else {
        stats_.mathics_count++;
    }

    // Step 4: Execution
    MathResult res = backend->execute(request);
    if (backend->backend_type() == MathBackendType::kFastNumeric && res.status != MathStatus::kSuccess &&
        mathics_backend_->is_available() && mathics_backend_->supports_operation(request.operation, request.mode)) {
        // the int64 fast path could not produce an exact result (overflow, unsupported syntax): the CAS takes over
        stats_.fast_path_count--;
        stats_.mathics_count++;
        res = mathics_backend_->execute(request);
    }
    res.complexity_score = complexity;
    stats_.total_execution_time_ms += res.execution_time_ms;

    // Step 5: Normalization
    normalize_result(res, request);

    // Step 6: Cache Put (if successful)
    if (res.status == MathStatus::kSuccess) {
        cache_->put(request, res);
    }

    // Step 7: Virtual Context Integration (External State Store)
    if (vctx_ && res.status == MathStatus::kSuccess) {
        vctx_->append_tool_result("mathics", request.expression, res.raw_result, 0, res.observation_tokens);
    }

    return res;
}

MathResult MathRuntime::evaluate(const std::string& expression, MathMode mode,
                                 const std::string& tenant_id, const std::string& session_id) {
    MathRequest req;
    req.operation = MathOperation::kEvaluate;
    req.expression = expression;
    req.mode = mode;
    req.tenant_id = tenant_id;
    req.session_id = session_id;
    return process_request(req);
}

MathResult MathRuntime::simplify(const std::string& expression,
                                 const std::string& tenant_id, const std::string& session_id) {
    MathRequest req;
    req.operation = MathOperation::kSimplify;
    req.expression = expression;
    req.tenant_id = tenant_id;
    req.session_id = session_id;
    return process_request(req);
}

MathResult MathRuntime::expand(const std::string& expression,
                               const std::string& tenant_id, const std::string& session_id) {
    MathRequest req;
    req.operation = MathOperation::kExpand;
    req.expression = expression;
    req.tenant_id = tenant_id;
    req.session_id = session_id;
    return process_request(req);
}

MathResult MathRuntime::factor(const std::string& expression,
                               const std::string& tenant_id, const std::string& session_id) {
    MathRequest req;
    req.operation = MathOperation::kFactor;
    req.expression = expression;
    req.tenant_id = tenant_id;
    req.session_id = session_id;
    return process_request(req);
}

MathResult MathRuntime::solve(const std::string& equation, const std::string& var,
                              const std::string& tenant_id, const std::string& session_id) {
    MathRequest req;
    req.operation = MathOperation::kSolve;
    req.expression = equation;
    req.variable = var;
    req.tenant_id = tenant_id;
    req.session_id = session_id;
    return process_request(req);
}

MathResult MathRuntime::differentiate(const std::string& expression, const std::string& var, int order,
                                      const std::string& tenant_id, const std::string& session_id) {
    MathRequest req;
    req.operation = MathOperation::kDifferentiate;
    req.expression = expression;
    req.variable = var;
    req.order = order;
    req.tenant_id = tenant_id;
    req.session_id = session_id;
    return process_request(req);
}

MathResult MathRuntime::integrate(const std::string& expression, const std::string& var,
                                   const std::string& tenant_id, const std::string& session_id) {
    MathRequest req;
    req.operation = MathOperation::kIntegrate;
    req.expression = expression;
    req.variable = var;
    req.tenant_id = tenant_id;
    req.session_id = session_id;
    return process_request(req);
}

MathResult MathRuntime::limit(const std::string& expression, const std::string& var, const std::string& point,
                              const std::string& tenant_id, const std::string& session_id) {
    MathRequest req;
    req.operation = MathOperation::kLimit;
    req.expression = expression;
    req.variable = var;
    req.point = point;
    req.tenant_id = tenant_id;
    req.session_id = session_id;
    return process_request(req);
}

MathResult MathRuntime::series(const std::string& expression, const std::string& var, const std::string& point, int order,
                               const std::string& tenant_id, const std::string& session_id) {
    MathRequest req;
    req.operation = MathOperation::kSeries;
    req.expression = expression;
    req.variable = var;
    req.point = point;
    req.order = order;
    req.tenant_id = tenant_id;
    req.session_id = session_id;
    return process_request(req);
}

MathResult MathRuntime::numeric_evaluate(const std::string& expression, int precision_digits,
                                         const std::string& tenant_id, const std::string& session_id) {
    MathRequest req;
    req.operation = MathOperation::kNumericEvaluate;
    req.expression = expression;
    req.mode = MathMode::kNumeric;
    req.precision_digits = precision_digits;
    req.tenant_id = tenant_id;
    req.session_id = session_id;
    return process_request(req);
}

MathVerificationResult MathRuntime::verify_calculation(const std::string& llm_output, const std::string& expected_expression) {
    stats_.verification_count++;
    MathVerificationResult ver;

    // Ground truth calculation
    MathResult gt = evaluate(expected_expression);
    if (gt.status != MathStatus::kSuccess) {
        ver.matches = false;
        ver.discrepancy_details = "Ground truth evaluation failed: " + gt.error_message;
        stats_.verification_failures++;
        return ver;
    }

    ver.ground_truth_result = gt.exact_result;

    // Extract claimed result from LLM output (e.g. "= 17382094" or "is 17,382,094")
    std::regex res_num_re(R"lit((?:=|\bis\b|\bresult\b|\bequals\b)\s*([0-9\.\,\-\/]+))lit", std::regex::icase);
    std::smatch m;
    std::string claimed;
    if (std::regex_search(llm_output, m, res_num_re)) {
        claimed = ExpressionParser::canonicalize(m[1].str());
    } else {
        claimed = ExpressionParser::canonicalize(llm_output);
    }

    ver.llm_claimed_result = claimed;

    std::string canon_gt = ExpressionParser::canonicalize(gt.exact_result);
    if (claimed == canon_gt) {
        ver.matches = true;
        ver.relative_error = 0.0;
        return ver;
    }

    // Try float comparison
    try {
        double d_claimed = std::stod(claimed);
        double d_gt = std::stod(canon_gt);
        double diff = std::abs(d_claimed - d_gt);
        double denom = std::max(std::abs(d_gt), 1e-9);
        ver.relative_error = diff / denom;
        if (diff < 1e-6) {
            ver.matches = true;
            return ver;
        }
    } catch (...) {
        // Non-numeric comparison
    }

    ver.matches = false;
    ver.discrepancy_details = "LLM output '" + claimed + "' does not match verified mathematical result '" + gt.exact_result + "'";
    stats_.verification_failures++;
    return ver;
}

std::string MathRuntime::intercept_and_evaluate(const std::string& text, std::vector<MathResult>& out_results) {
    auto intents = parser_->detect_calculation_intents(text);
    if (intents.empty()) {
        return text;
    }

    std::string transformed = text;
    for (const auto& intent : intents) {
        MathRequest req;
        req.operation = intent.operation;
        req.expression = intent.expression;
        req.variable = intent.variable;
        req.point = intent.point;
        req.order = intent.order;
        req.mode = intent.mode;

        MathResult res = process_request(req);
        out_results.push_back(res);

        if (res.status == MathStatus::kSuccess) {
            // Replace or append verified observation
            std::string obs = " " + res.compact_observation;
            if (!intent.raw_match.empty()) {
                size_t pos = transformed.find(intent.raw_match);
                if (pos != std::string::npos) {
                    transformed.replace(pos, intent.raw_match.length(), intent.raw_match + obs);
                }
            }
        }
    }

    return transformed;
}

MathRuntimeStats MathRuntime::get_stats() const {
    return stats_;
}

std::string MathRuntime::print_diagnostics() const {
    std::ostringstream ss;
    ss << "=== Strata Math Runtime Diagnostics ===\n";
    ss << "Total Calculations: " << stats_.total_calculations << "\n";
    ss << "Cache Hits: " << stats_.cache_hits << " (" << std::fixed << std::setprecision(1) << (stats_.cache_hit_rate() * 100.0) << "%)\n";
    ss << "Fast Path Calls: " << stats_.fast_path_count << "\n";
    ss << "Mathics Symbolic Calls: " << stats_.mathics_count << "\n";
    ss << "Verifications Run: " << stats_.verification_count << " (Failures: " << stats_.verification_failures << ")\n";
    ss << "Security Rejections: " << stats_.security_rejections << "\n";
    ss << "Avg Execution Time: " << std::fixed << std::setprecision(2) << stats_.avg_execution_time_ms() << " ms\n";
    ss << "Cached Entries: " << cache_->size() << " / " << cache_->max_entries() << "\n";
    ss << "FastNumeric Backend: Available (v" << fast_backend_->version() << ")\n";
    ss << "Mathics3 Backend: " << (mathics_backend_->is_available() ? "Available" : "Disabled") << " (v" << mathics_backend_->version() << ")\n";
    ss << "SageMath Backend: " << (sage_backend_->is_available() ? "Available" : "Disabled") << " (v" << sage_backend_->version() << ")\n";
    return ss.str();
}

} // namespace strata::math
