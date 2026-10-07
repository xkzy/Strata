// src/math/math_runtime.cpp - Unified Mathematical Runtime Implementation
#include "strata/math/math_runtime.hpp"

#include <thread>
#include <unordered_map>
#include <atomic>

#include <chrono>

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
      cas_backend_(std::make_unique<UnifiedCasBackend>()) {}

MathRuntime::~MathRuntime() = default;

IMathBackend* MathRuntime::route_backend(const MathRequest& request, uint32_t complexity) const {
    (void)complexity;
    // Calculator Router:
    // 1. If pure arithmetic or simple numeric evaluation with low complexity -> FastNumericBackend
    // 2. All other symbolic, algebraic, matrix, number theory, or Sage/Wolfram requests -> UnifiedCasBackend
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

    if (cas_backend_->is_available() && cas_backend_->supports_operation(request.operation, request.mode)) {
        return cas_backend_.get();
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
    { std::lock_guard<std::mutex> g(stats_mu_); stats_.total_calculations++; }

    // Step 1: Validate Expression
    std::string val_err;
    if (!validator_->validate(request.expression, val_err)) {
        { std::lock_guard<std::mutex> g(stats_mu_); stats_.security_rejections++; }
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
        { std::lock_guard<std::mutex> g(stats_mu_); stats_.cache_hits++; }
        cached_res.complexity_score = complexity;
        return cached_res;
    }
    { std::lock_guard<std::mutex> g(stats_mu_); stats_.cache_misses++; }

    // Step 3: Calculation Routing
    IMathBackend* backend = route_backend(request, complexity);
    if (!backend) {
        MathResult res;
        res.request_id = request.request_id;
        res.status = MathStatus::kUnsupportedOperation;
        res.error_message = "No suitable mathematical backend available for request";
        return res;
    }

    {
        std::lock_guard<std::mutex> g(stats_mu_);
        if (backend->backend_type() == MathBackendType::kFastNumeric) stats_.fast_path_count++;
        else stats_.mathics_count++;
    }

    // Step 4: Execution
    MathResult res = backend->execute(request);
    if (backend->backend_type() == MathBackendType::kFastNumeric && res.status != MathStatus::kSuccess &&
        cas_backend_->is_available() && cas_backend_->supports_operation(request.operation, request.mode)) {
        // the int64 fast path could not produce an exact result (overflow, unsupported syntax): the CAS takes over
        { std::lock_guard<std::mutex> g(stats_mu_); stats_.fast_path_count--; stats_.mathics_count++; }
        res = cas_backend_->execute(request);
    }
    res.complexity_score = complexity;
    { std::lock_guard<std::mutex> g(stats_mu_); stats_.total_execution_time_ms += res.execution_time_ms; }

    // Step 5: Normalization
    normalize_result(res, request);

    // Provenance: what produced this result and under which assumptions (reproducibility, cache correctness)
    {
        MathProvenance& pv = res.provenance;
        pv.normalized_expression = res.canonical_expression.empty() ? ExpressionParser::canonicalize(request.expression) : res.canonical_expression;
        pv.operation = math_operation_to_string(request.operation);
        pv.backend = res.backend_name;
        pv.backend_version = res.backend_version;
        pv.algorithm = res.backend_type == MathBackendType::kFastNumeric ? "int64 rational fast path (overflow-checked)" : "exact CAS evaluation";
        pv.assumptions = request.assumptions;
        pv.precision_digits = request.mode == MathMode::kNumeric ? request.precision_digits : 0;
        pv.exact = request.mode != MathMode::kNumeric;
        pv.ir_version = MathResultCache::ir_version();
        pv.input_hashes = {MathResultCache::content_hash(request.expression)};
        pv.expression_hash = MathResultCache::content_hash(MathResultCache::make_cache_key(request));
        pv.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }

    // Step 6: Cache Put (if successful)
    if (res.status == MathStatus::kSuccess) {
        cache_->put(request, res);
    }

    // Step 7: Virtual Context Integration (External State Store)
    if (vctx_ && res.status == MathStatus::kSuccess) {
        std::lock_guard<std::mutex> g(vctx_mu_);
        vctx_->append_tool_result("mathics", request.expression, res.raw_result, 0, res.observation_tokens);
    }

    return res;
}

MathCost MathRuntime::estimate_cost(const MathRequest& request) const {
    MathCost c;
    const uint32_t complexity = validator_->compute_complexity(request.expression);
    IMathBackend* b = route_backend(request, complexity);
    c.backend = b ? b->name() : "none";
    // work grows with the expression and with the operation; a matrix of n x n entries costs about n^3
    double op = 1.0;
    switch (request.operation) {
        case MathOperation::kEvaluate: op = 1.0; break;
        case MathOperation::kSimplify: case MathOperation::kExpand: op = 3.0; break;
        case MathOperation::kDifferentiate: op = 2.0; break;
        case MathOperation::kFactor: op = 6.0; break;
        case MathOperation::kSolve: case MathOperation::kSeries: op = 8.0; break;
        case MathOperation::kIntegrate: case MathOperation::kLimit: op = 12.0; break;
        case MathOperation::kMatrixOp: case MathOperation::kDeterminant: op = 5.0; break;
        default: op = 2.0; break;
    }
    const bool fast = b && b->backend_type() == MathBackendType::kFastNumeric;
    c.precision_cost = request.mode == MathMode::kNumeric ? 1.0 + std::pow(std::max(1, request.precision_digits) / 15.0, 1.6) : 1.0;
    c.cpu_units = (fast ? 0.001 : 0.05) * (1.0 + complexity / 50.0) * op * c.precision_cost;
    c.latency_ms = c.cpu_units;
    c.memory_bytes = static_cast<uint64_t>(request.expression.size()) * 64 + (fast ? 4096 : 262144);
    MathResult probe;
    c.cached = cache_ && const_cast<MathResultCache&>(*cache_).get(request, probe);
    if (c.cached) { c.cpu_units = 0.0005; c.latency_ms = 0.0005; }
    return c;
}

std::vector<MathResult> MathRuntime::process_batch(const std::vector<MathRequest>& requests, size_t max_threads) {
    std::vector<MathResult> out(requests.size());
    if (requests.empty()) return out;
    constexpr size_t kMaxBatch = 10000;   // a caller cannot make one call allocate and schedule unbounded work
    if (requests.size() > kMaxBatch) {
        for (size_t i = 0; i < out.size(); ++i) {
            out[i].request_id = requests[i].request_id;
            out[i].status = MathStatus::kResourceLimitExceeded;
            out[i].error_message = "batch too large (at most " + std::to_string(kMaxBatch) + " requests per call)";
            out[i].compact_observation = "[MathError: " + out[i].error_message + "]";
        }
        return out;
    }
    // identical requests are computed once (request-local cache)
    std::vector<size_t> unique;               // index of the first request with each key
    std::vector<size_t> first_of(requests.size());
    {
        std::unordered_map<std::string, size_t> seen;
        for (size_t i = 0; i < requests.size(); ++i) {
            auto key = MathResultCache::make_cache_key(requests[i]);
            auto it = seen.find(key);
            if (it == seen.end()) { seen[key] = i; unique.push_back(i); first_of[i] = i; }
            else first_of[i] = it->second;
        }
    }
    double total = 0;
    std::vector<double> cost(requests.size(), 0.0);
    for (size_t i : unique) { cost[i] = estimate_cost(requests[i]).cpu_units; total += cost[i]; }
    // never more threads than the configured concurrency limit, whatever the caller asks for
    const size_t cap = std::max<size_t>(1, std::min<size_t>(limits_.max_concurrent_evaluations, std::max(1u, std::thread::hardware_concurrency())));
    size_t threads = max_threads ? std::min(max_threads, cap) : cap;
    threads = std::min(threads, unique.size());
    // threads cost about 0.1 ms each to start: run in parallel only when the work is clearly larger than that
    if (threads <= 1 || total < 0.4) {
        for (size_t i : unique) out[i] = process_request(requests[i]);
    } else {
        std::sort(unique.begin(), unique.end(), [&](size_t a, size_t b) { return cost[a] > cost[b]; });   // largest first
        std::atomic<size_t> next{0};
        auto worker = [&]() {
            while (true) {
                const size_t k = next.fetch_add(1);
                if (k >= unique.size()) return;
                out[unique[k]] = process_request(requests[unique[k]]);
            }
        };
        std::vector<std::thread> pool;
        for (size_t t = 0; t + 1 < threads; ++t) pool.emplace_back(worker);
        worker();
        for (auto& t : pool) t.join();
    }
    for (size_t i = 0; i < requests.size(); ++i) if (first_of[i] != i) { out[i] = out[first_of[i]]; out[i].request_id = requests[i].request_id; }
    return out;
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
    ss << "Unified CAS Backend (Mathics3 + SageMath): " << (cas_backend_->is_available() ? "Available" : "Disabled") << " (v" << cas_backend_->version() << ")\n";
    return ss.str();
}

} // namespace strata::math
