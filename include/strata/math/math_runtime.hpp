// include/strata/math/math_runtime.hpp - Unified Mathematical Runtime
//
// Core abstraction integrating Mathics3 CAS, FastNumeric native execution,
// calculator routing, validation, verification loops, and bounded context observations.
#pragma once

#include "strata/context/virtual_context.hpp"
#include "strata/math/expression_parser.hpp"
#include "strata/math/math_backend.hpp"
#include "strata/math/math_cache.hpp"
#include "strata/math/math_types.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::math {

// What running a request is expected to cost (heuristic, deterministic): the scheduler uses it to decide whether
// running independent requests in parallel is worth the threads. Not a measurement.
struct MathCost {
    double cpu_units = 0;          // relative work; 1.0 is about a millisecond of one core
    uint64_t memory_bytes = 0;
    double latency_ms = 0;         // estimate on one core
    bool parallelizable = true;    // independent of other requests
    double precision_cost = 1.0;   // multiplier from the requested digits
    bool cached = false;           // an exact cache hit costs next to nothing
    std::string backend;           // the backend the router would pick
};

class MathRuntime {
public:
public:
    explicit MathRuntime(std::shared_ptr<context::VirtualContextManager> vctx = nullptr,
                         const MathSecurityLimits& limits = MathSecurityLimits());
    ~MathRuntime();

    // High-level Mathematical Operations
    MathResult evaluate(const std::string& expression, MathMode mode = MathMode::kExact,
                        const std::string& tenant_id = "default_tenant",
                        const std::string& session_id = "default_session");

    MathResult simplify(const std::string& expression,
                         const std::string& tenant_id = "default_tenant",
                         const std::string& session_id = "default_session");

    MathResult expand(const std::string& expression,
                       const std::string& tenant_id = "default_tenant",
                       const std::string& session_id = "default_session");

    MathResult factor(const std::string& expression,
                       const std::string& tenant_id = "default_tenant",
                       const std::string& session_id = "default_session");

    MathResult solve(const std::string& equation, const std::string& var = "x",
                      const std::string& tenant_id = "default_tenant",
                      const std::string& session_id = "default_session");

    MathResult differentiate(const std::string& expression, const std::string& var = "x", int order = 1,
                             const std::string& tenant_id = "default_tenant",
                             const std::string& session_id = "default_session");

    MathResult integrate(const std::string& expression, const std::string& var = "x",
                          const std::string& tenant_id = "default_tenant",
                          const std::string& session_id = "default_session");

    MathResult limit(const std::string& expression, const std::string& var = "x", const std::string& point = "0",
                      const std::string& tenant_id = "default_tenant",
                      const std::string& session_id = "default_session");

    MathResult series(const std::string& expression, const std::string& var = "x", const std::string& point = "0", int order = 6,
                       const std::string& tenant_id = "default_tenant",
                       const std::string& session_id = "default_session");

    MathResult numeric_evaluate(const std::string& expression, int precision_digits = 15,
                                const std::string& tenant_id = "default_tenant",
                                const std::string& session_id = "default_session");

    // Generic Request Processing via Calculator Router (thread-safe)
    MathResult process_request(const MathRequest& request);

    // Estimated cost of a request (see MathCost); cheap, no evaluation.
    MathCost estimate_cost(const MathRequest& request) const;
    // Runs independent requests, in the order given. Identical requests (same cache key) are computed once. Runs in parallel only when the
    // estimated total work is large enough to repay the threads; max_threads 0 = the configured limit.
    std::vector<MathResult> process_batch(const std::vector<MathRequest>& requests, size_t max_threads = 0);

    // Verification Loop: Verifies whether an LLM generation's arithmetic matches deterministic computation
    MathVerificationResult verify_calculation(const std::string& llm_output, const std::string& expected_expression);

    // LLM Calculation Interception: Scans text, executes mathematical expressions, and returns annotated string or observations
    std::string intercept_and_evaluate(const std::string& text, std::vector<MathResult>& out_results);

    // Diagnostics & Observability
    MathRuntimeStats get_stats() const;
    std::string print_diagnostics() const;

    // Subsystem Accessors
    ExpressionParser& parser() { return *parser_; }
    ExpressionValidator& validator() { return *validator_; }
    MathResultCache& cache() { return *cache_; }
    FastNumericBackend& fast_backend() { return *fast_backend_; }
    UnifiedCasBackend& cas_backend() { return *cas_backend_; }
    UnifiedCasBackend& mathics_backend() { return *cas_backend_; }
    UnifiedCasBackend& sage_backend() { return *cas_backend_; }

private:
    std::shared_ptr<context::VirtualContextManager> vctx_;
    MathSecurityLimits limits_;

    std::unique_ptr<ExpressionParser> parser_;
    std::unique_ptr<ExpressionValidator> validator_;
    std::unique_ptr<MathResultCache> cache_;
    std::unique_ptr<FastNumericBackend> fast_backend_;
    std::unique_ptr<UnifiedCasBackend> cas_backend_;

    mutable MathRuntimeStats stats_;
    mutable std::mutex stats_mu_;
    std::mutex vctx_mu_;

    IMathBackend* route_backend(const MathRequest& request, uint32_t complexity) const;
    void normalize_result(MathResult& result, const MathRequest& req) const;
    int64_t estimate_tokens(const std::string& text) const;
};

} // namespace strata::math
