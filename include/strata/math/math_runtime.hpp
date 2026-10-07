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
#include <string>
#include <vector>

namespace strata::math {

class MathRuntime {
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

    // Generic Request Processing via Calculator Router
    MathResult process_request(const MathRequest& request);

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
    MathicsBackend& mathics_backend() { return *mathics_backend_; }
    SageBackend& sage_backend() { return *sage_backend_; }

private:
    std::shared_ptr<context::VirtualContextManager> vctx_;
    MathSecurityLimits limits_;

    std::unique_ptr<ExpressionParser> parser_;
    std::unique_ptr<ExpressionValidator> validator_;
    std::unique_ptr<MathResultCache> cache_;
    std::unique_ptr<FastNumericBackend> fast_backend_;
    std::unique_ptr<MathicsBackend> mathics_backend_;
    std::unique_ptr<SageBackend> sage_backend_;

    mutable MathRuntimeStats stats_;

    IMathBackend* route_backend(const MathRequest& request, uint32_t complexity) const;
    void normalize_result(MathResult& result, const MathRequest& req) const;
    int64_t estimate_tokens(const std::string& text) const;
};

} // namespace strata::math
