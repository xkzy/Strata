// tests/math/test_math_runtime.cpp - Mathematical Runtime & Mathics Engine Test Suite
// The build defines NDEBUG (Release), which turns every assert() into nothing - including the required
// unified.initialize() call. Keep asserts active so this suite checks what it claims to.
#undef NDEBUG
#include "strata/math/expression_parser.hpp"
#include "strata/math/math_backend.hpp"
#include "strata/math/math_cache.hpp"
#include "strata/math/math_runtime.hpp"
#include "strata/math/math_types.hpp"
#include "strata/strata_unified.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

void test_expression_parser_and_validator() {
    std::cout << "[Test 1/8] Testing Mathematical Expression Parser & Security Validator..." << std::endl;
    strata::math::ExpressionValidator validator;
    strata::math::ExpressionParser parser;

    // Test 1: Valid expressions
    std::string err;
    assert(validator.validate("2384 * 7291", err));
    assert(validator.validate("1/3 + 1/6", err));
    assert(validator.validate("differentiate sin(x^2)", err));
    assert(validator.validate("solve x^2 + 5x + 6 == 0", err));
    assert(validator.is_pure_arithmetic("2384 * 7291"));
    assert(validator.is_pure_arithmetic("1/3 + 1/6"));
    assert(!validator.is_pure_arithmetic("sin(x^2)"));

    // Test 2: Security checks / injections
    assert(!validator.validate("__import__('os').system('ls')", err));
    assert(!validator.validate("exec('print(1)')", err));
    assert(!validator.validate("open('/etc/passwd')", err));

    // Test 3: Bracket nesting / unbalanced
    assert(!validator.validate("((1 + 2)", err));
    assert(!validator.validate("1 + 2)", err));

    // Test 4: Canonicalization
    assert(strata::math::ExpressionParser::canonicalize("2,384 * 7,291") == "2384*7291");
    assert(strata::math::ExpressionParser::canonicalize(" 1/3   +   1/6 ") == "1/3+1/6");

    // Test 5: Intent detection
    auto intents = parser.detect_calculation_intents("Please calculate 2384 * 7291 for me.");
    assert(!intents.empty());
    assert(intents[0].operation == strata::math::MathOperation::kEvaluate);

    auto diff_intents = parser.detect_calculation_intents("differentiate sin(x^2)");
    assert(!diff_intents.empty());
    assert(diff_intents[0].operation == strata::math::MathOperation::kDifferentiate);

    std::cout << "  Passed. Expression validation, security checks, and intent detection verified." << std::endl;
}

void test_fast_numeric_exact_rational_arithmetic() {
    std::cout << "[Test 2/8] Testing Fast Numeric Exact Rational Arithmetic..." << std::endl;
    strata::math::FastNumericBackend backend;

    // Test 1: Exact rational addition: 1/3 + 1/6 should equal 1/2, not 0.5
    strata::math::MathRequest req1;
    req1.expression = "1/3 + 1/6";
    req1.mode = strata::math::MathMode::kExact;
    auto res1 = backend.execute(req1);
    assert(res1.status == strata::math::MathStatus::kSuccess);
    assert(res1.exact_result == "1/2");

    // Test 2: Large integer multiplication: 2384 * 7291 = 17381744
    strata::math::MathRequest req2;
    req2.expression = "2384 * 7291";
    auto res2 = backend.execute(req2);
    assert(res2.status == strata::math::MathStatus::kSuccess);
    assert(res2.exact_result == "17381744");

    // Test 3: Sqrt calculation
    strata::math::MathRequest req3;
    req3.expression = "sqrt(123456789)";
    auto res3 = backend.execute(req3);
    assert(res3.status == strata::math::MathStatus::kSuccess);
    assert(!res3.numeric_result.empty());

    // Test 4: Matrix Determinant
    strata::math::MathRequest req4;
    req4.operation = strata::math::MathOperation::kDeterminant;
    req4.expression = "[[1,2],[3,4]]";
    auto res4 = backend.execute(req4);
    assert(res4.status == strata::math::MathStatus::kSuccess);
    // 1*4 - 2*3 = -2
    assert(std::stod(res4.exact_result) == -2.0);

    // Test 5: Combinatorics C(10, 3) = 120
    strata::math::MathRequest req5;
    req5.operation = strata::math::MathOperation::kProbability;
    req5.expression = "C(10, 3)";
    auto res5 = backend.execute(req5);
    assert(res5.status == strata::math::MathStatus::kSuccess);
    assert(res5.exact_result == "120");

    std::cout << "  Passed. Exact rationals, large arithmetic, determinants, and combinatorics verified." << std::endl;
}

void test_symbolic_mathics_operations() {
    std::cout << "[Test 3/8] Testing Symbolic Mathics Operations..." << std::endl;
    strata::math::MathicsBackend backend;

    // Test 1: Differentiation: differentiate sin(x^2) -> 2*x*cos(x^2)
    strata::math::MathRequest req1;
    req1.operation = strata::math::MathOperation::kDifferentiate;
    req1.expression = "sin(x^2)";
    req1.variable = "x";
    auto res1 = backend.execute(req1);
    assert(res1.status == strata::math::MathStatus::kSuccess);
    assert(res1.exact_result == "2*x*cos(x^2)");

    // Test 2: Integration: integrate x^2 sin(x)
    strata::math::MathRequest req2;
    req2.operation = strata::math::MathOperation::kIntegrate;
    req2.expression = "x^2*sin(x)";
    req2.variable = "x";
    auto res2 = backend.execute(req2);
    assert(res2.status == strata::math::MathStatus::kSuccess);
    assert(res2.exact_result.find("sin(x)") != std::string::npos);

    // Test 3: Solve equation: solve x^2 + 5x + 6 == 0
    strata::math::MathRequest req3;
    req3.operation = strata::math::MathOperation::kSolve;
    req3.expression = "x^2 + 5*x + 6 == 0";
    req3.variable = "x";
    auto res3 = backend.execute(req3);
    assert(res3.status == strata::math::MathStatus::kSuccess);
    assert(res3.exact_result.find("-3") != std::string::npos);
    assert(res3.exact_result.find("-2") != std::string::npos);

    // Test 4: Factorization: factor x^2 + 5x + 6 -> (x + 2)*(x + 3)
    strata::math::MathRequest req4;
    req4.operation = strata::math::MathOperation::kFactor;
    req4.expression = "x^2 + 5*x + 6";
    auto res4 = backend.execute(req4);
    assert(res4.status == strata::math::MathStatus::kSuccess);
    assert(res4.exact_result == "(x + 2)*(x + 3)");

    // Test 5: Expansion: expand (x + 2)*(x + 3) -> x^2 + 5*x + 6
    strata::math::MathRequest req5;
    req5.operation = strata::math::MathOperation::kExpand;
    req5.expression = "(x + 2)*(x + 3)";
    auto res5 = backend.execute(req5);
    assert(res5.status == strata::math::MathStatus::kSuccess);
    assert(res5.exact_result == "x^2 + 5*x + 6");

    // Test 6: Limit: limit sin(x)/x as x -> 0 = 1
    strata::math::MathRequest req6;
    req6.operation = strata::math::MathOperation::kLimit;
    req6.expression = "sin(x)/x";
    req6.variable = "x";
    req6.point = "0";
    auto res6 = backend.execute(req6);
    assert(res6.status == strata::math::MathStatus::kSuccess);
    assert(res6.exact_result == "1");

    std::cout << "  Passed. Symbolic differentiation, integration, equation solving, and factoring verified." << std::endl;
}

void test_fast_path_never_wraps() {
    std::cout << "[Test 4b] Fast path overflow falls back to the exact CAS..." << std::endl;
    strata::math::MathRuntime runtime;
    auto big = runtime.evaluate("99999999999 * 99999999999");
    assert(big.status == strata::math::MathStatus::kSuccess);
    assert(big.exact_result == "9999999999800000000001");
    assert(big.backend_type == strata::math::MathBackendType::kMathics);
    auto pw = runtime.evaluate("2^70");
    assert(pw.exact_result == "1180591620717411303424");
    auto neg = runtime.evaluate("2^-2");
    assert(neg.exact_result == "1/4");
    auto frac = runtime.evaluate("0.29 * 100");
    assert(frac.exact_result == "29");
    auto root = runtime.evaluate("2^0.5");
    assert(root.status == strata::math::MathStatus::kSuccess && root.exact_result != "1");
    auto small = runtime.evaluate("12 * 34");
    assert(small.exact_result == "408" && small.backend_type == strata::math::MathBackendType::kFastNumeric);
    std::cout << "  Passed. No silent int64 wraparound." << std::endl;
}

void test_calculator_router() {
    std::cout << "[Test 4/8] Testing Calculator Router & Backend Dispatch..." << std::endl;
    strata::math::MathRuntime runtime;

    // Arithmetic should route to FastNumeric
    auto res1 = runtime.evaluate("1024 * 4096");
    assert(res1.status == strata::math::MathStatus::kSuccess);
    assert(res1.backend_type == strata::math::MathBackendType::kFastNumeric);

    // Symbolic calculus should route to Mathics
    auto res2 = runtime.differentiate("sin(x^2)");
    assert(res2.status == strata::math::MathStatus::kSuccess);
    assert(res2.backend_type == strata::math::MathBackendType::kMathics);

    auto stats = runtime.get_stats();
    assert(stats.fast_path_count == 1);
    assert(stats.mathics_count == 1);

    std::cout << "  Passed. Calculator router successfully dispatches between fast path and CAS backend." << std::endl;
}

void test_result_cache_and_canonicalization() {
    std::cout << "[Test 5/8] Testing Result Caching and Canonicalization..." << std::endl;
    strata::math::MathRuntime runtime;

    // First call: Cache miss
    auto res1 = runtime.evaluate("2384 * 7291");
    assert(res1.status == strata::math::MathStatus::kSuccess);
    assert(!res1.cache_hit);

    // Second call with different whitespace: Cache hit through canonicalization
    auto res2 = runtime.evaluate("  2,384   *   7,291  ");
    assert(res2.status == strata::math::MathStatus::kSuccess);
    assert(res2.cache_hit);
    assert(res2.exact_result == res1.exact_result);

    auto stats = runtime.get_stats();
    assert(stats.cache_hits == 1);
    assert(stats.cache_misses == 1);
    assert(stats.cache_hit_rate() == 0.5);

    std::cout << "  Passed. Result cache and whitespace/comma canonicalization working as expected." << std::endl;
}

void test_verification_loop() {
    std::cout << "[Test 6/8] Testing Verification Loop & Arithmetic Hallucination Detection..." << std::endl;
    strata::math::MathRuntime runtime;

    // Case 1: Matching result from LLM
    std::string correct_llm_text = "The product is 17381744.";
    auto ver1 = runtime.verify_calculation(correct_llm_text, "2384 * 7291");
    assert(ver1.matches);
    assert(ver1.relative_error == 0.0);

    // Case 2: Hallucinated / incorrect arithmetic from LLM
    std::string hallucinated_llm_text = "The product is 17382094.";
    auto ver2 = runtime.verify_calculation(hallucinated_llm_text, "2384 * 7291");
    assert(!ver2.matches);
    assert(!ver2.discrepancy_details.empty());

    auto stats = runtime.get_stats();
    assert(stats.verification_count == 2);
    assert(stats.verification_failures == 1);

    std::cout << "  Passed. Verification loop accurately detects arithmetic correctness and discrepancies." << std::endl;
}

void test_llm_calculation_interception() {
    std::cout << "[Test 7/8] Testing LLM Calculation Interception..." << std::endl;
    strata::math::MathRuntime runtime;

    std::string user_prompt = "Calculate 2384 * 7291 for the simulation.";
    std::vector<strata::math::MathResult> results;
    std::string intercepted = runtime.intercept_and_evaluate(user_prompt, results);

    assert(!results.empty());
    assert(results[0].exact_result == "17381744");
    assert(intercepted.find("[MathResult:") != std::string::npos);

    std::cout << "  Passed. LLM prompt intercepted and augmented with deterministic mathematical result." << std::endl;
}

void test_unified_system_integration() {
    std::cout << "[Test 8/8] Testing Unified Runtime Integration..." << std::endl;
    strata::UnifiedRuntimeOptions opts;
    strata::StrataUnifiedRuntime unified(opts);

    std::string err;
    assert(unified.initialize(err));

    auto res = unified.math_runtime().evaluate("1/3 + 1/6");
    assert(res.status == strata::math::MathStatus::kSuccess);
    assert(res.exact_result == "1/2");

    std::string status = unified.print_full_system_status();
    assert(status.find("Strata Math Runtime Diagnostics") != std::string::npos);

    std::cout << "  Passed. Math runtime integrated with StrataUnifiedRuntime." << std::endl;
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "        STRATA MATHEMATICAL BACKING ENGINE (MATHICS) TEST SUITE       " << std::endl;
    std::cout << "======================================================================" << std::endl;

    test_expression_parser_and_validator();
    test_fast_numeric_exact_rational_arithmetic();
    test_symbolic_mathics_operations();
    test_fast_path_never_wraps();
    test_calculator_router();
    test_result_cache_and_canonicalization();
    test_verification_loop();
    test_llm_calculation_interception();
    test_unified_system_integration();

    std::cout << "======================================================================" << std::endl;
    std::cout << "ALL 8 MATHEMATICAL ENGINE TEST SUITES PASSED SUCCESSFULLY!" << std::endl;
    std::cout << "======================================================================" << std::endl;
    return 0;
}
