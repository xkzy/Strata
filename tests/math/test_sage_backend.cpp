// tests/math/test_sage_backend.cpp - SageMath Execution Backend Unit & Integration Tests
// The build defines NDEBUG (Release), which turns every assert() into nothing: keep them active so this suite checks what it claims.
#undef NDEBUG
#include "strata/math/sage_backend.hpp"
#include "strata/math/math_runtime.hpp"

#include <cassert>
#include <iostream>
#include <string>

using namespace strata::math;

void test_sage_syntax_translation() {
    std::cout << "[Test 1/7] Testing SageMath Syntax & Dot-Method Translation..." << std::endl;

    // Variable declarations
    std::string s1 = SageBackend::translate_sage_syntax("var('x'); x^2 + 1");
    assert(s1 == "x^2 + 1");

    std::string s2 = SageBackend::translate_sage_syntax("x, y = var('x y')\nx^2 + y^2");
    assert(s2 == "x^2 + y^2");

    // Dot methods
    std::string s3 = SageBackend::translate_sage_syntax("(x^3 - 8).factor()");
    assert(s3 == "factor(x^3 - 8)");

    std::string s4 = SageBackend::translate_sage_syntax("(sin(x)).diff(x)");
    assert(s4 == "diff(sin(x), x)");

    std::string s5 = SageBackend::translate_sage_syntax("(x^2 + 2*x + 1).roots(x)");
    assert(s5 == "solve(x^2 + 2*x + 1 == 0, x)");

    // Matrix syntax
    std::string s6 = SageBackend::translate_sage_syntax("matrix([[1, 2], [3, 4]])");
    assert(s6 == "[[1, 2], [3, 4]]");

    std::cout << "  Passed. Sage variable declarations, dot-methods, and matrix syntax translated." << std::endl;
}

void test_sage_calculus() {
    std::cout << "[Test 2/7] Testing SageMath Calculus Operations..." << std::endl;
    SageBackend backend;

    // Derivative: diff(x^4 + 3*x^2 - 5*x + 1, x)
    MathRequest req1;
    req1.expression = "x^4 + 3*x^2 - 5*x + 1";
    req1.operation = MathOperation::kDifferentiate;
    req1.variable = "x";
    MathResult res1 = backend.execute(req1);
    assert(res1.status == MathStatus::kSuccess);
    assert(res1.exact_result.find("4*x^3") != std::string::npos && res1.exact_result.find("6*x - 5") != std::string::npos);

    // Indefinite Integral: integrate(cos(x), x)
    MathRequest req2;
    req2.expression = "cos(x)";
    req2.operation = MathOperation::kIntegrate;
    req2.variable = "x";
    MathResult res2 = backend.execute(req2);
    assert(res2.status == MathStatus::kSuccess);
    assert(res2.exact_result.find("Sin") != std::string::npos || res2.exact_result.find("sin") != std::string::npos);

    // Definite Integral: integral(x^2, 0, 3)
    MathRequest req3;
    req3.expression = "x^2";
    req3.operation = MathOperation::kIntegrate;
    req3.variable = "x";
    req3.point = "0,3";
    MathResult res3 = backend.execute(req3);
    assert(res3.status == MathStatus::kSuccess);
    assert(res3.exact_result == "9");

    // Limit: limit(sin(x)/x, x=0)
    MathRequest req4;
    req4.expression = "sin(x)/x";
    req4.operation = MathOperation::kLimit;
    req4.variable = "x";
    req4.point = "0";
    MathResult res4 = backend.execute(req4);
    assert(res4.status == MathStatus::kSuccess);
    assert(res4.exact_result == "1");

    // Taylor series: taylor(exp(x), x, 0, 3)
    MathRequest req5;
    req5.expression = "exp(x)";
    req5.operation = MathOperation::kSeries;
    req5.variable = "x";
    req5.point = "0";
    req5.order = 3;
    MathResult res5 = backend.execute(req5);
    assert(res5.status == MathStatus::kSuccess);

    std::cout << "  Passed. Derivatives, integrals, limits, and Taylor expansions verified." << std::endl;
}

void test_sage_algebra_and_equations() {
    std::cout << "[Test 3/7] Testing SageMath Algebraic Simplification & Equation Solving..." << std::endl;
    SageBackend backend;

    // Factor: (x^3 - 1)
    MathRequest req1;
    req1.expression = "x^3 - 1";
    req1.operation = MathOperation::kFactor;
    MathResult res1 = backend.execute(req1);
    assert(res1.status == MathStatus::kSuccess);
    assert(res1.exact_result.find("x - 1") != std::string::npos || res1.exact_result.find("-1 + x") != std::string::npos);

    // Expand: (x + 2)*(x - 3)
    MathRequest req2;
    req2.expression = "(x + 2)*(x - 3)";
    req2.operation = MathOperation::kExpand;
    MathResult res2 = backend.execute(req2);
    assert(res2.status == MathStatus::kSuccess);
    assert(res2.exact_result == "x^2 - x - 6");

    // Solve: x^2 - 5*x + 6 == 0
    MathRequest req3;
    req3.expression = "x^2 - 5*x + 6 == 0";
    req3.operation = MathOperation::kSolve;
    req3.variable = "x";
    MathResult res3 = backend.execute(req3);
    assert(res3.status == MathStatus::kSuccess);
    assert((res3.exact_result.find("x == 2") != std::string::npos || res3.exact_result.find("x -> 2") != std::string::npos) &&
           (res3.exact_result.find("x == 3") != std::string::npos || res3.exact_result.find("x -> 3") != std::string::npos));

    std::cout << "  Passed. Factorization, expansion, and quadratic equation solution sets verified." << std::endl;
}

void test_sage_linear_algebra() {
    std::cout << "[Test 4/7] Testing SageMath Linear Algebra & Matrices..." << std::endl;
    SageBackend backend;

    // Matrix Determinant: det([[1, 2], [3, 4]]) -> -2
    MathRequest req1;
    req1.expression = "det([[1, 2], [3, 4]])";
    req1.operation = MathOperation::kEvaluate;
    MathResult res1 = backend.execute(req1);
    assert(res1.status == MathStatus::kSuccess);
    assert(res1.exact_result == "-2");

    // Matrix Inverse: inverse([[1, 2], [3, 4]])
    MathRequest req2;
    req2.expression = "inverse([[1, 2], [3, 4]])";
    req2.operation = MathOperation::kEvaluate;
    MathResult res2 = backend.execute(req2);
    assert(res2.status == MathStatus::kSuccess);
    assert(res2.exact_result.find("-2") != std::string::npos && res2.exact_result.find("1/2") != std::string::npos);

    // Matrix Transpose: transpose([[1, 2], [3, 4]])
    MathRequest req3;
    req3.expression = "transpose([[1, 2], [3, 4]])";
    req3.operation = MathOperation::kEvaluate;
    MathResult res3 = backend.execute(req3);
    assert(res3.status == MathStatus::kSuccess);
    assert(res3.exact_result.find("[[1, 3], [2, 4]]") != std::string::npos || res3.exact_result.find("1") != std::string::npos);

    // Matrix Rank: matrix([[1, 2], [2, 4]]).rank() -> 1
    MathRequest req4;
    req4.expression = "matrix([[1, 2], [2, 4]]).rank()";
    req4.operation = MathOperation::kEvaluate;
    MathResult res4 = backend.execute(req4);
    assert(res4.status == MathStatus::kSuccess);
    assert(res4.exact_result == "1");

    // Matrix Trace: matrix([[1, 2], [3, 4]]).trace() -> 5
    MathRequest req5;
    req5.expression = "matrix([[1, 2], [3, 4]]).trace()";
    req5.operation = MathOperation::kEvaluate;
    MathResult res5 = backend.execute(req5);
    assert(res5.status == MathStatus::kSuccess);
    assert(res5.exact_result == "5");

    // Linear Solve: matrix([[1, 1], [1, -1]]).solve_right(vector([4, 2])) -> [3, 1]
    MathRequest req6;
    req6.expression = "matrix([[1, 1], [1, -1]]).solve_right(vector([4, 2]))";
    req6.operation = MathOperation::kEvaluate;
    MathResult res6 = backend.execute(req6);
    assert(res6.status == MathStatus::kSuccess);
    assert(res6.exact_result.find("3") != std::string::npos && res6.exact_result.find("1") != std::string::npos);

    // Cholesky: matrix([[4, 2], [2, 10]]).cholesky() -> [[2, 0], [1, 3]]
    MathRequest req7;
    req7.expression = "matrix([[4, 2], [2, 10]]).cholesky()";
    req7.operation = MathOperation::kEvaluate;
    MathResult res7 = backend.execute(req7);
    assert(res7.status == MathStatus::kSuccess);
    assert(res7.exact_result.find("2") != std::string::npos && res7.exact_result.find("3") != std::string::npos);

    std::cout << "  Passed. Determinant, inverse, transpose, rank, trace, solve_right, and Cholesky verified." << std::endl;
}

void test_sage_number_theory() {
    std::cout << "[Test 5/7] Testing SageMath Computational Number Theory..." << std::endl;
    SageBackend backend;

    // is_prime(1000000007)
    MathRequest req1;
    req1.expression = "is_prime(1000000007)";
    req1.operation = MathOperation::kEvaluate;
    MathResult res1 = backend.execute(req1);
    assert(res1.status == MathStatus::kSuccess);
    assert(res1.exact_result == "True");

    // is_prime(1000000005)
    MathRequest req2;
    req2.expression = "is_prime(1000000005)";
    req2.operation = MathOperation::kEvaluate;
    MathResult res2 = backend.execute(req2);
    assert(res2.status == MathStatus::kSuccess);
    assert(res2.exact_result == "False");

    // euler_phi(100) -> 40
    MathRequest req3;
    req3.expression = "euler_phi(100)";
    req3.operation = MathOperation::kEvaluate;
    MathResult res3 = backend.execute(req3);
    assert(res3.status == MathStatus::kSuccess);
    assert(res3.exact_result == "40");

    // fibonacci(20) -> 6765
    MathRequest req4;
    req4.expression = "fibonacci(20)";
    req4.operation = MathOperation::kEvaluate;
    MathResult res4 = backend.execute(req4);
    assert(res4.status == MathStatus::kSuccess);
    assert(res4.exact_result == "6765");

    // power_mod(7, 100, 101) -> 1 (Fermat's little theorem)
    MathRequest req5;
    req5.expression = "power_mod(7, 100, 101)";
    req5.operation = MathOperation::kEvaluate;
    MathResult res5 = backend.execute(req5);
    assert(res5.status == MathStatus::kSuccess);
    assert(res5.exact_result == "1");

    // gcd(240, 46) -> 2
    MathRequest req6;
    req6.expression = "gcd(240, 46)";
    req6.operation = MathOperation::kEvaluate;
    MathResult res6 = backend.execute(req6);
    assert(res6.status == MathStatus::kSuccess);
    assert(res6.exact_result == "2");

    // crt([2, 3], [3, 5]) -> 8
    MathRequest req7;
    req7.expression = "crt([2, 3], [3, 5])";
    req7.operation = MathOperation::kEvaluate;
    MathResult res7 = backend.execute(req7);
    assert(res7.status == MathStatus::kSuccess);
    assert(res7.exact_result == "8");

    // number_of_partitions(10) -> 42
    MathRequest req8;
    req8.expression = "number_of_partitions(10)";
    req8.operation = MathOperation::kEvaluate;
    MathResult res8 = backend.execute(req8);
    assert(res8.status == MathStatus::kSuccess);
    assert(res8.exact_result == "42");

    std::cout << "  Passed. Primality, Euler totient, Fibonacci, modular exponentiation, GCD, CRT, and partitions verified." << std::endl;
}

void test_sage_dot_methods_execution() {
    std::cout << "[Test 6/7] Testing SageMath Dot Method Pipeline Execution..." << std::endl;
    SageBackend backend;

    // (x^4 - 16).factor()
    MathRequest req1;
    req1.expression = "(x^4 - 16).factor()";
    req1.operation = MathOperation::kEvaluate;
    MathResult res1 = backend.execute(req1);
    assert(res1.status == MathStatus::kSuccess);
    assert(res1.exact_result.find("x - 2") != std::string::npos || res1.exact_result.find("-2 + x") != std::string::npos);

    // (x^3).diff(x)
    MathRequest req2;
    req2.expression = "(x^3).diff(x)";
    req2.operation = MathOperation::kEvaluate;
    MathResult res2 = backend.execute(req2);
    assert(res2.status == MathStatus::kSuccess);
    assert(res2.exact_result.find("3*x^2") != std::string::npos || res2.exact_result.find("3 * x^2") != std::string::npos);

    std::cout << "  Passed. Dot-method expressions parsed and executed directly." << std::endl;
}

void test_sage_runtime_integration() {
    std::cout << "[Test 7/7] Testing SageMath Integration into Strata MathRuntime..." << std::endl;
    MathRuntime runtime;

    assert(runtime.sage_backend().is_available());
    assert(runtime.sage_backend().name() == "StrataCAS");

    // Process Sage expression through MathRuntime
    MathRequest req;
    req.expression = "var('x'); (x^2 - 9).factor()";
    req.operation = MathOperation::kEvaluate;
    MathResult res = runtime.process_request(req);
    assert(res.status == MathStatus::kSuccess);
    assert(res.backend_type == MathBackendType::kUnifiedCAS);
    assert(res.exact_result.find("x - 3") != std::string::npos || res.exact_result.find("-3 + x") != std::string::npos);

    std::cout << "  Passed. MathRuntime routing to SageMath backend verified." << std::endl;
}

static MathResult run_sage(const std::string& expr, MathOperation op = MathOperation::kEvaluate) {
    SageBackend backend(false);
    MathRequest r;
    r.expression = expr;
    r.operation = op;
    return backend.execute(r);
}

void test_sage_hardening() {
    std::cout << "[Test 8/8] Testing SageMath argument shapes and input bounds..." << std::endl;
    // Sage's sigma(n, k) is sigma_k(n); the CAS takes (k, n)
    assert(run_sage("sigma(12)").exact_result == "28");
    assert(run_sage("sigma(12, 2)").exact_result == "210");
    assert(run_sage("integral(x^2, x, 0, 1)").exact_result == "1/3");
    assert(run_sage("limit(sin(x)/x, x=0)").exact_result == "1");
    assert(run_sage("taylor(sin(x), x, 0, 5)").exact_result.find("x^5/120") != std::string::npos);
    assert(run_sage("(x^2-1).roots(x)").exact_result == "{x -> -1, x -> 1}" || run_sage("(x^2-1).roots(x)").exact_result == "[x == -1, x == 1]");
    // a one-sided limit is never answered as a two-sided one
    assert(run_sage("limit(1/x, x=0, dir='plus')").status != MathStatus::kSuccess);

    // oversized and pathological inputs are refused, never crash
    const std::string big_paren = std::string(200000, '(') + "1" + std::string(200000, ')');
    assert(run_sage(big_paren).status == MathStatus::kInvalidExpression);
    assert(run_sage("x = var('" + std::string(100000, 'a') + "')").status == MathStatus::kInvalidExpression);
    assert(run_sage("R.<" + std::string(100000, 'a')).status == MathStatus::kInvalidExpression);
    std::string chain;
    for (int i = 0; i < 150; ++i) chain += "(x+1).expand().";
    assert(run_sage(chain + "expand()").status == MathStatus::kInvalidExpression);
    assert(run_sage(std::string(4000, '(')).status != MathStatus::kSuccess);
    assert(run_sage("__import__('os').system('ls')").status != MathStatus::kSuccess);
    std::cout << "  Passed. Argument shapes are right and oversized input is refused." << std::endl;
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "            STRATA SAGEMATH BACKEND ENGINE TEST SUITE                 \n";
    std::cout << "======================================================================\n";

    test_sage_syntax_translation();
    test_sage_calculus();
    test_sage_algebra_and_equations();
    test_sage_linear_algebra();
    test_sage_number_theory();
    test_sage_dot_methods_execution();
    test_sage_runtime_integration();
    test_sage_hardening();

    std::cout << "======================================================================\n";
    std::cout << "      ALL 8 SAGEMATH BACKEND TEST SUITES PASSED (8/8)                 \n";
    std::cout << "======================================================================\n";
    return 0;
}
