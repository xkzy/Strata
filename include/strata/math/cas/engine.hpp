// SPDX-License-Identifier: GPL-3.0-or-later
// include/strata/math/cas/engine.hpp - the native CAS evaluator and algorithms
//
// Port of the evaluation model of Mathics3 (GPL-3.0-or-later): canonical Plus / Times / Power evaluation on exact
// rationals, Expand, D, and the algorithms Mathics delegates to SymPy, implemented natively. Every public method
// either returns a correct result or throws CasError (cannot do it) / CasLimitError (over budget). Nothing is
// ever returned unevaluated as if it were an answer.
#pragma once

#include <functional>
#include "strata/math/cas/expr.hpp"
#include "strata/math/cas/parser.hpp"
#include "strata/math/cas/poly.hpp"

#include <chrono>
#include <string>
#include <vector>

namespace strata::math::cas {

struct Budget {
    uint64_t max_steps = 4'000'000;     // evaluation steps
    size_t max_terms = 20'000;          // terms in one expanded sum
    uint64_t max_exponent = 20'000;     // integer exponent applied to a non-trivial base
    size_t max_bits = 400'000;          // size of any exact integer
    size_t max_depth = 400;             // recursion depth
    double timeout_ms = 5000.0;
};

class Engine {
public:
    explicit Engine(Budget budget = Budget());

    Expr eval(const Expr& e);
    Expr expand(const Expr& e);
    Expr diff(const Expr& f, const std::string& var);
    Expr simplify(const Expr& e);
    // Numeric value of an expression with no free symbols (Pi, E allowed). Returns false if it has free symbols
    // or is complex.
    bool numeric_value(const Expr& e, double& out);

    // ---- polynomial algorithms (algebra.cpp) ----
    // Rational-function normal form: numerator and denominator as polynomials in `var` over Q, with common factors
    // cancelled. Returns false if `e` is not a rational function of `var` with rational coefficients.
    bool rational_function(const Expr& e, const std::string& var, Poly& num, Poly& den);
    // Coefficients of `e` (expanded internally) as a polynomial in `var`; coeffs[k] multiplies var^k and may
    // contain other symbols. Returns false if `e` is not a polynomial in `var`.
    bool poly_coefficients(const Expr& e, const std::string& var, std::vector<Expr>& coeffs);
    static Expr poly_to_expr(const Poly& p, const std::string& var);
    Expr cancel(const Expr& e);     // rational function in one variable -> reduced num/den
    Expr together(const Expr& e);   // same normal form, numerator and denominator expanded
    Expr factor(const Expr& e);
    // Solve for `vars`. `eqs` is an Equal, an expression (== 0) or a List of them. The result is a List of solution
    // sets, each a List of Rule(var, value). A CasUnsupported is thrown rather than returning a partial set.
    Expr solve(const Expr& eqs, const std::vector<std::string>& vars);

    // ---- calculus (series.cpp, limit.cpp, integrate.cpp) ----
    // Laurent/Taylor series of f around var = x0, terms up to and including (var-x0)^n. Result:
    // SeriesData(var, x0, List(coeffs), nmin, nmax, 1) with nmax = n + 1, like Mathics.
    Expr series(const Expr& f, const std::string& var, const Expr& x0, int64_t n);
    // Two-sided limit as var -> x0 (x0 may be Infinity / -Infinity); dir = +1 / -1 for one-sided, 0 for both.
    Expr limit(const Expr& f, const std::string& var, const Expr& x0, int dir = 0);
    // Antiderivative (no constant). Every result is checked by differentiation; CasUnsupported otherwise.
    Expr integrate(const Expr& f, const std::string& var);
    Expr integrate_definite(const Expr& f, const std::string& var, const Expr& a, const Expr& b);

    // Leading behaviour of g(t) as t -> 0 (g written in the symbol "$t"): g ~ lead * t^v. is_zero is set if g
    // vanishes to high order (limit 0). Throws CasUnsupported when g is not a Laurent series.
    void series_valuation(const Expr& g, int64_t& v, Expr& lead, bool& is_zero);

    // Replaces symbols by expressions (no evaluation).
    static Expr substitute(const Expr& e, const std::vector<std::pair<std::string, Expr>>& map);
    // Looks for a point where `e` is clearly non-zero (machine-precision evaluation at fixed sample points).
    // Returns true with a description like "x = 0.7 gives 0.7"; false if e is numerically zero at every valid
    // sample point or could not be evaluated anywhere (then nothing is known).
    bool find_nonzero_witness(const Expr& e, std::string& witness, int* evaluated = nullptr);

    // Helpers shared by the algorithm files.
    Expr plus(std::vector<Expr> args);
    Expr times(std::vector<Expr> args);
    Expr power(const Expr& base, const Expr& exp);
    Expr neg(const Expr& e);
    Expr sub(const Expr& a, const Expr& b);
    Expr div(const Expr& a, const Expr& b);

    void tick(uint64_t n = 1);

private:
    Budget budget_;
    uint64_t steps_ = 0;
    size_t depth_ = 0;
    std::chrono::steady_clock::time_point start_;

    struct DepthGuard {
        Engine& e;
        explicit DepthGuard(Engine& en) : e(en) {
            if (++e.depth_ > e.budget_.max_depth) { --e.depth_; throw CasLimitError("expression nesting too deep"); }
        }
        ~DepthGuard() { --e.depth_; }
    };

    Expr eval_apply(const Expr& e);
    Expr eval_function(const std::string& head, std::vector<Expr> args);
    Expr rational_power(const Rational& base, const Rational& exp);
    friend struct SeriesOps;
    friend struct LimitOps;
    friend struct IntegrateOps;
    Expr matrix_function(const std::string& head, const std::vector<Expr>& args);   // nullptr if not applicable
    Expr number_theory(const std::string& head, const std::vector<Expr>& args);     // numtheory.cpp; nullptr if not applicable
    Expr iteration_function(const std::string& head, const std::vector<Expr>& args);
    Expr expand_rec(const Expr& e);
    Expr expand_product(const Expr& a, const Expr& b);
    Expr pythagorean(const Expr& e);
    Expr diff_rec(const Expr& f, const std::string& var);
    bool numeric_rec(const Expr& e, double& out);
};

} // namespace strata::math::cas
