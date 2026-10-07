// SPDX-License-Identifier: GPL-3.0-or-later
// tests/math/test_cas.cpp - native CAS test suite (src/math/cas)
//
// There is no reference CAS to compare against, so correctness is checked by properties that must hold for any
// correct implementation (derivative vs finite difference, derivative of an integral, expand of a factorization,
// solve residuals, series vs function, determinant vs numeric elimination, parse/print round trip) plus a table of
// known results. CHECK is used instead of assert so the suite still checks under NDEBUG.
#include "strata/math/cas/engine.hpp"
#include "strata/math/math_backend.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

using namespace strata::math;
using namespace strata::math::cas;

static int g_checks = 0, g_failed = 0;

#define CHECK(cond, msg)                                                                          \
    do {                                                                                          \
        ++g_checks;                                                                               \
        if (!(cond)) {                                                                            \
            ++g_failed;                                                                           \
            std::printf("  FAILED: %s (line %d)\n", std::string(msg).c_str(), __LINE__);          \
        }                                                                                         \
    } while (0)

static std::string run(const std::string& text) {
    Engine en;
    try { return to_string(en.eval(parse(text))); }
    catch (const CasUnsupported& e) { return std::string("UNSUPPORTED: ") + e.what(); }
    catch (const CasLimitError& e) { return std::string("LIMIT: ") + e.what(); }
    catch (const CasMathError& e) { return std::string("MATHERROR: ") + e.what(); }
    catch (const CasError& e) { return std::string("ERROR: ") + e.what(); }
}

static void expect(const std::string& input, const std::string& want) {
    std::string got = run(input);
    if (got != want) std::printf("    %s\n      got  %s\n      want %s\n", input.c_str(), got.c_str(), want.c_str());
    CHECK(got == want, input);
}

static bool numeric_at(Engine& en, const Expr& e, const std::string& var, double x, double& out) {
    try { return en.numeric_value(Engine::substitute(e, {{var, real(x)}}), out); }
    catch (const CasError&) { return false; }
}

// ---------------------------------------------------------------------------------------------------------------
static void test_bigint_and_rationals() {
    std::printf("[CAS 1/12] BigInt and exact rationals...\n");
    std::mt19937_64 rng(42);
    int bad = 0;
    for (int i = 0; i < 4000; ++i) {
        __int128 a = static_cast<__int128>(static_cast<int64_t>(rng())) >> (rng() % 40);
        __int128 b = static_cast<__int128>(static_cast<int64_t>(rng())) >> (rng() % 50);
        if (b == 0) b = 7;
        auto big = [](__int128 v) {
            bool n = v < 0;
            unsigned __int128 t = n ? -v : v;
            std::string s = t == 0 ? "0" : "";
            while (t) { s.insert(s.begin(), static_cast<char>('0' + static_cast<int>(t % 10))); t /= 10; }
            BigInt out;
            BigInt::from_string((n ? "-" : "") + s, out);
            return out;
        };
        auto str = [](__int128 v) {
            bool n = v < 0;
            unsigned __int128 t = n ? -v : v;
            std::string s = t == 0 ? "0" : "";
            while (t) { s.insert(s.begin(), static_cast<char>('0' + static_cast<int>(t % 10))); t /= 10; }
            return (n ? "-" : "") + s;
        };
        BigInt A = big(a), B = big(b);
        if ((A + B).to_string() != str(a + b)) ++bad;
        if ((A - B).to_string() != str(a - b)) ++bad;
        if ((A * B).to_string() != str(a * b)) ++bad;
        if ((A / B).to_string() != str(a / b)) ++bad;
        if ((A % B).to_string() != str(a % b)) ++bad;
    }
    CHECK(bad == 0, "BigInt matches __int128 on 4000 random operand pairs");
    BigInt f(1);
    for (int i = 2; i <= 60; ++i) f = f * BigInt(i);
    CHECK(f.to_string() == "8320987112741390144276341183223364380754172606361245952449277696409600000000000000", "60!");
    expect("2^100", "1267650600228229401496703205376");
    expect("1/3+1/6", "1/2");
    expect("100!/98!", "9900");
    expect("99999999999*99999999999", "9999999999800000000001");
}

static void test_evaluator() {
    std::printf("[CAS 2/12] Evaluator, Expand, D, Simplify...\n");
    expect("2*x+3*x", "5*x");
    expect("x*x", "x^2");
    expect("x-(x+1)", "-1");
    expect("5x", "5*x");
    expect("sqrt(8)", "2*sqrt(2)");
    expect("sqrt(1/2)", "sqrt(2)/2");
    expect("2^(1/2)*2^(1/2)", "2");
    expect("sqrt(-4)", "2*I");
    expect("(2+2x)/2", "x + 1");
    expect("expand((x+1)^2)", "x^2 + 2*x + 1");
    expect("expand((x+2)*(x+3))", "x^2 + 5*x + 6");
    expect("expand((x+y)^3)", "x^3 + 3*x^2*y + 3*x*y^2 + y^3");
    expect("D[sin(x^2), x]", "2*x*cos(x^2)");
    expect("D[x^3, x]", "3*x^2");
    expect("D[x^x, x]", "x^x*log(x) + x^x");
    expect("simplify((x+1)^2-(x^2+2*x+1))", "0");
    expect("simplify(sin(x)^2+cos(x)^2)", "1");
    expect("cos(pi/3)", "1/2");
    expect("sin(-x)", "-sin(x)");
    expect("1/0", "MATHERROR: division by zero");
}

static void test_limits_of_resources() {
    std::printf("[CAS 3/12] Hostile input fails fast and cleanly...\n");
    auto t0 = std::chrono::steady_clock::now();
    CHECK(run("expand((x+1)^5000)").rfind("LIMIT", 0) == 0, "expand((x+1)^5000) is stopped");
    CHECK(run("expand((x+y+z+1)^200)").rfind("LIMIT", 0) == 0, "multivariate blow-up is stopped");
    CHECK(run("2^(2^100)").rfind("LIMIT", 0) == 0, "2^(2^100) is stopped");
    CHECK(run("2^100000000").rfind("LIMIT", 0) == 0, "huge exact power is stopped");
    CHECK(run("100000!").rfind("LIMIT", 0) == 0, "huge factorial is stopped");
    CHECK(run("x^^2").rfind("ERROR", 0) == 0, "malformed input is an error");
    CHECK(run("sin(").rfind("ERROR", 0) == 0, "unterminated call is an error");
    std::string deep(300, '(');
    deep += "x" + std::string(300, ')');
    CHECK(run(deep).rfind("LIMIT", 0) == 0, "nesting depth is capped");
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    CHECK(ms < 3000.0, "all hostile inputs finish within 3 s (" + std::to_string(ms) + " ms)");
}

static void test_hostile_shapes_do_not_crash() {
    std::printf("[Test] Pathological inputs are refused, not crashed on...\n");
    CHECK(run(std::string(200000, '-') + "x").rfind("LIMIT", 0) == 0, "200,000 unary minus signs");
    CHECK(run(std::string(200000, '+') + "x").rfind("LIMIT", 0) == 0, "200,000 unary plus signs");
    CHECK(run("x" + std::string(200000, '!')).rfind("LIMIT", 0) == 0, "200,000 factorial marks");
    std::string prod = "x";
    for (int i = 0; i < 300000; ++i) prod += "*x";
    CHECK(run(prod).rfind("LIMIT", 0) == 0, "300,000 factors");
    CHECK(run(std::string(2000000, '1')).rfind("LIMIT", 0) == 0, "a 2 MB expression");
    CHECK(run("(x+1)^(-9223372036854775808)").rfind("LIMIT", 0) == 0, "INT64_MIN exponent");
    CHECK(run("(x+1)^9223372036854775807").rfind("LIMIT", 0) == 0, "INT64_MAX exponent");
    CHECK(run("fibonacci(-9223372036854775808)").rfind("LIMIT", 0) == 0, "INT64_MIN Fibonacci index");
    CHECK(run("nextprime(10,-9223372036854775808)") != "", "INT64_MIN prime step is refused");
    std::string flat = "x";
    for (int i = 0; i < 50000; ++i) flat += "*x";
    CHECK(run(flat) == "x^50001", "a long but legitimate product still works");
}

static void test_derivative_vs_finite_difference() {
    std::printf("[CAS 4/12] D[f] against central finite differences on random expressions...\n");
    std::mt19937 rng(7);
    const char* atoms[] = {"x", "x^2", "x^3", "sin(x)", "cos(x)", "exp(x)", "log(x+3)", "atan(x)", "sqrt(x+4)", "1/(x^2+1)", "tanh(x)", "sinh(x)"};
    const int n_atoms = sizeof(atoms) / sizeof(atoms[0]);
    Engine en;
    int tested = 0, bad = 0;
    for (int it = 0; it < 150; ++it) {
        std::string f = atoms[rng() % n_atoms];
        int parts = static_cast<int>(rng() % 3);
        for (int k = 0; k < parts; ++k) {
            const char* op = (rng() % 3 == 0) ? "*" : ((rng() % 2) ? "+" : "-");
            f = "(" + f + ")" + op + "(" + atoms[rng() % n_atoms] + ")";
        }
        if (rng() % 3 == 0) f = "sin(" + f + ")";
        try {
            Expr e = parse(f);
            Expr d = en.diff(e, "x");
            for (double x0 : {0.3, 1.1, -0.2}) {
                double h = 1e-5, fp = 0, fm = 0, dv = 0;
                if (!numeric_at(en, e, "x", x0 + h, fp) || !numeric_at(en, e, "x", x0 - h, fm) || !numeric_at(en, d, "x", x0, dv)) continue;
                double fd = (fp - fm) / (2 * h);
                ++tested;
                if (std::fabs(fd - dv) > 1e-4 * (1.0 + std::fabs(fd))) { ++bad; std::printf("    d/dx %s at %g: fd=%g got=%g\n", f.c_str(), x0, fd, dv); }
            }
        } catch (const CasError&) {
        }
    }
    CHECK(tested > 200, "enough derivative points were evaluable (" + std::to_string(tested) + ")");
    CHECK(bad == 0, "no derivative disagrees with a finite difference (" + std::to_string(tested) + " points)");
}

static void test_factor_expand_roundtrip() {
    std::printf("[CAS 5/12] Factorization: products reproduce, irreducibles stay irreducible...\n");
    expect("factor(x^2+5*x+6)", "(x + 2)*(x + 3)");
    expect("factor(x^4+4)", "(x^2 - 2*x + 2)*(x^2 + 2*x + 2)");
    expect("factor(x^4+1)", "x^4 + 1");
    expect("factor(x^6-1)", "(x - 1)*(x + 1)*(x^2 - x + 1)*(x^2 + x + 1)");
    expect("factor(2*x^2-2)", "2*(x - 1)*(x + 1)");
    expect("factor((x^2-1)/(x-1))", "x + 1");
    std::mt19937 rng(11);
    std::vector<std::string> irr = {"x^2+1", "x^2-2", "x^3-2", "x^4+x+1", "x^4+1", "x^5-x-1", "x^4+x^3+x^2+x+1", "x^2+3", "x^5-3"};
    Engine en;
    int bad = 0, total = 0;
    for (int it = 0; it < 80; ++it) {
        std::string p = "1";
        int count = 1 + static_cast<int>(rng() % 3);
        for (int k = 0; k < count; ++k) p += "*(" + irr[rng() % irr.size()] + ")";
        if (rng() % 2) p += "*(x-" + std::to_string(static_cast<int>(rng() % 5)) + ")";
        ++total;
        try {
            Expr e = en.expand(parse(p));
            Expr f = en.factor(e);
            Expr back = en.expand(f);
            if (!equal(back, e)) { ++bad; std::printf("    expand(factor(%s)) differs\n", p.c_str()); }
        } catch (const CasError& ex) { ++bad; std::printf("    %s: %s\n", p.c_str(), ex.what()); }
    }
    CHECK(bad == 0, "expand(factor(p)) == p for " + std::to_string(total) + " random products");
}

static void test_solve() {
    std::printf("[CAS 6/12] Solve: complete solution sets, residuals, honest refusals...\n");
    expect("solve(x^2+5*x+6==0,x)", "{{x -> -3}, {x -> -2}}");
    expect("solve(2*x+5==15,x)", "{{x -> 5}}");
    expect("solve(x^2+1==0,x)", "{{x -> -I}, {x -> I}}");
    expect("solve(x^2==x,x)", "{{x -> 0}, {x -> 1}}");
    expect("solve((x^2-1)/(x-1)==0,x)", "{{x -> -1}}");
    expect("solve({x+y==3,x-y==1},{x,y})", "{{x -> 2, y -> 1}}");
    expect("solve({x+y==3,x+y==4},{x,y})", "{}");
    CHECK(run("solve(x^3-2==0,x)").rfind("UNSUPPORTED", 0) == 0, "x^3 = 2 is refused, not given a partial set");
    CHECK(run("solve(sin(x)==0,x)").rfind("UNSUPPORTED", 0) == 0, "sin x = 0 is refused, not given a partial set");
    // random polynomials with known rational roots: every root found, each substitutes to zero
    std::mt19937 rng(3);
    Engine en;
    int bad = 0;
    for (int it = 0; it < 60; ++it) {
        int nroots = 1 + static_cast<int>(rng() % 4);
        std::vector<int> roots;
        std::string p = std::to_string(1 + static_cast<int>(rng() % 3));
        for (int k = 0; k < nroots; ++k) {
            int r = static_cast<int>(rng() % 9) - 4;
            roots.push_back(r);
            p += "*(x-(" + std::to_string(r) + "))";
        }
        try {
            Expr sol = en.solve(parse(p + "==0"), {"x"});
            std::vector<int> uniq = roots;
            std::sort(uniq.begin(), uniq.end());
            uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
            if (sol->args.size() != uniq.size()) { ++bad; continue; }
            for (size_t i = 0; i < uniq.size(); ++i) {
                Expr v = sol->args[i]->args[0]->args[1];
                if (!(v->is_number() && v->q == Rational(uniq[i]))) ++bad;
            }
        } catch (const CasError&) { ++bad; }
    }
    CHECK(bad == 0, "60 random polynomials: every root found exactly, nothing extra");
}

static void test_series() {
    std::printf("[CAS 7/12] Series against the function near the expansion point...\n");
    expect("series(exp(x),{x,0,4})", "1 + x + x^2/2 + x^3/6 + x^4/24 + O[x]^5");
    expect("series(sin(x),{x,0,7})", "x - x^3/6 + x^5/120 - x^7/5040 + O[x]^8");
    expect("series(tan(x),{x,0,7})", "x + x^3/3 + 2*x^5/15 + 17*x^7/315 + O[x]^8");
    expect("series(1/sin(x),{x,0,4})", "1/x + x/6 + 7*x^3/360 + O[x]^5");
    CHECK(run("series(log(x),{x,0,3})").rfind("UNSUPPORTED", 0) == 0, "log(x) at 0 is not a power series: refused");
    CHECK(run("series(abs(x),{x,0,3})").rfind("UNSUPPORTED", 0) == 0, "abs(x) at 0 is not analytic: refused");
    Engine en;
    const char* fns[] = {"exp(x)", "sin(x)", "cos(x)", "1/(1-x)", "log(1+x)", "atan(x)", "tan(x)", "sqrt(1+x)", "exp(sin(x))", "1/(1+x^2)", "sinh(x)", "(1+x)^(1/3)"};
    int bad = 0;
    for (const char* fn : fns) {
        const int n = 8;
        Expr s = en.series(parse(fn), "x", zero(), n);
        // evaluate the truncated polynomial at x = 0.05 and compare with the function
        const Expr& coeffs = s->args[2];
        int64_t nmin = 0;
        s->args[3]->q.num.to_int64(nmin);
        double x0 = 0.05, approx = 0.0, exact = 0.0;
        for (size_t i = 0; i < coeffs->args.size(); ++i) {
            double c = 0.0;
            if (!en.numeric_value(coeffs->args[i], c)) { ++bad; continue; }
            approx += c * std::pow(x0, static_cast<double>(nmin + static_cast<int64_t>(i)));
        }
        if (!numeric_at(en, parse(fn), "x", x0, exact)) { ++bad; continue; }
        if (std::fabs(approx - exact) > 50.0 * std::pow(x0, n + 1)) { ++bad; std::printf("    series of %s: approx=%g exact=%g\n", fn, approx, exact); }
    }
    CHECK(bad == 0, "12 series agree with the function to O(x^9) at x = 0.05");
}

static void test_limits() {
    std::printf("[CAS 8/12] Limits...\n");
    struct Case { const char* f; const char* pt; const char* want; };
    const Case cases[] = {
        {"sin(x)/x", "0", "1"}, {"(1-cos(x))/x^2", "0", "1/2"}, {"(1+1/x)^x", "oo", "E"}, {"(1+2/x)^(3*x)", "oo", "exp(6)"},
        {"1/x", "0", "Indeterminate"}, {"1/x", "0+", "Infinity"}, {"1/x", "0-", "-Infinity"}, {"1/x^2", "0", "Infinity"},
        {"x^2/exp(x)", "oo", "0"}, {"x*exp(-x)", "oo", "0"}, {"log(x)/x", "oo", "0"}, {"atan(x)", "oo", "pi/2"},
        {"(x^2+1)/(2*x^2-3)", "oo", "1/2"}, {"(x^2-1)/(x-1)", "1", "2"}, {"(exp(x)-1)/x", "0", "1"}, {"(sin(x)-x)/x^3", "0", "-1/6"},
        {"x*log(x)", "0+", "0"}, {"x^x", "0+", "1"}, {"x-log(x)", "oo", "Infinity"}, {"abs(x)/x", "0", "Indeterminate"},
        {"abs(x)/x", "0+", "1"}, {"abs(x)/x", "0-", "-1"}, {"x^2+3", "2", "7"},
    };
    for (const auto& c : cases) {
        std::string pt = c.pt;
        int dir = 0;
        if (pt.size() > 1 && (pt.back() == '+' || pt.back() == '-')) { dir = pt.back() == '+' ? 1 : -1; pt.pop_back(); }
        Engine en;
        std::string got;
        try { got = to_string(en.limit(parse(c.f), "x", parse(pt), dir)); } catch (const CasError& e) { got = std::string("ERR ") + e.what(); }
        if (got != c.want) std::printf("    limit %s at %s: got %s want %s\n", c.f, c.pt, got.c_str(), c.want);
        CHECK(got == c.want, std::string("limit of ") + c.f + " at " + c.pt);
    }
    CHECK(run("limit(sin(x),x->oo)").rfind("UNSUPPORTED", 0) == 0, "oscillating limit is refused");
}

static void test_integrals() {
    std::printf("[CAS 9/12] Integrate: derivative of the result equals the integrand...\n");
    const char* integrands[] = {
        "x^2", "3*x^2+2*x+1", "1/x", "1/x^2", "sin(x)", "cos(2*x)", "exp(3*x)", "x^2*sin(x)", "x*exp(x)", "x^3*exp(2*x)", "log(x)", "x*log(x)",
        "1/(x^2+1)", "1/(x^2-1)", "1/(x^2+5*x+6)", "(x+1)/(x^2+x+1)", "1/(x^2+1)^2", "x/(x^2+1)", "(x^3+1)/(x^2-1)", "sin(x)^2", "cos(x)^4",
        "sin(x)^3*cos(x)^2", "sin(x)*cos(x)", "exp(x)*sin(x)", "exp(2*x)*cos(3*x)", "2*x*cos(x^2)", "x*exp(x^2)", "1/(x*log(x))", "tan(x)",
        "atan(x)", "x*atan(x)", "1/sqrt(1-x^2)", "sqrt(x)", "1/sqrt(x)", "(2*x+1)^5", "sin(x)/cos(x)^2", "cos(x)*exp(sin(x))", "1/(x^3-1)",
        "log(x)^2", "log(x)^3", "1/(x^2+2*x+5)", "(3*x+2)/(x^2+4)^2", "x^2/(x^2+1)", "1/(x*(x+1)*(x+2))",
    };
    Engine en;
    int bad = 0, done = 0;
    for (const char* f : integrands) {
        try {
            Expr e = parse(f);
            Expr F = en.integrate(e, "x");
            Expr dF = en.diff(F, "x");
            int ok_points = 0;
            for (double x0 : {0.37, 0.91, 1.7, 2.3}) {
                double a = 0, b = 0;
                if (!numeric_at(en, dF, "x", x0, a) || !numeric_at(en, e, "x", x0, b)) continue;
                ++ok_points;
                if (std::fabs(a - b) > 1e-7 * (1.0 + std::fabs(b))) { ++bad; std::printf("    integral of %s: d/dx F=%g f=%g at %g\n", f, a, b, x0); }
            }
            if (ok_points < 2) { ++bad; std::printf("    integral of %s: too few evaluable points\n", f); }
            ++done;
        } catch (const CasError& ex) { ++bad; std::printf("    integral of %s: %s\n", f, ex.what()); }
    }
    CHECK(bad == 0, "d/dx of every antiderivative equals its integrand (" + std::to_string(done) + " integrals)");
    expect("integrate(x^2,x)", "x^3/3");
    expect("integrate(1/(x^2+1),x)", "atan(x)");
    expect("integrate(x^2*sin(x),x)", "-x^2*cos(x) + 2*x*sin(x) + 2*cos(x)");
    CHECK(run("integrate(exp(x^2),x)").rfind("UNSUPPORTED", 0) == 0, "exp(x^2) has no elementary antiderivative here: refused");
    CHECK(run("integrate(sin(x)/x,x)").rfind("UNSUPPORTED", 0) == 0, "sin(x)/x is refused");
    CHECK(run("integrate(1/(x^4+1),x)").rfind("UNSUPPORTED", 0) == 0, "1/(x^4+1) needs algebraic logs: refused");
}

static void test_definite_integrals() {
    std::printf("[CAS 10/12] Definite integrals, improper endpoints and singularities...\n");
    expect("integrate(x^2,{x,0,3})", "9");
    expect("integrate(sin(x),{x,0,pi})", "2");
    expect("integrate(1/x^2,{x,1,oo})", "1");
    expect("integrate(1/(1+x^2),{x,-oo,oo})", "pi");
    expect("integrate(exp(-x),{x,0,oo})", "1");
    expect("integrate(log(x),{x,0,1})", "-1");
    expect("integrate(x^2*sin(x),{x,0,pi})", "pi^2 - 4");
    expect("integrate(1/sqrt(x),{x,0,4})", "4");
    expect("integrate(x^2,{x,3,0})", "-9");
    CHECK(run("integrate(1/x^2,{x,-1,1})").rfind("MATHERROR", 0) == 0, "pole inside [-1,1] is reported, not -2");
    CHECK(run("integrate(1/(x-1),{x,0,2})").rfind("MATHERROR", 0) == 0, "pole at 1 inside [0,2] is reported");
    CHECK(run("integrate(1/x,{x,0,1})").rfind("MATHERROR", 0) == 0, "divergent endpoint is reported");
}

static void test_matrices_and_combinatorics() {
    std::printf("[CAS 11/12] Determinants, inverses and exact combinatorics...\n");
    expect("det({{1,2},{3,4}})", "-2");
    expect("det({{a,b},{c,d}})", "a*d - b*c");
    expect("inverse({{1,2},{3,4}})", "{{-2, 1}, {3/2, -1/2}}");
    expect("transpose({{1,2,3},{4,5,6}})", "{{1, 4}, {2, 5}, {3, 6}}");
    expect("dot({{1,2},{3,4}},{{5,6},{7,8}})", "{{19, 22}, {43, 50}}");
    expect("binomial(52,5)", "2598960");
    expect("permutations(10,3)", "720");
    expect("multinomial(2,3,4)", "1260");
    CHECK(run("inverse({{1,2},{2,4}})").rfind("MATHERROR", 0) == 0, "singular matrix has no inverse");
    // random integer matrices: exact determinant against double-precision elimination
    std::mt19937 rng(5);
    Engine en;
    int bad = 0;
    for (int it = 0; it < 60; ++it) {
        const int n = 2 + static_cast<int>(rng() % 4);
        std::vector<std::vector<double>> a(n, std::vector<double>(n));
        std::string text = "det({";
        for (int i = 0; i < n; ++i) {
            text += i ? ",{" : "{";
            for (int j = 0; j < n; ++j) {
                int v = static_cast<int>(rng() % 19) - 9;
                a[i][j] = v;
                text += (j ? "," : "") + std::string(v < 0 ? "(" + std::to_string(v) + ")" : std::to_string(v));
            }
            text += "}";
        }
        text += "})";
        double det = 1.0;
        for (int c = 0; c < n; ++c) {
            int p = c;
            for (int r = c + 1; r < n; ++r) if (std::fabs(a[r][c]) > std::fabs(a[p][c])) p = r;
            if (std::fabs(a[p][c]) < 1e-12) { det = 0; break; }
            if (p != c) { std::swap(a[p], a[c]); det = -det; }
            det *= a[c][c];
            for (int r = c + 1; r < n; ++r) { double f = a[r][c] / a[c][c]; for (int k = c; k < n; ++k) a[r][k] -= f * a[c][k]; }
        }
        double got = 0.0;
        if (!en.numeric_value(parse(text), got) || std::fabs(got - det) > 1e-6 * (1.0 + std::fabs(det))) { ++bad; std::printf("    %s: got %g want %g\n", text.c_str(), got, det); }
    }
    CHECK(bad == 0, "60 random determinants agree with numeric elimination");
}

static void test_parse_print_roundtrip_and_backend() {
    std::printf("[CAS 12/12] Parse/print round trip and backend status mapping...\n");
    std::mt19937 rng(9);
    const char* atoms[] = {"x", "y", "2", "3/4", "sin(x)", "exp(y)", "x^2", "(x+1)", "log(x)", "sqrt(y)"};
    Engine en;
    int bad = 0;
    for (int it = 0; it < 200; ++it) {
        std::string f = atoms[rng() % 10];
        for (int k = 0, parts = static_cast<int>(rng() % 4); k < parts; ++k) {
            const char* ops[] = {"+", "-", "*", "/"};
            f = "(" + f + ")" + ops[rng() % 4] + "(" + atoms[rng() % 10] + ")";
        }
        try {
            Expr e = en.eval(parse(f));
            Expr again = en.eval(parse(to_string(e)));
            if (!equal(e, again)) { ++bad; std::printf("    round trip: %s -> %s -> %s\n", f.c_str(), to_string(e).c_str(), to_string(again).c_str()); }
        } catch (const CasError&) {
        }
    }
    CHECK(bad == 0, "parse(print(e)) == e on 200 random expressions");

    MathicsBackend backend;
    auto status = [&](MathOperation op, const std::string& expr, const std::string& var = "x") {
        MathRequest r;
        r.operation = op;
        r.expression = expr;
        r.variable = var;
        return backend.execute(r).status;
    };
    CHECK(status(MathOperation::kIntegrate, "exp(x^2)") == MathStatus::kUnsupportedOperation, "unsupported integral -> kUnsupportedOperation");
    CHECK(status(MathOperation::kSimplify, "x^^2") == MathStatus::kInvalidExpression, "bad syntax -> kInvalidExpression");
    CHECK(status(MathOperation::kExpand, "(x+1)^5000") == MathStatus::kResourceLimitExceeded, "budget -> kResourceLimitExceeded");
    CHECK(status(MathOperation::kEvaluate, "1/0") == MathStatus::kExecutionError, "division by zero -> kExecutionError");
    CHECK(status(MathOperation::kExpand, "(x+1)^2") == MathStatus::kSuccess, "good input -> kSuccess");
    MathRequest r;
    r.operation = MathOperation::kIntegrate;
    r.expression = "exp(x^2)";
    CHECK(backend.execute(r).exact_result.empty(), "a refused request never carries a result string");
}

static void test_number_theory_and_iteration() {
    std::printf("[Test] Number theory, sequences, Range/Table/Sum/Product...\n");
    // primality against a sieve
    {
        std::vector<bool> comp(20001, false);
        bool all = true;
        for (int i = 2; i <= 20000; ++i) {
            const bool prime = !comp[i];
            for (long j = static_cast<long>(i) * i; j <= 20000; j += i) comp[j] = true;
            if ((run("primeq(" + std::to_string(i) + ")") == "True") != prime) all = false;
        }
        CHECK(all, "PrimeQ agrees with a sieve for every n <= 20000");
    }
    CHECK(run("primeq(561)") == "False", "Carmichael 561 is composite");
    CHECK(run("primeq(3215031751)") == "False", "strong pseudoprime to bases 2,3,5,7 is composite");
    CHECK(run("primeq(2^89-1)") == "True", "Mersenne prime 2^89-1");
    CHECK(run("primeq(2^67-1)") == "False", "2^67-1 is composite");
    CHECK(run("primeq(-7)") == "True", "PrimeQ[-7] (primes in Z)");
    CHECK(run("prime(1)") == "2" && run("prime(100)") == "541" && run("prime(1000)") == "7919", "Prime[n]");
    CHECK(run("primepi(1000)") == "168", "PrimePi[1000]");
    CHECK(run("nextprime(100)") == "101" && run("nextprime(100,-1)") == "97", "NextPrime both directions");
    // factorization: product of primes^e reproduces n, every factor prime
    for (const char* n : {"360", "2^64+1", "2^67-1", "1000000007*998244353", "600851475143", "123456789012345678901"}) {
        std::string f = run(std::string("factorinteger(") + n + ")");
        std::string rebuilt = "1";
        // {{p, e}, ...} -> p^e*...
        std::string cur; std::vector<std::string> nums;
        for (char c : f) { if (std::isdigit(static_cast<unsigned char>(c))) cur += c; else { if (!cur.empty()) nums.push_back(cur); cur.clear(); } }
        if (!cur.empty()) nums.push_back(cur);
        bool primes_ok = nums.size() % 2 == 0;
        for (size_t i = 0; i + 1 < nums.size(); i += 2) {
            rebuilt += "*(" + nums[i] + ")^" + nums[i + 1];
            if (run("primeq(" + nums[i] + ")") != "True") primes_ok = false;
        }
        CHECK(primes_ok && run(rebuilt) == run(n), (std::string("FactorInteger reproduces ") + n).c_str());
    }
    CHECK(run("factorinteger(2^64+1)") == "{{274177, 1}, {67280421310721, 1}}", "F6 factors");
    CHECK(run("factorinteger(1)") == "{}" && run("factorinteger(-12)") == "{{-1, 1}, {2, 2}, {3, 1}}", "FactorInteger edge cases");
    CHECK(run("factorinteger(3/4)") == "{{2, -2}, {3, 1}}", "FactorInteger of a rational");
    // phi / divisors / sigma / mobius vs brute force
    {
        bool ok = true;
        for (int n = 1; n <= 300; ++n) {
            int phi = 0, cnt = 0, sig = 0;
            std::string divs = "{";
            for (int d = 1; d <= n; ++d) {
                int a = d, b = n;
                while (b) { int t = a % b; a = b; b = t; }
                if (a == 1) ++phi;
                if (n % d == 0) { ++cnt; sig += d; divs += (divs.size() > 1 ? ", " : "") + std::to_string(d); }
            }
            divs += "}";
            if (run("eulerphi(" + std::to_string(n) + ")") != std::to_string(phi)) ok = false;
            if (run("divisors(" + std::to_string(n) + ")") != divs) ok = false;
            if (run("divisorsigma(1," + std::to_string(n) + ")") != std::to_string(sig)) ok = false;
            if (run("divisorsigma(0," + std::to_string(n) + ")") != std::to_string(cnt)) ok = false;
        }
        CHECK(ok, "EulerPhi / Divisors / DivisorSigma match brute force for n <= 300");
    }
    CHECK(run("moebiusmu(30)") == "-1" && run("moebiusmu(12)") == "0" && run("moebiusmu(1)") == "1", "MoebiusMu");
    // modular arithmetic
    CHECK(run("powermod(2,100,13)") == "3", "PowerMod");
    CHECK(run("powermod(3,-1,7)") == "5", "modular inverse");
    CHECK(run("powermod(2,-1,4)") == "PowerMod(2, -1, 4)", "non-invertible stays unevaluated");
    CHECK(run("powermod(7,123456789,1000000007)") == "467332791", "large PowerMod");
    CHECK(run("extendedgcd(240,46)") == "{2, {-9, 47}}", "ExtendedGCD");
    CHECK(run("jacobisymbol(2,7)") == "1" && run("jacobisymbol(3,7)") == "-1" && run("jacobisymbol(7,7)") == "0", "JacobiSymbol");
    CHECK(run("quotient(-7,2)") == "-4" && run("quotient(7,2)") == "3", "Quotient floors");
    CHECK(run("coprimeq(9,28)") == "True" && run("coprimeq(9,27)") == "False" && run("divisible(12,4)") == "True", "CoprimeQ / Divisible");
    // sequences
    CHECK(run("fibonacci(10)") == "55" && run("fibonacci(-10)") == "-55" && run("fibonacci(0)") == "0", "Fibonacci");
    CHECK(run("fibonacci(100)") == "354224848179261915075", "Fibonacci(100)");
    CHECK(run("lucasl(10)") == "123" && run("lucasl(1)") == "1", "LucasL");
    CHECK(run("fibonacci(30)+fibonacci(31)-fibonacci(32)") == "0", "Fibonacci recurrence");
    CHECK(run("integerdigits(1234)") == "{1, 2, 3, 4}" && run("integerdigits(10,2)") == "{1, 0, 1, 0}" && run("fromdigits({1,2,3,4})") == "1234", "IntegerDigits / FromDigits");
    // iteration
    CHECK(run("range(5)") == "{1, 2, 3, 4, 5}" && run("range(2,10,3)") == "{2, 5, 8}", "Range");
    CHECK(run("table(i^2,{i,1,5})") == "{1, 4, 9, 16, 25}", "Table");
    CHECK(run("sum(i^2,{i,1,100})") == "338350" && run("product(i,{i,1,10})") == "3628800", "explicit Sum / Product");
    CHECK(run("total({1,2,3,4})") == "10", "Total");
    // symbolic sums are cross-checked against explicit evaluation
    {
        std::string closed = run("sum(i^3+2*i+1,{i,1,n})");
        bool ok = closed.find("Sum") == std::string::npos && closed.find("sum") == std::string::npos;
        for (int nn : {0, 1, 2, 5, 17, 40}) {
            std::string N = "(" + std::to_string(nn) + ")", c = closed;
            for (size_t pos = 0; (pos = c.find('n', pos)) != std::string::npos; pos += N.size()) c.replace(pos, 1, N);
            if (run(c) != run("sum(i^3+2*i+1,{i,1," + std::to_string(nn) + "})")) ok = false;
        }
        CHECK(ok, "closed-form Sum equals explicit summation for several n");
    }
    CHECK(run("expand(sum(i,{i,1,n}))") == run("expand(n*(n+1)/2)"), "Sum[i, {i,1,n}] = n(n+1)/2");
    CHECK(run("expand(sum(i^2,{i,1,n}))") == run("expand(n*(n+1)*(2*n+1)/6)"), "sum of squares");
    CHECK(run("expand(sum(i^3,{i,1,n}))") == run("expand((n*(n+1)/2)^2)"), "sum of cubes = (n(n+1)/2)^2");
    CHECK(run("expand(sum(i,{i,3,n}))") == run("expand(n*(n+1)/2-3)"), "lower bound other than 1");
    CHECK(run("expand(sum(1,{i,1,n}))") == "n", "Sum of a constant");
    CHECK(run("simplify(sum(2^i,{i,0,n}))") == run("simplify(2^(n+1)-1)"), "geometric series");
    CHECK(run("product(i,{i,1,n})") == "n!" || run("product(i,{i,1,n})") == "Factorial(n)", "Product[i,{i,1,n}] = n!");
    CHECK(run("sum(sin(i)/i,{i,1,n})") == "Sum(sin(i)/i, {i, 1, n})", "unknown symbolic sum stays unevaluated");
    CHECK(run("sum(i,{i,1,100000000})").rfind("LIMIT", 0) == 0, "huge explicit sum is stopped");
}

static void test_extended_math() {
    std::printf("[extended] linear algebra, vector calculus, partial fractions, inequalities, modular arithmetic\n");
    CHECK(run("rank({{1,2},{2,4}})") == "1", "rank of a singular matrix");
    CHECK(run("rank({{1,2,3},{4,5,6},{7,8,10}})") == "3", "rank of a regular 3x3");
    CHECK(run("tr({{1,2},{3,4}})") == "5", "trace");
    CHECK(run("charpoly({{1,2},{3,4}}, x)") == "x^2 - 5*x - 2", "characteristic polynomial");
    CHECK(run("eigenvalues({{2,1},{1,2}})") == "{3, 1}", "eigenvalues, with the larger first");
    CHECK(run("eigenvalues({{2,0},{0,2}})") == "{2, 2}", "eigenvalue multiplicity");
    CHECK(run("eigenvectors({{2,1},{1,2}})") == "{{1, 1}, {-1, 1}}", "eigenvectors in the order of the eigenvalues");
    CHECK(run("nullspace({{1,2},{2,4}})") == "{{-2, 1}}", "null space");
    {   // every null-space vector really is in the kernel
        Engine en;
        Expr ns = en.eval(parse("nullspace({{1,2,3},{2,4,6},{1,0,1}})"));
        bool ok = !ns->args.empty();
        for (const auto& v : ns->args) ok = ok && to_string(en.eval(app("Dot", {parse("{{1,2,3},{2,4,6},{1,0,1}}"), v}))) == "{0, 0, 0}";
        CHECK(ok, "null space vectors are annihilated by the matrix");
    }
    CHECK(run("lu({{4,3},{6,3}})") == "{{{1, 0}, {0, 1}}, {{1, 0}, {3/2, 1}}, {{4, 3}, {0, -3/2}}}", "LU: P, L, U");
    CHECK(run("lu({{0,1},{1,0}})").find("{{0, 1}, {1, 0}}") != std::string::npos, "LU pivots a zero");
    CHECK(run("D[x^3*y^2, x, y]") == "6*x^2*y", "mixed partial derivative D[f, x, y] differentiates by both variables (it used to ignore y)");
    CHECK(run("D[x^4, {x, 2}]") == "12*x^2", "D[f, {x, n}]");
    CHECK(run("diff(x^4, x, 2)") == "12*x^2", "Sage's diff(f, x, n)");
    CHECK(run("D[x^3*y^2, x, 2]") == "6*x*y^2", "an integer after a variable is the order");
    CHECK(run("D[x^3, 5]").find("ERROR") == 0, "a non-variable derivative argument is an error, not ignored");
    CHECK(run("grad(x^2*y, {x,y})") == "{2*x*y, x^2}", "gradient");
    CHECK(run("jacobian({x*y, x+y}, {x,y})") == "{{y, x}, {1, 1}}", "Jacobian");
    CHECK(run("hessian(x^2*y, {x,y})") == "{{2*y, 2*x}, {2*x, 0}}", "Hessian");
    CHECK(run("collect(x*y + x*z + y, x)") == "x*(y + z) + y", "collect");
    CHECK(run("apart(1/(x^2-1), x)") == "-1/(2*(x + 1)) + 1/(2*(x - 1))", "partial fractions, linear factors");
    for (const char* f : {"(x^3+2)/(x^2*(x+1))", "(x^2+1)/((x-1)*(x^2+x+1))", "(3*x+5)/((x+1)^2*(x-2))", "1/(x^3-x)"}) {   // apart must give back the same function
        Engine en;
        Expr orig = parse(f), parts = en.eval(parse(std::string("apart(") + f + ", x)"));
        bool same = true;
        for (double x : {0.5, 3.0, -4.5, 7.25}) {
            double a = 0, b = 0;
            same = same && numeric_at(en, orig, "x", x, a) && numeric_at(en, parts, "x", x, b) && std::fabs(a - b) < 1e-9 * (1 + std::fabs(a));
        }
        CHECK(same, std::string("apart(") + f + ") equals the original at sample points");
    }
    CHECK(run("reduce(x^2 > 4, x)") == "Or(Less(x, -2), Greater(x, 2))", "quadratic inequality");
    CHECK(run("reduce(x^2 - 5*x + 6 <= 0, x)") == "And(GreaterEqual(x, 2), LessEqual(x, 3))", "closed interval");
    CHECK(run("reduce((x-1)/(x+2) > 0, x)") == "Or(Less(x, -2), Greater(x, 1))", "rational inequality: the pole is excluded");
    CHECK(run("reduce(x^2 + 1 < 0, x)") == "False" && run("reduce(x^2 + 1 > 0, x)") == "True", "no real roots");
    CHECK(run("chineseremainder({2,3,2},{3,5,7})") == "23", "CRT");
    CHECK(run("chineseremainder({1,2},{4,6})").find("no common solution") != std::string::npos, "inconsistent congruences are refused");
    CHECK(run("modularinverse(3, 7)") == "5", "modular inverse");
    CHECK(run("modularinverse(4, 8)").find("no inverse") != std::string::npos, "no inverse when not coprime");
    CHECK(run("partitionsp(100)") == "190569292", "partition numbers");
    CHECK(run("partitionsp(0)") == "1", "p(0) = 1");
    CHECK(run("PolynomialMod[x^3 + 5 x^2 + 7, 3]") == "x^3 + 2*x^2 + 1", "PolynomialMod, one variable inferred");
    CHECK(run("PolynomialMod[x^3 + 5 x^2 + 7, x, 3]") == "x^3 + 2*x^2 + 1", "PolynomialMod, named variable");
    CHECK(run("PolynomialGCDMod[x^2 - 1, x^2 + 2 x + 1, x, 5]") == "x + 1", "gcd in GF(5)[x]");
    CHECK(run("PolynomialGCDMod[x^2 + 1, x + 3, x, 5]") == "1" || run("PolynomialGCDMod[x^2 + 1, x + 3, x, 5]") == "x + 3", "gcd in GF(5)[x]: x^2+1 = (x+3)(x+2) mod 5");
    CHECK(run("MultiplicativeOrder[2, 101]") == "100" && run("MultiplicativeOrder[3, 7]") == "6", "multiplicative order");
    CHECK(run("MultiplicativeOrder[4, 8]").find("not coprime") != std::string::npos, "order of a non-unit is refused");
    CHECK(run("PrimitiveRoot[7]") == "3" && run("PrimitiveRoot[23]") == "5" && run("PrimitiveRoot[101]") == "2", "smallest primitive roots");
    CHECK(run("PrimitiveRoot[8]").find("no primitive root") != std::string::npos && run("PrimitiveRoot[15]").find("no primitive root") != std::string::npos, "no primitive root for non-cyclic groups");
    CHECK(run("FactorMod[x^4 + 1, x, 2]") == "(x + 1)^4", "a p-th power in GF(2)[x]");
    CHECK(run("FactorMod[x^2 + 1, x, 5]") == "(x + 2)*(x + 3)", "splits mod 5");
    CHECK(run("FactorMod[x^2 + 1, x, 3]") == "x^2 + 1", "irreducible mod 3");
    CHECK(run("FactorMod[x^5 - x, x, 5]") == "x*(x + 1)*(x + 2)*(x + 3)*(x + 4)", "x^p - x is the product of all linear factors");
    CHECK(run("FactorMod[x^7 - 1, x, 2]") == "(x + 1)*(x^3 + x^2 + 1)*(x^3 + x + 1)", "cyclotomic structure mod 2");
    CHECK(run("FactorMod[3 x^3 + 2 x + 1, x, 7]") == "3*(x^3 + 3*x + 5)", "the leading coefficient is the content");
    {   // characteristic 2 needs the trace map: x^15 - 1 = (x+1)(x^2+x+1)(x^4+x+1)(x^4+x^3+1)(x^4+x^3+x^2+x+1) over GF(2)
        const std::string f = run("FactorMod[x^15 - 1, x, 2]");
        bool all = true;
        for (const char* t : {"(x + 1)", "(x^2 + x + 1)", "(x^4 + x + 1)", "(x^4 + x^3 + 1)", "(x^4 + x^3 + x^2 + x + 1)"}) all = all && f.find(t) != std::string::npos;
        CHECK(all, "equal-degree splitting in characteristic 2");
    }
    CHECK(run("FactorMod[x^2 + 1, x, 9]").find("prime modulus") != std::string::npos, "a composite modulus is refused");
    CHECK(run("3 < 5") == "True" && run("5 <= 3") == "False" && run("2 != 2") == "False", "comparisons of numbers");
}

int main() {
    std::printf("=================================================================\n");
    std::printf("   STRATA NATIVE CAS TEST SUITE                                  \n");
    std::printf("=================================================================\n");
    test_bigint_and_rationals();
    test_evaluator();
    test_limits_of_resources();
    test_hostile_shapes_do_not_crash();
    test_derivative_vs_finite_difference();
    test_factor_expand_roundtrip();
    test_solve();
    test_series();
    test_limits();
    test_integrals();
    test_definite_integrals();
    test_matrices_and_combinatorics();
    test_parse_print_roundtrip_and_backend();
    test_number_theory_and_iteration();
    test_extended_math();
    std::printf("=================================================================\n");
    std::printf("   %d checks, %d failed\n", g_checks, g_failed);
    std::printf("=================================================================\n");
    return g_failed == 0 ? 0 : 1;
}
