// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/cas/engine.cpp - evaluator, Expand, D, Simplify (see engine.hpp)
#include "strata/math/cas/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace strata::math::cas {

Engine::Engine(Budget budget) : budget_(budget), start_(std::chrono::steady_clock::now()) {}

void Engine::tick(uint64_t n) {
    steps_ += n;
    if (steps_ > budget_.max_steps) throw CasLimitError("computation exceeded its step budget");
    if ((steps_ & 1023u) < n) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
        if (ms > budget_.timeout_ms) throw CasLimitError("computation timed out");
    }
}

namespace {

bool is_const(const Expr& e) { return e->kind == Kind::Number || e->kind == Kind::Real; }
double to_d(const Expr& e) { return e->kind == Kind::Number ? e->q.to_double() : e->d; }

// A numeric coefficient: exact until a machine real touches it.
struct Coef {
    Rational q = Rational(0);
    double d = 0.0;
    bool real = false;
    void add(const Expr& e) {
        if (e->kind == Kind::Real) { if (!real) { d = q.to_double(); real = true; } d += e->d; }
        else if (real) d += e->q.to_double();
        else q = q + e->q;
    }
    Expr expr() const { return real ? ::strata::math::cas::real(d) : num(q); }
    bool is_zero() const { return real ? d == 0.0 : q.is_zero(); }
    bool is_one() const { return real ? d == 1.0 : q.is_one(); }
};

struct ProdCoef {
    Rational q = Rational(1);
    double d = 1.0;
    bool real = false;
    void mul(const Expr& e) {
        if (e->kind == Kind::Real) { if (!real) { d = q.to_double(); real = true; } d *= e->d; }
        else if (real) d *= e->q.to_double();
        else q = q * e->q;
    }
    Expr expr() const { return real ? ::strata::math::cas::real(d) : num(q); }
    bool is_zero() const { return real ? d == 0.0 : q.is_zero(); }
    bool is_one() const { return !real && q.is_one(); }
};

// Splits a canonical term into its numeric coefficient and the symbolic rest.
void split_term(const Expr& t, ProdCoef& coef, Expr& rest) {
    if (t->has_head("Times")) {
        std::vector<Expr> others;
        for (const auto& a : t->args) {
            if (is_const(a)) coef.mul(a); else others.push_back(a);
        }
        if (others.empty()) rest = one();
        else if (others.size() == 1) rest = others[0];
        else rest = app("Times", std::move(others));
        return;
    }
    if (is_const(t)) { coef.mul(t); rest = one(); return; }
    rest = t;
}

Expr make_term(const ProdCoef& c, const Expr& rest) {
    if (c.is_one()) return rest;
    if (rest->has_head("Times")) {
        std::vector<Expr> v{c.expr()};
        v.insert(v.end(), rest->args.begin(), rest->args.end());
        return app("Times", std::move(v));
    }
    return apply2("Times", c.expr(), rest);
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------
// Plus
// ---------------------------------------------------------------------------------------------------------------
Expr Engine::plus(std::vector<Expr> args) {
    tick(args.size());
    std::vector<Expr> flat;
    flat.reserve(args.size());
    for (auto& a : args) {
        if (a->has_head("Plus")) flat.insert(flat.end(), a->args.begin(), a->args.end());
        else flat.push_back(std::move(a));
    }
    if (flat.size() > budget_.max_terms * 4) throw CasLimitError("sum has too many terms");

    Coef constant;
    struct Term { Expr rest; ProdCoef c; };
    std::vector<Term> terms;
    for (const auto& t : flat) {
        if (is_const(t)) { constant.add(t); continue; }
        Term tm;
        split_term(t, tm.c, tm.rest);
        terms.push_back(std::move(tm));
    }
    std::stable_sort(terms.begin(), terms.end(), [](const Term& a, const Term& b) { return compare(a.rest, b.rest) < 0; });

    std::vector<Expr> out;
    for (size_t i = 0; i < terms.size();) {
        ProdCoef sum;
        sum.q = Rational(0);
        sum.d = 0.0;
        size_t j = i;
        while (j < terms.size() && equal(terms[j].rest, terms[i].rest)) {
            const ProdCoef& c = terms[j].c;
            if (c.real) { if (!sum.real) { sum.d = sum.q.to_double(); sum.real = true; } sum.d += c.d; }
            else if (sum.real) sum.d += c.q.to_double();
            else sum.q = sum.q + c.q;
            ++j;
        }
        if (!sum.is_zero()) out.push_back(make_term(sum, terms[i].rest));
        i = j;
    }
    if (!constant.is_zero() || (constant.real && out.empty())) out.push_back(constant.expr());
    if (out.empty()) return zero();
    if (out.size() == 1) return out[0];
    std::stable_sort(out.begin(), out.end(), [](const Expr& a, const Expr& b) { return compare(a, b) < 0; });
    return app("Plus", std::move(out));
}

Expr Engine::neg(const Expr& e) { return times({minus_one(), e}); }
Expr Engine::sub(const Expr& a, const Expr& b) { return plus({a, neg(b)}); }
Expr Engine::div(const Expr& a, const Expr& b) { return times({a, power(b, minus_one())}); }

// ---------------------------------------------------------------------------------------------------------------
// Times
// ---------------------------------------------------------------------------------------------------------------
Expr Engine::times(std::vector<Expr> args) {
    tick(args.size());
    std::vector<Expr> flat;
    flat.reserve(args.size());
    for (auto& a : args) {
        if (a->has_head("Times")) flat.insert(flat.end(), a->args.begin(), a->args.end());
        else flat.push_back(std::move(a));
    }

    ProdCoef coef;
    // base -> accumulated exponent (as a list to be summed)
    struct Fac { Expr base; std::vector<Expr> exps; };
    std::vector<Fac> facs;
    auto add_factor = [&](const Expr& f) {
        Expr b = f, e = one();
        if (f->has_head("Power", 2)) { b = f->args[0]; e = f->args[1]; }
        for (auto& fc : facs) {
            if (equal(fc.base, b)) { fc.exps.push_back(e); return; }
        }
        facs.push_back({b, {e}});
    };
    for (const auto& f : flat) {
        if (is_const(f)) coef.mul(f);
        else add_factor(f);
    }
    if (coef.is_zero()) {
        bool has_inf = false;
        for (const auto& f : flat) if (f->is_symbol_named("Infinity")) has_inf = true;
        if (!has_inf) return zero();
    }

    // Combine like bases, which may produce new numbers (2^(1/2) * 2^(1/2) = 2) - repeat until stable.
    std::vector<Expr> rest;
    for (int round = 0; round < 4; ++round) {
        rest.clear();
        bool changed = false;
        std::vector<Fac> next;
        for (auto& fc : facs) {
            Expr e = fc.exps.size() == 1 ? fc.exps[0] : plus(fc.exps);
            Expr p = (fc.exps.size() == 1 && fc.exps[0]->is_number() && fc.exps[0]->q.is_one()) ? fc.base : power(fc.base, e);
            if (is_const(p)) { coef.mul(p); changed = true; }
            else if (p->has_head("Times")) {
                for (const auto& sub_f : p->args) {
                    if (is_const(sub_f)) coef.mul(sub_f);
                    else {
                        Expr b = sub_f, ee = one();
                        if (sub_f->has_head("Power", 2)) { b = sub_f->args[0]; ee = sub_f->args[1]; }
                        bool merged = false;
                        for (auto& nf : next) if (equal(nf.base, b)) { nf.exps.push_back(ee); merged = true; break; }
                        if (!merged) next.push_back({b, {ee}});
                    }
                }
                changed = true;
            } else {
                Expr b = p, ee = one();
                if (p->has_head("Power", 2)) { b = p->args[0]; ee = p->args[1]; }
                bool merged = false;
                for (auto& nf : next) if (equal(nf.base, b)) { nf.exps.push_back(ee); merged = true; changed = true; break; }
                if (!merged) next.push_back({b, {ee}});
            }
        }
        facs = std::move(next);
        if (!changed) break;
    }
    for (auto& fc : facs) {
        Expr e = fc.exps.size() == 1 ? fc.exps[0] : plus(fc.exps);
        if (e->is_number() && e->q.is_one()) rest.push_back(fc.base);
        else if (e->is_number() && e->q.is_zero()) continue;
        else rest.push_back(apply2("Power", fc.base, e));
    }
    if (coef.is_zero()) return zero();
    std::stable_sort(rest.begin(), rest.end(), [](const Expr& a, const Expr& b) { return compare(a, b) < 0; });

    // -1 * (a + b) distributes, as in Mathematica: x - (x + 1) must collapse.
    if (rest.size() == 1 && !coef.real && coef.q == Rational(-1) && rest[0]->has_head("Plus")) {
        std::vector<Expr> terms;
        for (const auto& t : rest[0]->args) terms.push_back(times({minus_one(), t}));
        return plus(std::move(terms));
    }
    // A fractional numeric factor collapses into a sum when every term absorbs it: (2 + 2 x)/2 -> 1 + x, while
    // (1 + x)/2 stays as it is.
    if (rest.size() == 1 && !coef.real && !coef.q.is_integer() && rest[0]->has_head("Plus")) {
        bool all_integer = true;
        for (const auto& t : rest[0]->args) {
            ProdCoef tc;
            Expr tr;
            split_term(t, tc, tr);
            if (tc.real || !(tc.q * coef.q).is_integer()) { all_integer = false; break; }
        }
        if (all_integer) {
            std::vector<Expr> terms;
            for (const auto& t : rest[0]->args) terms.push_back(times({num(coef.q), t}));
            return plus(std::move(terms));
        }
    }
    if (rest.empty()) return coef.expr();
    if (coef.is_one() && rest.size() == 1) return rest[0];
    std::vector<Expr> out;
    if (!coef.is_one()) out.push_back(coef.expr());
    out.insert(out.end(), rest.begin(), rest.end());
    return app("Times", std::move(out));
}

// ---------------------------------------------------------------------------------------------------------------
// Power
// ---------------------------------------------------------------------------------------------------------------
namespace {

// Largest s with s^q dividing n (n > 0), by trial division on small primes plus a perfect-power test of the rest.
BigInt extract_qth_power(BigInt n, int64_t q, BigInt& rest) {
    BigInt s(1);
    int64_t v = 0;
    if (n.to_int64(v) && v > 0) {
        int64_t m = v;
        for (int64_t p = 2; p <= 1000000 && p * p <= m; p += (p == 2 ? 1 : 2)) {
            int cnt = 0;
            while (m % p == 0) { m /= p; ++cnt; }
            for (int k = 0; k < cnt / q; ++k) s = s * BigInt(p);
            int left = cnt % static_cast<int>(q);
            for (int k = 0; k < left; ++k) m *= 1;  // keep the leftover primes in the rest, rebuilt below
            if (left) { for (int k = 0; k < left; ++k) { /* re-multiplied into rest */ } }
            // rebuild: leftover primes go back into m's complement
            BigInt back(1);
            for (int k = 0; k < left; ++k) back = back * BigInt(p);
            rest = rest * back;
        }
        rest = rest * BigInt(m);
        // perfect q-th power check on the cofactor
        if (q == 2) {
            BigInt r = BigInt::isqrt(rest);
            if (r * r == rest && !r.is_zero()) { s = s * r; rest = BigInt(1); }
        }
        return s;
    }
    rest = n;
    if (q == 2) {
        BigInt r = BigInt::isqrt(n);
        if (r * r == n) { rest = BigInt(1); return r; }
    }
    return s;
}

} // namespace

Expr Engine::rational_power(const Rational& b, const Rational& e) {
    if (e.is_integer()) {
        int64_t k = 0;
        if (!e.num.to_int64(k)) throw CasLimitError("exponent too large");
        uint64_t mag = k < 0 ? static_cast<uint64_t>(-(k + 1)) + 1 : static_cast<uint64_t>(k);
        if (mag > 1 && (b.num.bit_length() + b.den.bit_length()) * mag > budget_.max_bits)
            throw CasLimitError("result of exact power is too large");
        if (b.is_zero() && k < 0) throw CasMathError("division by zero");
        return num(b.pow(k));
    }
    int64_t q = 0, p = 0;
    if (!e.den.to_int64(q) || !e.num.to_int64(p) || q > 64) return apply2("Power", num(b), num(e));
    if (b.is_negative()) {
        if (q == 2) {
            // (-n)^(p/2) = I^p * n^(p/2)
            Expr pos = rational_power(-b, e);
            int64_t pm = ((p % 4) + 4) % 4;
            Expr i_pow = pm == 0 ? one() : pm == 1 ? symbol("I") : pm == 2 ? minus_one() : neg(symbol("I"));
            return times({i_pow, pos});
        }
        return apply2("Power", num(b), num(e));
    }
    if (b.is_zero()) return p > 0 ? zero() : throw CasMathError("division by zero");
    // b = n/d = (n * d^(q-1)) / d^q  ->  b^(1/q) = (n * d^(q-1))^(1/q) / d
    BigInt m = b.num * BigInt::pow(b.den, static_cast<uint64_t>(q - 1));
    BigInt rest(1);
    BigInt s = extract_qth_power(m, q, rest);
    Rational root_coeff(s, b.den);               // rational part of b^(1/q)
    // raise to p: coeff^p * rest^(p/q); split p = a*q + c with 0 <= c < q
    int64_t a = p >= 0 ? p / q : -((-p + q - 1) / q);
    int64_t c = p - a * q;
    Rational coeff = root_coeff.pow(p) * Rational::from_bigint(rest).pow(a);
    Expr out_rest;
    if (rest.is_one() || c == 0) return num(coeff);
    out_rest = apply2("Power", num(Rational::from_bigint(rest)), num(Rational(BigInt(c), BigInt(q))));
    if (coeff.is_one()) return out_rest;
    return app("Times", {num(coeff), out_rest});
}

Expr Engine::power(const Expr& b, const Expr& e) {
    tick();
    if (e->is_number()) {
        if (e->q.is_zero()) return one();
        if (e->q.is_one()) return b;
    }
    if (b->is_number() && b->q.is_one()) return one();
    if (b->is_number() && b->q.is_zero() && e->is_number()) {
        if (e->q.is_negative()) throw CasMathError("division by zero");
        return zero();
    }
    if (b->is_number() && e->is_number()) return rational_power(b->q, e->q);
    if (is_const(b) && is_const(e) && (b->kind == Kind::Real || e->kind == Kind::Real)) {
        double bb = to_d(b), ee = to_d(e);
        if (bb < 0 && ee != std::floor(ee)) throw CasUnsupported("complex result is not supported");
        return real(std::pow(bb, ee));
    }
    if (b->kind == Kind::Real && e->is_integer()) {
        int64_t k = 0;
        if (e->q.num.to_int64(k)) return real(std::pow(b->d, static_cast<double>(k)));
    }
    // I^n
    if (b->is_symbol_named("I") && e->is_integer()) {
        int64_t k = 0;
        if (e->q.num.to_int64(k)) {
            int64_t m = ((k % 4) + 4) % 4;
            return m == 0 ? one() : m == 1 ? b : m == 2 ? minus_one() : neg(b);
        }
    }
    // E^Log[x] = x
    if (b->is_symbol_named("E") && e->has_head("Log", 1)) return e->args[0];
    // (a^m)^n = a^(m n) for integer n
    if (b->has_head("Power", 2) && e->is_integer()) return power(b->args[0], times({b->args[1], e}));
    // (a b)^n = a^n b^n for integer n
    if (b->has_head("Times") && e->is_integer()) {
        std::vector<Expr> fs;
        for (const auto& f : b->args) fs.push_back(power(f, e));
        return times(std::move(fs));
    }
    // (a^m)^(p/q) with symbolic a stays as is; Power of sums stays as is.
    if (e->is_integer() && b->has_head("Plus")) {
        int64_t k = 0;
        if (!e->q.num.to_int64(k) || abs_u64(k) > budget_.max_exponent)
            throw CasLimitError("exponent too large");
    }
    return apply2("Power", b, e);
}

// ---------------------------------------------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------------------------------------------
namespace {

// sin/cos/tan at rational multiples of Pi that have simple closed forms. Returns nullptr-like by ok=false.
bool trig_at_pi_multiple(const std::string& f, const Rational& r, Expr& out, Engine& en) {
    // reduce r mod 2 for sin/cos, mod 1 for tan
    Rational two(2);
    // r = n/d; work with value in [0, 2)
    BigInt q, rem;
    BigInt::divmod(r.num, r.den * BigInt(2), q, rem);       // r = (num/den) -> compare to 2: floor(num / (2 den))
    if (rem.is_negative()) rem = rem + r.den * BigInt(2);
    Rational x(rem, r.den);                                 // x in [0, 2)
    // supported denominators: 1, 2, 3, 4, 6 (and 12 skipped)
    int64_t d = 0, n = 0;
    if (!x.den.to_int64(d) || !x.num.to_int64(n)) return false;
    if (d != 1 && d != 2 && d != 3 && d != 4 && d != 6) return false;
    // represent as angle = n/d * Pi, n/d in [0,2). Use a table over sixths/quarters via exact values.
    auto sqrt_of = [&](int64_t k) { return en.power(integer(k), num(Rational(BigInt(1), BigInt(2)))); };
    auto half = [&](Expr e) { return en.times({num(Rational(BigInt(1), BigInt(2))), e}); };
    // normalise to a common denominator 12 for lookup
    int64_t twelfths = n * (12 / d);
    // sin values by twelfths (0..23), only multiples of 2 (sixths) and 3 (quarters) are supported
    if (twelfths % 2 != 0 && twelfths % 3 != 0) return false;
    auto sin12 = [&](int64_t k, Expr& v) -> bool {
        k %= 24;
        bool negate = false;
        if (k >= 12) { negate = true; k -= 12; }
        Expr s;
        switch (k) {
            case 0: case 12: s = zero(); break;
            case 2: s = num(Rational(BigInt(1), BigInt(2))); break;
            case 3: s = half(sqrt_of(2)); break;
            case 4: s = half(sqrt_of(3)); break;
            case 6: s = one(); break;
            case 8: s = half(sqrt_of(3)); break;
            case 9: s = half(sqrt_of(2)); break;
            case 10: s = num(Rational(BigInt(1), BigInt(2))); break;
            default: return false;
        }
        v = negate ? en.neg(s) : s;
        return true;
    };
    Expr sv, cv;
    if (!sin12(twelfths, sv) || !sin12(twelfths + 6, cv)) return false;
    if (f == "Sin") { out = sv; return true; }
    if (f == "Cos") { out = cv; return true; }
    if (f == "Tan") {
        if (cv->is_number() && cv->q.is_zero()) return false;  // pole
        out = en.div(sv, cv);
        return true;
    }
    return false;
}

// Extracts a rational multiple of Pi from e (returns true with r set), e.g. Pi, 2*Pi, Pi/3, 0.
bool as_pi_multiple(const Expr& e, Rational& r) {
    if (e->is_number() && e->q.is_zero()) { r = Rational(0); return true; }
    if (e->is_symbol_named("Pi")) { r = Rational(1); return true; }
    if (e->has_head("Times", 2) && e->args[0]->is_number() && e->args[1]->is_symbol_named("Pi")) { r = e->args[0]->q; return true; }
    return false;
}

} // namespace

Expr Engine::eval_function(const std::string& head, std::vector<Expr> args) {
    tick();
    auto keep = [&]() { return app(head, args); };
    if (head == "Sqrt" && args.size() == 1) return power(args[0], num(Rational(BigInt(1), BigInt(2))));
    if (head == "Exp" && args.size() == 1) return power(symbol("E"), args[0]);
    if (head == "Log" && args.size() == 2) return div(eval_function("Log", {args[1]}), eval_function("Log", {args[0]}));
    if (head == "Factorial" && args.size() == 1) {
        if (args[0]->is_integer() && !args[0]->q.is_negative()) {
            int64_t n = 0;
            if (!args[0]->q.num.to_int64(n) || n > 20000) throw CasLimitError("factorial argument too large");
            BigInt f(1);
            for (int64_t i = 2; i <= n; ++i) { f = f * BigInt(i); tick(); if (f.bit_length() > budget_.max_bits) throw CasLimitError("factorial too large"); }
            return num(Rational::from_bigint(f));
        }
        return keep();
    }
    if (head == "Binomial" && args.size() == 2 && args[0]->is_integer() && args[1]->is_integer()) {
        int64_t n = 0, k = 0;
        if (!args[0]->q.num.to_int64(n) || !args[1]->q.num.to_int64(k)) throw CasLimitError("binomial arguments too large");
        if (k < 0 || n < 0 || k > n) return zero();
        if (k > n - k) k = n - k;
        if (k > 20000) throw CasLimitError("binomial too large");
        BigInt r(1);
        for (int64_t i = 1; i <= k; ++i) {
            r = r * BigInt(n - k + i) / BigInt(i);
            tick();
            if (r.bit_length() > budget_.max_bits) throw CasLimitError("binomial too large");
        }
        return num(Rational::from_bigint(r));
    }
    if (Expr mf = matrix_function(head, args)) return mf;
    if (Expr nf = number_theory(head, args)) return nf;
    if (Expr xf = extended_math(head, args)) return xf;
    if (Expr df = discrete_math(head, args)) return df;
    if (args.size() != 1) return keep();
    const Expr& a = args[0];

    if (head == "Abs") {
        if (a->is_number()) return num(a->q.is_negative() ? -a->q : a->q);
        if (a->kind == Kind::Real) return real(std::fabs(a->d));
        if (a->has_head("Times") && !a->args.empty() && a->args[0]->is_number() && a->args[0]->q.is_negative()) {
            std::vector<Expr> rest(a->args.begin() + 1, a->args.end());
            rest.insert(rest.begin(), num(-a->args[0]->q));
            return times({eval_function("Abs", {times(std::move(rest))})});
        }
        return keep();
    }
    if (head == "Sign") {
        if (a->is_number()) return integer(a->q.sign());
        if (a->kind == Kind::Real) return integer(a->d > 0 ? 1 : a->d < 0 ? -1 : 0);
        return keep();
    }
    if (head == "Floor" || head == "Ceiling") {
        if (a->is_number()) {
            BigInt q, r;
            BigInt::divmod(a->q.num, a->q.den, q, r);
            if (!r.is_zero() && ((head == "Floor" && a->q.is_negative()) || (head == "Ceiling" && !a->q.is_negative()))) q = q + BigInt(head == "Floor" ? -1 : 1);
            return num(Rational::from_bigint(q));
        }
        if (a->kind == Kind::Real) return real(head == "Floor" ? std::floor(a->d) : std::ceil(a->d));
        return keep();
    }

    // Real arguments: machine-precision evaluation.
    if (a->kind == Kind::Real) {
        double x = a->d, r = 0.0;
        if (head == "Sin") r = std::sin(x); else if (head == "Cos") r = std::cos(x); else if (head == "Tan") r = std::tan(x);
        else if (head == "Cot") r = 1.0 / std::tan(x); else if (head == "Sec") r = 1.0 / std::cos(x); else if (head == "Csc") r = 1.0 / std::sin(x);
        else if (head == "ArcTan") r = std::atan(x);
        else if (head == "Sinh") r = std::sinh(x); else if (head == "Cosh") r = std::cosh(x); else if (head == "Tanh") r = std::tanh(x);
        else if (head == "ArcSinh") r = std::asinh(x);
        else if (head == "ArcSin") { if (x < -1 || x > 1) throw CasUnsupported("complex result is not supported"); r = std::asin(x); }
        else if (head == "ArcCos") { if (x < -1 || x > 1) throw CasUnsupported("complex result is not supported"); r = std::acos(x); }
        else if (head == "ArcCosh") { if (x < 1) throw CasUnsupported("complex result is not supported"); r = std::acosh(x); }
        else if (head == "ArcTanh") { if (x <= -1 || x >= 1) throw CasUnsupported("complex result is not supported"); r = std::atanh(x); }
        else if (head == "Log") { if (x <= 0) throw CasUnsupported("complex result is not supported"); r = std::log(x); }
        else return keep();
        return real(r);
    }

    if (head == "Log") {
        if (a->is_number()) {
            if (a->q.is_one()) return zero();
            if (a->q.is_zero()) throw CasMathError("Log[0] is infinite");
            if (a->q.is_negative()) throw CasUnsupported("complex logarithm is not supported");
        }
        if (a->is_symbol_named("E")) return one();
        if (a->has_head("Power", 2) && a->args[0]->is_symbol_named("E") && (a->args[1]->is_integer() || a->args[1]->is_number())) return a->args[1];
        return keep();
    }

    // Parity: pull a negative numeric coefficient out of odd/even functions.
    static const char* odd[] = {"Sin", "Tan", "Cot", "Csc", "ArcSin", "ArcTan", "Sinh", "Tanh", "ArcSinh", "ArcTanh"};
    static const char* even[] = {"Cos", "Sec", "Cosh"};
    bool is_odd = std::any_of(std::begin(odd), std::end(odd), [&](const char* s) { return head == s; });
    bool is_even = std::any_of(std::begin(even), std::end(even), [&](const char* s) { return head == s; });
    if ((is_odd || is_even) && a->has_head("Times") && !a->args.empty() && a->args[0]->is_number() && a->args[0]->q.is_negative()) {
        std::vector<Expr> rest(a->args.begin(), a->args.end());
        rest[0] = num(-rest[0]->q);
        Expr pos = eval_function(head, {times(std::move(rest))});
        return is_odd ? neg(pos) : pos;
    }
    if ((is_odd || is_even) && a->is_number() && a->q.is_negative() && !a->q.is_zero()) {
        Expr pos = eval_function(head, {num(-a->q)});
        return is_odd ? neg(pos) : pos;
    }

    // Exact special values.
    if (a->is_number() && a->q.is_zero()) {
        if (head == "Sin" || head == "Tan" || head == "Sinh" || head == "Tanh" || head == "ArcSin" || head == "ArcTan" || head == "ArcSinh" || head == "ArcTanh") return zero();
        if (head == "Cos" || head == "Cosh" || head == "Sec") return one();
        if (head == "ArcCos") return times({num(Rational(BigInt(1), BigInt(2))), symbol("Pi")});
    }
    if (head == "ArcTan" && a->is_number() && a->q.is_one()) return times({num(Rational(BigInt(1), BigInt(4))), symbol("Pi")});
    if (head == "ArcSin" && a->is_number() && a->q.is_one()) return times({num(Rational(BigInt(1), BigInt(2))), symbol("Pi")});
    if (head == "ArcCos" && a->is_number() && a->q.is_one()) return zero();
    Rational pm;
    if ((head == "Sin" || head == "Cos" || head == "Tan") && as_pi_multiple(a, pm)) {
        Expr v;
        if (trig_at_pi_multiple(head, pm, v, *this)) return v;
    }
    if (head == "Cot" || head == "Sec" || head == "Csc") {
        // reciprocal forms evaluate through their partner when exact
        const char* partner = head == "Cot" ? "Tan" : head == "Sec" ? "Cos" : "Sin";
        Expr v = eval_function(partner, {a});
        if (!v->has_head(partner)) return div(one(), v);
    }
    return keep();
}

// ---------------------------------------------------------------------------------------------------------------
// eval
// ---------------------------------------------------------------------------------------------------------------
Expr Engine::eval(const Expr& e) {
    DepthGuard g(*this);
    tick();
    if (!e->is_apply()) return e;
    return eval_apply(e);
}

Expr Engine::eval_apply(const Expr& e) {
    const std::string& h = e->name;
    // Held / structural heads evaluate their arguments only.
    std::vector<Expr> args;
    args.reserve(e->args.size());
    for (const auto& a : e->args) args.push_back(eval(a));

    if (h == "Plus") return plus(std::move(args));
    if (h == "Times") return times(std::move(args));
    if (h == "Power" && args.size() == 2) return power(args[0], args[1]);
    if (h == "List" || h == "Equal" || h == "Rule") return app(h, std::move(args));
    if (h == "Expand" && args.size() == 1) return expand(args[0]);
    if (h == "Simplify" && args.size() == 1) return simplify(args[0]);
    if (h == "D" && args.size() >= 2) {
        // D[f, x], D[f, x, y] (mixed partial), D[f, {x, n}] (n-th derivative), and Sage's diff(f, x, n)
        Expr r = args[0];
        for (size_t k = 1; k < args.size(); ++k) {
            const Expr& a = args[k];
            if (a->is_symbol()) {
                int n = 1;
                if (k + 1 < args.size() && args[k + 1]->is_integer()) {   // diff(f, x, 2)
                    int64_t t = 0;
                    if (!args[k + 1]->q.num.to_int64(t) || t < 0 || t > 64) throw CasMathError("derivative order must be between 0 and 64");
                    n = static_cast<int>(t);
                    ++k;
                }
                for (int i = 0; i < n; ++i) r = diff(r, a->name);
            } else if (a->has_head("List", 2) && a->args[0]->is_symbol() && a->args[1]->is_integer()) {
                int64_t t = 0;
                if (!a->args[1]->q.num.to_int64(t) || t < 0 || t > 64) throw CasMathError("derivative order must be between 0 and 64");
                for (int i = 0; i < t; ++i) r = diff(r, a->args[0]->name);
            } else {
                throw CasParseError("D: the variables must be symbols or {symbol, order}");
            }
        }
        return r;
    }
    if (h == "Factor" && args.size() == 1) return factor(args[0]);
    if ((h == "Together" || h == "Cancel") && args.size() == 1) return together(args[0]);
    if (h == "Solve" && args.size() >= 1) {
        std::vector<std::string> vars;
        if (args.size() >= 2) {
            if (args[1]->is_symbol()) vars.push_back(args[1]->name);
            else if (args[1]->has_head("List")) for (const auto& v : args[1]->args) { if (!v->is_symbol()) throw CasParseError("Solve: variables must be symbols"); vars.push_back(v->name); }
        }
        return solve(args[0], vars);
    }
    if (h == "Integrate" && args.size() == 2) {
        if (args[1]->is_symbol()) return integrate(args[0], args[1]->name);
        if (args[1]->has_head("List", 3) && args[1]->args[0]->is_symbol())
            return integrate_definite(args[0], args[1]->args[0]->name, args[1]->args[1], args[1]->args[2]);
        throw CasParseError("Integrate: second argument must be x or {x, a, b}");
    }
    if (h == "Limit" && args.size() == 2 && args[1]->has_head("Rule", 2) && args[1]->args[0]->is_symbol())
        return limit(args[0], args[1]->args[0]->name, args[1]->args[1], 0);
    if (h == "Series" && args.size() == 2 && args[1]->has_head("List", 3) && args[1]->args[0]->is_symbol() && args[1]->args[2]->is_integer()) {
        int64_t n = 0;
        args[1]->args[2]->q.num.to_int64(n);
        return series(args[0], args[1]->args[0]->name, args[1]->args[1], n);
    }
    if (h == "N" && args.size() >= 1) {
        double v = 0;
        if (numeric_value(args[0], v)) return real(v);
        return args[0];
    }
    return eval_function(h, std::move(args));
}

// ---------------------------------------------------------------------------------------------------------------
// Expand
// ---------------------------------------------------------------------------------------------------------------
Expr Engine::expand(const Expr& e) { return eval(expand_rec(eval(e))); }

Expr Engine::expand_product(const Expr& a, const Expr& b) {
    const std::vector<Expr> as = a->has_head("Plus") ? a->args : std::vector<Expr>{a};
    const std::vector<Expr> bs = b->has_head("Plus") ? b->args : std::vector<Expr>{b};
    if (as.size() * bs.size() > budget_.max_terms) throw CasLimitError("expansion has too many terms");
    std::vector<Expr> out;
    out.reserve(as.size() * bs.size());
    for (const auto& x : as)
        for (const auto& y : bs) out.push_back(times({x, y}));
    return plus(std::move(out));
}

Expr Engine::expand_rec(const Expr& e) {
    DepthGuard g(*this);
    tick();
    if (!e->is_apply()) return e;
    if (e->name == "Plus") {
        std::vector<Expr> ts;
        for (const auto& a : e->args) ts.push_back(expand_rec(a));
        return plus(std::move(ts));
    }
    if (e->name == "Times") {
        Expr acc = one();
        for (const auto& a : e->args) acc = expand_product(acc, expand_rec(a));
        return acc;
    }
    if (e->name == "Power" && e->args.size() == 2) {
        Expr b = expand_rec(e->args[0]);
        const Expr& x = e->args[1];
        if (x->is_integer() && !x->q.is_negative() && b->has_head("Plus")) {
            int64_t n = 0;
            if (!x->q.num.to_int64(n) || static_cast<uint64_t>(n) > budget_.max_exponent) throw CasLimitError("exponent too large");
            Expr result = one(), base = b;
            int64_t k = n;
            while (k > 0) {
                if (k & 1) result = expand_product(result, base);
                k >>= 1;
                if (k) base = expand_product(base, base);
            }
            return result;
        }
        return power(b, x);
    }
    return e;
}

// ---------------------------------------------------------------------------------------------------------------
// D
// ---------------------------------------------------------------------------------------------------------------
Expr Engine::diff(const Expr& f, const std::string& var) { return eval(diff_rec(eval(f), var)); }

Expr Engine::diff_rec(const Expr& f, const std::string& var) {
    DepthGuard g(*this);
    tick();
    if (f->is_symbol()) return f->name == var ? one() : zero();
    if (!f->is_apply()) return zero();
    if (!depends_on(f, var)) return zero();
    const std::string& h = f->name;
    if (h == "Plus") {
        std::vector<Expr> ts;
        for (const auto& t : f->args) if (depends_on(t, var)) ts.push_back(diff_rec(t, var));
        return plus(std::move(ts));
    }
    if (h == "Times") {
        std::vector<Expr> ts;
        for (size_t i = 0; i < f->args.size(); ++i) {
            if (!depends_on(f->args[i], var)) continue;
            std::vector<Expr> fs;
            for (size_t j = 0; j < f->args.size(); ++j) fs.push_back(j == i ? diff_rec(f->args[j], var) : f->args[j]);
            ts.push_back(times(std::move(fs)));
        }
        return plus(std::move(ts));
    }
    if (h == "Power" && f->args.size() == 2) {
        const Expr& b = f->args[0];
        const Expr& x = f->args[1];
        std::vector<Expr> ts;
        if (depends_on(b, var)) ts.push_back(times({x, power(b, plus({x, minus_one()})), diff_rec(b, var)}));
        if (depends_on(x, var)) {
            Expr lg = b->is_symbol_named("E") ? one() : eval_function("Log", {b});
            ts.push_back(times({f, lg, diff_rec(x, var)}));
        }
        return plus(std::move(ts));
    }
    if (f->args.size() == 1) {
        const Expr& u = f->args[0];
        Expr du = diff_rec(u, var);
        Expr outer;
        auto sq = [&](const Expr& v) { return power(v, integer(2)); };
        auto minus_half = num(Rational(BigInt(-1), BigInt(2)));
        if (h == "Sin") outer = eval_function("Cos", {u});
        else if (h == "Cos") outer = neg(eval_function("Sin", {u}));
        else if (h == "Tan") outer = power(eval_function("Cos", {u}), integer(-2));
        else if (h == "Cot") outer = neg(power(eval_function("Sin", {u}), integer(-2)));
        else if (h == "Sec") outer = times({eval_function("Sec", {u}), eval_function("Tan", {u})});
        else if (h == "Csc") outer = neg(times({eval_function("Csc", {u}), eval_function("Cot", {u})}));
        else if (h == "ArcSin") outer = power(plus({one(), neg(sq(u))}), minus_half);
        else if (h == "ArcCos") outer = neg(power(plus({one(), neg(sq(u))}), minus_half));
        else if (h == "ArcTan") outer = power(plus({one(), sq(u)}), minus_one());
        else if (h == "Sinh") outer = eval_function("Cosh", {u});
        else if (h == "Cosh") outer = eval_function("Sinh", {u});
        else if (h == "Tanh") outer = power(eval_function("Cosh", {u}), integer(-2));
        else if (h == "ArcSinh") outer = power(plus({sq(u), one()}), minus_half);
        else if (h == "ArcCosh") outer = power(plus({sq(u), minus_one()}), minus_half);
        else if (h == "ArcTanh") outer = power(plus({one(), neg(sq(u))}), minus_one());
        else if (h == "Log") outer = power(u, minus_one());
        else if (h == "Abs") outer = eval_function("Sign", {u});
        else if (h == "Sign") return zero();
        else throw CasUnsupported("no derivative rule for " + h);
        return times({outer, du});
    }
    throw CasUnsupported("no derivative rule for " + h);
}

// ---------------------------------------------------------------------------------------------------------------
// Simplify (phase 1: canonical form, expansion, Pythagorean identities)
// ---------------------------------------------------------------------------------------------------------------
Expr Engine::pythagorean(const Expr& e) {
    DepthGuard g(*this);
    if (!e->is_apply()) return e;
    std::vector<Expr> args;
    for (const auto& a : e->args) args.push_back(pythagorean(a));
    Expr cur = eval(app(e->name, args));
    if (!cur->has_head("Plus")) return cur;
    // c*R*Sin(u)^2 + c*R*Cos(u)^2 -> c*R   and   c*R*Cosh(u)^2 - c*R*Sinh(u)^2 -> c*R
    struct Info { ProdCoef c; Expr rest; std::string fn; Expr u; int idx; };
    auto classify = [&](const Expr& t, int idx, Info& out) {
        ProdCoef c;
        Expr rest;
        split_term(t, c, rest);
        std::vector<Expr> others;
        std::vector<Expr> fs = rest->has_head("Times") ? rest->args : std::vector<Expr>{rest};
        bool found = false;
        for (const auto& f : fs) {
            if (!found && f->has_head("Power", 2) && f->args[1]->is_number() && f->args[1]->q == Rational(2) &&
                f->args[0]->is_apply() && f->args[0]->args.size() == 1 &&
                (f->args[0]->name == "Sin" || f->args[0]->name == "Cos" || f->args[0]->name == "Sinh" || f->args[0]->name == "Cosh")) {
                out.fn = f->args[0]->name;
                out.u = f->args[0]->args[0];
                found = true;
            } else {
                others.push_back(f);
            }
        }
        if (!found) return false;
        out.c = c;
        out.rest = others.empty() ? one() : (others.size() == 1 ? others[0] : app("Times", others));
        out.idx = idx;
        return true;
    };
    std::vector<Info> infos;
    for (size_t i = 0; i < cur->args.size(); ++i) {
        Info in;
        if (classify(cur->args[i], static_cast<int>(i), in)) infos.push_back(in);
    }
    std::vector<bool> used(cur->args.size(), false);
    std::vector<Expr> extra;
    for (size_t i = 0; i < infos.size(); ++i) {
        if (used[infos[i].idx]) continue;
        for (size_t j = i + 1; j < infos.size(); ++j) {
            if (used[infos[j].idx]) continue;
            if (!equal(infos[i].rest, infos[j].rest) || !equal(infos[i].u, infos[j].u)) continue;
            bool trig = (infos[i].fn == "Sin" && infos[j].fn == "Cos") || (infos[i].fn == "Cos" && infos[j].fn == "Sin");
            bool hyp = (infos[i].fn == "Cosh" && infos[j].fn == "Sinh") || (infos[i].fn == "Sinh" && infos[j].fn == "Cosh");
            if (!infos[i].c.real && !infos[j].c.real) {
                if (trig && infos[i].c.q == infos[j].c.q) {
                    extra.push_back(make_term(infos[i].c, infos[i].rest));
                    used[infos[i].idx] = used[infos[j].idx] = true;
                    break;
                }
                if (hyp && infos[i].c.q == -infos[j].c.q) {
                    const Info& ch = infos[i].fn == "Cosh" ? infos[i] : infos[j];
                    extra.push_back(make_term(ch.c, ch.rest));
                    used[infos[i].idx] = used[infos[j].idx] = true;
                    break;
                }
            }
        }
    }
    if (extra.empty()) return cur;
    std::vector<Expr> out = extra;
    for (size_t i = 0; i < cur->args.size(); ++i) if (!used[i]) out.push_back(cur->args[i]);
    return plus(std::move(out));
}

Expr Engine::simplify(const Expr& e) {
    Expr e1 = eval(e);
    Expr best = e1;
    auto consider = [&](const Expr& c) { if (leaf_count(c) < leaf_count(best)) best = c; };
    Expr e2 = expand(e1);
    consider(e2);
    consider(pythagorean(e1));
    consider(pythagorean(e2));
    return best;
}

// ---------------------------------------------------------------------------------------------------------------
// numeric value
// ---------------------------------------------------------------------------------------------------------------
bool Engine::numeric_value(const Expr& e, double& out) {
    Expr v = eval(e);
    return numeric_rec(v, out);
}

bool Engine::numeric_rec(const Expr& e, double& out) {
    DepthGuard g(*this);
    switch (e->kind) {
        case Kind::Number: out = e->q.to_double(); return true;
        case Kind::Real: out = e->d; return true;
        case Kind::Symbol:
            if (e->name == "Pi") { out = M_PI; return true; }
            if (e->name == "E") { out = M_E; return true; }
            return false;
        case Kind::Apply: break;
    }
    std::vector<double> v;
    for (const auto& a : e->args) {
        double x = 0;
        if (!numeric_rec(a, x)) return false;
        v.push_back(x);
    }
    const std::string& h = e->name;
    if (h == "Plus") { double s = 0; for (double x : v) s += x; out = s; return true; }
    if (h == "Times") { double s = 1; for (double x : v) s *= x; out = s; return true; }
    if (h == "Power" && v.size() == 2) {
        if (v[0] < 0 && v[1] != std::floor(v[1])) return false;
        out = std::pow(v[0], v[1]);
        return std::isfinite(out);
    }
    if (v.size() == 1) {
        double x = v[0];
        if (h == "Sin") out = std::sin(x); else if (h == "Cos") out = std::cos(x); else if (h == "Tan") out = std::tan(x);
        else if (h == "Cot") out = 1.0 / std::tan(x); else if (h == "Sec") out = 1.0 / std::cos(x); else if (h == "Csc") out = 1.0 / std::sin(x);
        else if (h == "ArcTan") out = std::atan(x); else if (h == "Sinh") out = std::sinh(x); else if (h == "Cosh") out = std::cosh(x);
        else if (h == "Tanh") out = std::tanh(x); else if (h == "ArcSinh") out = std::asinh(x);
        else if (h == "ArcSin") { if (x < -1 || x > 1) return false; out = std::asin(x); }
        else if (h == "ArcCos") { if (x < -1 || x > 1) return false; out = std::acos(x); }
        else if (h == "ArcCosh") { if (x < 1) return false; out = std::acosh(x); }
        else if (h == "ArcTanh") { if (x <= -1 || x >= 1) return false; out = std::atanh(x); }
        else if (h == "Log") { if (x <= 0) return false; out = std::log(x); }
        else if (h == "Abs") out = std::fabs(x);
        else if (h == "Sign") out = x > 0 ? 1 : x < 0 ? -1 : 0;
        else return false;
        return std::isfinite(out);
    }
    return false;
}


// ---------------------------------------------------------------------------------------------------------------
// substitution / witnesses
// ---------------------------------------------------------------------------------------------------------------
Expr Engine::substitute(const Expr& e, const std::vector<std::pair<std::string, Expr>>& map) {
    if (e->is_symbol()) {
        for (const auto& m : map) if (m.first == e->name) return m.second;
        return e;
    }
    if (!e->is_apply()) return e;
    std::vector<Expr> args;
    args.reserve(e->args.size());
    for (const auto& a : e->args) args.push_back(substitute(a, map));
    return app(e->name, std::move(args));
}

bool Engine::find_nonzero_witness(const Expr& e, std::string& witness, int* evaluated) {
    if (evaluated) *evaluated = 0;
    std::vector<std::string> syms;
    free_symbols(e, syms);
    static const double samples[] = {0.7, 1.3, -0.45, 2.1, 0.123, -1.9, 3.7, 0.31};
    const size_t n_samples = sizeof(samples) / sizeof(samples[0]);
    for (size_t k = 0; k < n_samples; ++k) {
        std::vector<std::pair<std::string, Expr>> point;
        std::string desc;
        for (size_t i = 0; i < syms.size(); ++i) {
            double v = samples[(k + 3 * i) % n_samples] * (1.0 + 0.11 * static_cast<double>(i));
            point.emplace_back(syms[i], real(v));
            char buf[48];
            std::snprintf(buf, sizeof buf, "%s%s = %.6g", i ? ", " : "", syms[i].c_str(), v);
            desc += buf;
        }
        try {
            double val = 0.0;
            if (!numeric_value(substitute(e, point), val)) continue;
            if (evaluated) ++*evaluated;
            // Compare against the size of the terms so cancellation noise is not mistaken for a difference.
            double scale = 1.0;
            if (e->has_head("Plus")) {
                scale = 0.0;
                for (const auto& t : e->args) {
                    double tv = 0.0;
                    if (numeric_value(substitute(t, point), tv)) scale = std::max(scale, std::fabs(tv));
                }
                if (scale < 1.0) scale = 1.0;
            }
            if (std::fabs(val) > 1e-8 * scale) {
                char buf[64];
                std::snprintf(buf, sizeof buf, "%.6g", val);
                witness = (desc.empty() ? std::string("value") : desc) + " gives " + buf;
                return true;
            }
        } catch (const CasLimitError&) {
            throw;
        } catch (const CasError&) {
            continue;
        }
    }
    return false;
}

} // namespace strata::math::cas
