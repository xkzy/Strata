// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/cas/algebra.cpp - Cancel, Together, Factor, Solve (what Mathics delegates to SymPy), implemented natively
#include "strata/math/cas/engine.hpp"

#include <algorithm>
#include <map>

namespace strata::math::cas {

namespace {

Poly poly_pow(const Poly& a, int64_t n, const TickFn& tick) {
    Poly result{Rational(1)};
    Poly base = a;
    while (n > 0) {
        tick(base.size() * result.size() + 1);
        if (n & 1) result = poly_mul(result, base);
        n >>= 1;
        if (n) base = poly_mul(base, base);
    }
    return result;
}

// Divides out the gcd of two polynomials and makes the denominator's leading coefficient 1.
void reduce(Poly& num, Poly& den) {
    if (num.empty()) { den = Poly{Rational(1)}; return; }
    Poly g = poly_gcd(num, den);
    if (poly_degree(g) > 0) {
        Poly q, r;
        poly_divmod(num, g, q, r); num = q;
        poly_divmod(den, g, q, r); den = q;
    }
    Rational lc = den.back();
    if (!lc.is_one()) { num = poly_scale(num, lc.reciprocal()); den = poly_scale(den, lc.reciprocal()); }
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------
// Expr <-> Poly
// ---------------------------------------------------------------------------------------------------------------
bool Engine::poly_coefficients(const Expr& e0, const std::string& var, std::vector<Expr>& coeffs) {
    Expr e = expand(e0);
    std::map<int64_t, std::vector<Expr>> by_degree;
    const std::vector<Expr> terms = e->has_head("Plus") ? e->args : std::vector<Expr>{e};
    for (const auto& t : terms) {
        std::vector<Expr> factors = t->has_head("Times") ? t->args : std::vector<Expr>{t};
        int64_t k = 0;
        std::vector<Expr> coef;
        for (const auto& f : factors) {
            if (!depends_on(f, var)) { coef.push_back(f); continue; }
            if (f->is_symbol() && f->name == var) { ++k; continue; }
            if (f->has_head("Power", 2) && f->args[0]->is_symbol() && f->args[0]->name == var && f->args[1]->is_integer() &&
                !f->args[1]->q.is_negative()) {
                int64_t n = 0;
                if (!f->args[1]->q.num.to_int64(n) || n > 100000) throw CasLimitError("polynomial degree too large");
                k += n;
                continue;
            }
            return false;
        }
        by_degree[k].push_back(coef.empty() ? one() : times(coef));
    }
    int64_t max_k = by_degree.empty() ? -1 : by_degree.rbegin()->first;
    coeffs.assign(static_cast<size_t>(max_k + 1), zero());
    for (auto& [k, v] : by_degree) coeffs[static_cast<size_t>(k)] = plus(v);
    while (!coeffs.empty() && coeffs.back()->is_number() && coeffs.back()->q.is_zero()) coeffs.pop_back();
    return true;
}

Expr Engine::poly_to_expr(const Poly& p, const std::string& var) {
    std::vector<Expr> terms;
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i].is_zero()) continue;
        Expr c = num(p[i]);
        if (i == 0) terms.push_back(c);
        else if (i == 1) terms.push_back(p[i].is_one() ? symbol(var) : app("Times", {c, symbol(var)}));
        else {
            Expr pw = app("Power", {symbol(var), integer(static_cast<int64_t>(i))});
            terms.push_back(p[i].is_one() ? pw : app("Times", {c, pw}));
        }
    }
    Engine tmp;
    return terms.empty() ? zero() : tmp.plus(terms);
}

bool Engine::rational_function(const Expr& e, const std::string& var, Poly& num_p, Poly& den_p) {
    DepthGuard g(*this);
    tick();
    const TickFn tk = [this](uint64_t n) { tick(n); };
    switch (e->kind) {
        case Kind::Number: num_p = e->q.is_zero() ? Poly{} : Poly{e->q}; den_p = Poly{Rational(1)}; return true;
        case Kind::Real: return false;
        case Kind::Symbol:
            if (e->name == var) { num_p = Poly{Rational(0), Rational(1)}; den_p = Poly{Rational(1)}; return true; }
            return false;
        case Kind::Apply: break;
    }
    if (e->name == "Plus") {
        Poly n{}, d{Rational(1)};
        for (const auto& a : e->args) {
            Poly an, ad;
            if (!rational_function(a, var, an, ad)) return false;
            n = poly_add(poly_mul(n, ad), poly_mul(an, d));
            d = poly_mul(d, ad);
            tick(n.size() + d.size());
            Poly gg = poly_gcd(n, d);
            if (poly_degree(gg) > 0) { Poly q, r; poly_divmod(n, gg, q, r); n = q; poly_divmod(d, gg, q, r); d = q; }
        }
        reduce(n, d);
        num_p = n; den_p = d;
        return true;
    }
    if (e->name == "Times") {
        Poly n{Rational(1)}, d{Rational(1)};
        for (const auto& a : e->args) {
            Poly an, ad;
            if (!rational_function(a, var, an, ad)) return false;
            n = poly_mul(n, an);
            d = poly_mul(d, ad);
            tick(n.size() + d.size());
        }
        if (n.empty()) { num_p = Poly{}; den_p = Poly{Rational(1)}; return true; }
        reduce(n, d);
        num_p = n; den_p = d;
        return true;
    }
    if (e->name == "Power" && e->args.size() == 2 && e->args[1]->is_integer()) {
        int64_t k = 0;
        if (!e->args[1]->q.num.to_int64(k) || abs_u64(k) > budget_.max_exponent) throw CasLimitError("exponent too large");
        Poly bn, bd;
        if (!rational_function(e->args[0], var, bn, bd)) return false;
        if (k >= 0) { num_p = poly_pow(bn, k, tk); den_p = poly_pow(bd, k, tk); }
        else {
            if (bn.empty()) throw CasMathError("division by zero");
            num_p = poly_pow(bd, -k, tk); den_p = poly_pow(bn, -k, tk);
        }
        reduce(num_p, den_p);
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------------------------------------------
// Cancel / Together / Factor
// ---------------------------------------------------------------------------------------------------------------
static std::string single_variable(const Expr& e) {
    std::vector<std::string> syms;
    free_symbols(e, syms);
    return syms.size() == 1 ? syms[0] : std::string();
}

Expr Engine::together(const Expr& e0) {
    Expr e = eval(e0);
    std::string var = single_variable(e);
    if (var.empty()) return e;
    Poly n, d;
    if (!rational_function(e, var, n, d)) return e;
    Expr ne = poly_to_expr(n, var), de = poly_to_expr(d, var);
    return poly_degree(d) == 0 && d[0].is_one() ? ne : div(ne, de);
}

Expr Engine::cancel(const Expr& e) { return together(e); }

Expr Engine::factor(const Expr& e0) {
    Expr e = eval(e0);
    if (e->is_number() || e->kind == Kind::Real || e->is_symbol()) return e;
    std::string var = single_variable(e);
    if (var.empty()) throw CasUnsupported("factoring with several variables is not implemented");
    Poly n, d;
    if (!rational_function(e, var, n, d)) throw CasUnsupported("factor needs a rational function of one variable");
    const TickFn tk = [this](uint64_t k) { tick(k); };
    auto factor_poly = [&](const Poly& p) -> Expr {
        if (p.empty()) return zero();
        Factorization f = factor_over_q(p, tk);
        std::vector<Expr> parts;
        if (!f.content.is_one()) parts.push_back(num(f.content));
        for (const auto& [h, m] : f.factors) {
            Expr base = poly_to_expr(h, var);
            parts.push_back(m == 1 ? base : app("Power", {base, integer(m)}));
        }
        if (parts.empty()) return one();
        // keep factors unexpanded, in factor_over_q's order (constant, then by degree and coefficients)
        return parts.size() == 1 ? parts[0] : app("Times", parts);
    };
    Expr fn = factor_poly(n);
    if (poly_degree(d) == 0 && d[0].is_one()) return fn;
    return div(fn, factor_poly(d));
}

// ---------------------------------------------------------------------------------------------------------------
// Solve
// ---------------------------------------------------------------------------------------------------------------
namespace {

Expr rule(const std::string& var, Expr v) { return app("Rule", {symbol(var), std::move(v)}); }

} // namespace

Expr Engine::solve(const Expr& eqs0, const std::vector<std::string>& vars_in) {
    Expr eqs = eqs0;
    std::vector<Expr> equations;
    if (eqs->has_head("List")) equations = eqs->args; else equations.push_back(eqs);
    std::vector<Expr> zeros;  // each expression is required to be 0
    for (const auto& q : equations) {
        Expr qe = eval(q);
        if (qe->has_head("Equal", 2)) zeros.push_back(eval(sub(qe->args[0], qe->args[1])));
        else zeros.push_back(qe);
    }
    std::vector<std::string> vars = vars_in;
    if (vars.empty()) {
        std::vector<std::string> syms;
        for (const auto& z : zeros) { std::vector<std::string> s; free_symbols(z, s); syms.insert(syms.end(), s.begin(), s.end()); }
        std::sort(syms.begin(), syms.end());
        syms.erase(std::unique(syms.begin(), syms.end()), syms.end());
        if (syms.size() == 1) vars = syms;
        else if (std::find(syms.begin(), syms.end(), "x") != syms.end() && zeros.size() == 1) vars = {"x"};
        else if (syms.size() == zeros.size()) vars = syms;
        else if (syms.empty()) {
            // constants only: the "equation" is just true or false
            for (const auto& z : zeros) if (!(z->is_number() && z->q.is_zero())) return app("List", {});
            return app("List", {app("List", {})});
        }
        else throw CasUnsupported("cannot tell which variables to solve for");
    }

    // ---- single equation, single variable ----
    if (zeros.size() == 1 && vars.size() == 1) {
        const std::string& x = vars[0];
        Expr f = zeros[0];
        if (!depends_on(f, x)) {
            if (f->is_number() && f->q.is_zero()) throw CasUnsupported("the equation is an identity: every value solves it");
            if (f->is_number()) return app("List", {});   // non-zero constant: no solutions
            throw CasUnsupported("equation does not contain the variable");
        }
        // Rational function with rational coefficients: numerator roots minus denominator roots.
        Poly n, d;
        std::vector<Expr> coeffs;
        std::vector<Expr> roots;
        const TickFn tk = [this](uint64_t k) { tick(k); };
        Poly rn, rd;
        bool rat = false;
        {
            std::vector<std::string> syms;
            free_symbols(f, syms);
            if (syms.size() == 1 && syms[0] == x) rat = rational_function(f, x, rn, rd);
        }
        if (rat) {
            if (rn.empty()) throw CasUnsupported("the equation is an identity: every value solves it");
            if (poly_degree(rn) == 0) return app("List", {});
            Factorization fz = factor_over_q(rn, tk);
            for (const auto& [h, m] : fz.factors) {
                (void)m;
                int deg = poly_degree(h);
                if (deg == 1) {
                    roots.push_back(num(-h[0] / h[1]));
                } else if (deg == 2) {
                    Rational a = h[2], b = h[1], c = h[0];
                    Expr disc = eval(num(b * b - Rational(4) * a * c));
                    Expr sq = power(disc, num(Rational(BigInt(1), BigInt(2))));
                    Expr two_a = num(Rational(2) * a);
                    roots.push_back(div(plus({num(-b), neg(sq)}), two_a));
                    roots.push_back(div(plus({num(-b), sq}), two_a));
                } else {
                    throw CasUnsupported("irreducible factor of degree " + std::to_string(deg) +
                                         " has no closed form in this CAS (complete solution set not available)");
                }
            }
        } else {
            // symbolic coefficients: only degree 1 and 2 have a generic closed form
            if (!poly_coefficients(f, x, coeffs)) throw CasUnsupported("equation is not a polynomial in " + x);
            int deg = static_cast<int>(coeffs.size()) - 1;
            if (deg == 1) {
                roots.push_back(div(neg(coeffs[0]), coeffs[1]));
            } else if (deg == 2) {
                const Expr &c = coeffs[0], &b = coeffs[1], &a = coeffs[2];
                Expr disc = plus({times({b, b}), neg(times({integer(4), a, c}))});
                Expr sq = power(disc, num(Rational(BigInt(1), BigInt(2))));
                Expr two_a = times({integer(2), a});
                roots.push_back(div(plus({neg(b), neg(sq)}), two_a));
                roots.push_back(div(plus({neg(b), sq}), two_a));
            } else {
                throw CasUnsupported("symbolic equation of degree " + std::to_string(deg) + " is not supported");
            }
        }
        // distinct roots, deterministic order
        std::vector<Expr> uniq;
        for (auto& r : roots) {
            bool dup = false;
            for (auto& u : uniq) if (equal(u, r)) { dup = true; break; }
            if (!dup) uniq.push_back(r);
        }
        auto numeric_key = [&](const Expr& r) {
            double v = 0;
            if (numeric_value(r, v)) return v;
            return 1e300;  // complex or symbolic: after the real ones
        };
        std::stable_sort(uniq.begin(), uniq.end(), [&](const Expr& a, const Expr& b) { return numeric_key(a) < numeric_key(b); });
        std::vector<Expr> sets;
        for (const auto& r : uniq) sets.push_back(app("List", {rule(x, r)}));
        return app("List", sets);
    }

    // ---- linear system with rational coefficients ----
    if (vars.size() == zeros.size() || (vars.size() >= 1 && zeros.size() >= 1)) {
        const size_t nv = vars.size(), ne = zeros.size();
        std::vector<std::vector<Rational>> M(ne, std::vector<Rational>(nv + 1, Rational(0)));
        for (size_t i = 0; i < ne; ++i) {
            Expr f = expand(zeros[i]);
            const std::vector<Expr> terms = f->has_head("Plus") ? f->args : std::vector<Expr>{f};
            for (const auto& t : terms) {
                Rational c(1);
                int which = -1;
                std::vector<Expr> factors = t->has_head("Times") ? t->args : std::vector<Expr>{t};
                for (const auto& fac : factors) {
                    if (fac->is_number()) { c = c * fac->q; continue; }
                    bool matched = false;
                    for (size_t v = 0; v < nv; ++v) {
                        if (fac->is_symbol() && fac->name == vars[v]) {
                            if (which != -1) throw CasUnsupported("system is not linear");
                            which = static_cast<int>(v); matched = true; break;
                        }
                    }
                    if (!matched) throw CasUnsupported("only linear systems with rational coefficients are supported");
                }
                if (which < 0) M[i][nv] = M[i][nv] - c;   // constant moves to the right-hand side
                else M[i][which] = M[i][which] + c;
            }
        }
        // Gauss-Jordan elimination
        std::vector<int> pivot_col;
        size_t row = 0;
        for (size_t col = 0; col < nv && row < ne; ++col) {
            size_t pr = row;
            while (pr < ne && M[pr][col].is_zero()) ++pr;
            if (pr == ne) continue;
            std::swap(M[pr], M[row]);
            Rational inv = M[row][col].reciprocal();
            for (auto& x : M[row]) x = x * inv;
            for (size_t r2 = 0; r2 < ne; ++r2) {
                if (r2 == row || M[r2][col].is_zero()) continue;
                Rational f = M[r2][col];
                for (size_t k = 0; k <= nv; ++k) M[r2][k] = M[r2][k] - f * M[row][k];
                tick(nv + 1);
            }
            pivot_col.push_back(static_cast<int>(col));
            ++row;
        }
        for (size_t r2 = row; r2 < ne; ++r2) if (!M[r2][nv].is_zero()) return app("List", {});  // inconsistent
        std::vector<bool> is_pivot(nv, false);
        for (int c : pivot_col) is_pivot[c] = true;
        std::vector<Expr> rules;
        for (size_t r2 = 0; r2 < pivot_col.size(); ++r2) {
            int c = pivot_col[r2];
            std::vector<Expr> terms{num(M[r2][nv])};
            for (size_t v = 0; v < nv; ++v) {
                if (!is_pivot[v] && !M[r2][v].is_zero()) terms.push_back(times({num(-M[r2][v]), symbol(vars[v])}));
            }
            rules.push_back(rule(vars[c], plus(terms)));
        }
        if (rules.empty()) throw CasUnsupported("the system does not constrain the variables");
        return app("List", {app("List", rules)});
    }
    throw CasUnsupported("this kind of system is not supported");
}

} // namespace strata::math::cas
