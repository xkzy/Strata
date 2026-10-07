// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/cas/integrate.cpp - Integrate (the Integrate builtin of Mathics delegates to SymPy; implemented natively)
//
// Methods: linearity, rational functions (partial fractions), a table of elementary functions of a linear
// argument, polynomial times exp/sin/cos/log/atan by parts, powers of sin/cos by exact Fourier expansion,
// exp times sin/cos, and u-substitution. Every antiderivative is verified by differentiation before it is returned.
#include "strata/math/cas/engine.hpp"

#include <algorithm>
#include <cmath>

namespace strata::math::cas {

namespace {

Expr replace_expr(const Expr& e, const Expr& from, const Expr& to) {
    if (equal(e, from)) return to;
    if (!e->is_apply()) return e;
    std::vector<Expr> args;
    bool changed = false;
    for (const auto& a : e->args) {
        Expr r = replace_expr(a, from, to);
        if (r.get() != a.get()) changed = true;
        args.push_back(std::move(r));
    }
    return changed ? app(e->name, std::move(args)) : e;
}

// Gaussian rational (re + im i)
struct G {
    Rational re = Rational(0), im = Rational(0);
    G operator*(const G& o) const { return {re * o.re - im * o.im, re * o.im + im * o.re}; }
    G operator+(const G& o) const { return {re + o.re, im + o.im}; }
};

void collect_candidates(const Expr& e, const std::string& x, std::vector<Expr>& out) {
    if (!e->is_apply() || !depends_on(e, x)) return;
    for (const auto& a : e->args) {
        if (a->is_apply() && depends_on(a, x)) {
            bool dup = false;
            for (const auto& o : out) if (equal(o, a)) { dup = true; break; }
            if (!dup) out.push_back(a);
        }
        collect_candidates(a, x, out);
    }
    // exponent / base pieces of a Power are candidates too
    if (e->has_head("Power", 2)) {
        for (const auto& a : e->args) {
            if (a->is_apply() && depends_on(a, x)) {
                bool dup = false;
                for (const auto& o : out) if (equal(o, a)) { dup = true; break; }
                if (!dup) out.push_back(a);
            }
        }
    }
}

} // namespace

struct IntegrateOps {
    Engine& en;
    explicit IntegrateOps(Engine& e) : en(e) {}

    Expr half() { return num(Rational(BigInt(1), BigInt(2))); }

    // u = a*x + b with a, b free of x and a != 0
    bool linear(const Expr& u, const std::string& x, Expr& a, Expr& b) {
        std::vector<Expr> c;
        if (!en.poly_coefficients(u, x, c) || c.size() != 2) return false;
        a = c[1]; b = c[0];
        return !(a->is_number() && a->q.is_zero());
    }

    // ---- rational functions ----
    Expr rational(const Poly& N, const Poly& D, const std::string& x) {
        const TickFn tk = [this](uint64_t k) { en.tick(k); };
        Poly Q, R;
        poly_divmod(N, D, Q, R);
        std::vector<Expr> terms;
        // polynomial part
        for (size_t i = 0; i < Q.size(); ++i) {
            if (Q[i].is_zero()) continue;
            terms.push_back(en.times({num(Q[i] / Rational(static_cast<int64_t>(i + 1))), en.power(symbol(x), integer(static_cast<int64_t>(i + 1)))}));
        }
        if (R.empty()) return en.plus(terms);

        Factorization fz = factor_over_q(D, tk);
        // Dp = prod f^m
        std::vector<std::pair<Poly, int>> fac = fz.factors;
        Poly Dp{Rational(1)};
        for (auto& [f, m] : fac) for (int k = 0; k < m; ++k) Dp = poly_mul(Dp, f);
        const Rational c = fz.content;                       // D = c * Dp
        // unknowns: for each (i, j) a polynomial P_ij of degree < deg f_i
        struct Unk { size_t fi; int j; int d; };
        std::vector<Unk> unk;
        for (size_t i = 0; i < fac.size(); ++i)
            for (int j = 1; j <= fac[i].second; ++j)
                for (int d = 0; d < poly_degree(fac[i].first); ++d) unk.push_back({i, j, d});
        const size_t n = unk.size();
        if (static_cast<int>(n) != poly_degree(Dp)) throw CasUnsupported("partial fractions: inconsistent degrees");
        // R / c = sum P_ij * (Dp / f_i^j)   ->  linear system on coefficients
        std::vector<std::vector<Rational>> M(n, std::vector<Rational>(n + 1, Rational(0)));
        Poly Rc = poly_scale(R, c.reciprocal());
        for (size_t col = 0; col < n; ++col) {
            Poly fj{Rational(1)};
            for (int k = 0; k < unk[col].j; ++k) fj = poly_mul(fj, fac[unk[col].fi].first);
            Poly cof, rem;
            poly_divmod(Dp, fj, cof, rem);
            Poly basis(static_cast<size_t>(unk[col].d) + 1, Rational(0));
            basis[static_cast<size_t>(unk[col].d)] = Rational(1);
            Poly term = poly_mul(basis, cof);
            for (size_t r2 = 0; r2 < term.size() && r2 < n; ++r2) M[r2][col] = term[r2];
        }
        for (size_t r2 = 0; r2 < Rc.size() && r2 < n; ++r2) M[r2][n] = Rc[r2];
        // Gauss-Jordan
        for (size_t col = 0, row = 0; col < n; ++col, ++row) {
            size_t pr = row;
            while (pr < n && M[pr][col].is_zero()) ++pr;
            if (pr == n) throw CasUnsupported("partial fractions: singular system");
            std::swap(M[pr], M[row]);
            Rational inv = M[row][col].reciprocal();
            for (auto& v : M[row]) v = v * inv;
            for (size_t r2 = 0; r2 < n; ++r2) {
                if (r2 == row || M[r2][col].is_zero()) continue;
                Rational f = M[r2][col];
                for (size_t k = 0; k <= n; ++k) M[r2][k] = M[r2][k] - f * M[row][k];
            }
            en.tick(n);
        }
        std::vector<Poly> P(unk.size());
        // regroup coefficients into P_ij
        std::vector<std::vector<Poly>> byfac(fac.size());
        for (size_t i = 0; i < fac.size(); ++i) byfac[i].assign(static_cast<size_t>(fac[i].second) + 1, Poly{});
        for (size_t u = 0; u < n; ++u) {
            Poly& p = byfac[unk[u].fi][static_cast<size_t>(unk[u].j)];
            if (p.size() <= static_cast<size_t>(unk[u].d)) p.resize(static_cast<size_t>(unk[u].d) + 1, Rational(0));
            p[static_cast<size_t>(unk[u].d)] = M[u][n];
        }
        const Expr X = symbol(x);
        for (size_t i = 0; i < fac.size(); ++i) {
            const Poly& f = fac[i].first;
            const int deg = poly_degree(f);
            for (int j = 1; j <= fac[i].second; ++j) {
                Poly p = byfac[i][static_cast<size_t>(j)];
                poly_trim(p);
                if (p.empty()) continue;
                if (deg == 1) {
                    // p / (a x + b)^j
                    Rational a = f[1];
                    Expr fx = Engine::poly_to_expr(f, x);
                    if (j == 1) terms.push_back(en.times({num(p[0] / a), app("Log", {fx})}));
                    else terms.push_back(en.times({num(p[0] / (a * Rational(1 - j))), en.power(fx, integer(1 - j))}));
                } else if (deg == 2) {
                    Rational a = f[2], b = f[1], cc = f[0];
                    Rational p1 = p.size() > 1 ? p[1] : Rational(0), p0 = p[0];
                    Rational A = p1 / (Rational(2) * a);
                    Rational B = p0 - p1 * b / (Rational(2) * a);
                    Expr fx = Engine::poly_to_expr(f, x);
                    if (!A.is_zero()) {
                        if (j == 1) terms.push_back(en.times({num(A), app("Log", {fx})}));
                        else terms.push_back(en.times({num(A / Rational(1 - j)), en.power(fx, integer(1 - j))}));
                    }
                    if (!B.is_zero()) {
                        // D = 4ac - b^2 > 0.  int dx/f   = 2/sqrt(D) * atan((2ax+b)/sqrt(D))
                        //                     int dx/f^n = (2ax+b) / ((n-1) D f^(n-1)) + 2(2n-3)a / ((n-1) D) int dx/f^(n-1)
                        Rational Dd = Rational(4) * a * cc - b * b;
                        if (!(Rational(0) < Dd)) throw CasUnsupported("reducible quadratic reached partial fractions");
                        Expr sq = en.power(num(Dd), half());
                        Expr lin = Engine::poly_to_expr(Poly{b, Rational(2) * a}, x);   // 2a x + b
                        Expr I = en.times({integer(2), app("ArcTan", {en.div(lin, sq)}), en.power(sq, minus_one())});
                        for (int jj = 2; jj <= j; ++jj) {
                            Expr t1 = en.div(lin, en.times({num(Rational(jj - 1) * Dd), en.power(fx, integer(jj - 1))}));
                            Expr t2 = en.times({num(Rational(2) * Rational(2 * jj - 3) * a / (Rational(jj - 1) * Dd)), I});
                            I = en.plus({t1, t2});
                        }
                        terms.push_back(en.times({num(B), I}));
                    }
                } else {
                    // only p proportional to f' with j == 1 integrates to a logarithm
                    Poly fp = poly_deriv(f);
                    Poly q, r;
                    poly_divmod(p, fp, q, r);
                    if (j == 1 && r.empty() && poly_degree(q) == 0) terms.push_back(en.times({num(q[0]), app("Log", {Engine::poly_to_expr(f, x)})}));
                    else throw CasUnsupported("rational integral needs logarithms of algebraic numbers (irreducible factor of degree " + std::to_string(deg) + ")");
                }
            }
        }
        return en.plus(terms);
    }

    // ---- polynomial helpers ----
    bool as_polynomial(const Expr& e, const std::string& x, Poly& p) {
        std::vector<Expr> c;
        if (!en.poly_coefficients(e, x, c)) return false;
        p.clear();
        for (const auto& v : c) {
            if (!v->is_number()) return false;
            p.push_back(v->q);
        }
        poly_trim(p);
        return true;
    }

    // ---- table of elementary functions of a linear argument; returns F with dF/du = g, or nullptr ----
    Expr table(const Expr& g, const std::string& x) {
        Expr a, b;
        auto func_case = [&](const std::string& h, const Expr& u, Expr& F) -> bool {
            if (h == "Sin") { F = en.neg(app("Cos", {u})); return true; }
            if (h == "Cos") { F = app("Sin", {u}); return true; }
            if (h == "Tan") { F = en.neg(app("Log", {app("Cos", {u})})); return true; }
            if (h == "Cot") { F = app("Log", {app("Sin", {u})}); return true; }
            if (h == "Sinh") { F = app("Cosh", {u}); return true; }
            if (h == "Cosh") { F = app("Sinh", {u}); return true; }
            if (h == "Tanh") { F = app("Log", {app("Cosh", {u})}); return true; }
            if (h == "Log") { F = en.plus({en.times({u, app("Log", {u})}), en.neg(u)}); return true; }
            if (h == "ArcTan") { F = en.plus({en.times({u, app("ArcTan", {u})}), en.neg(en.times({half(), app("Log", {en.plus({one(), en.power(u, integer(2))})})}))}); return true; }
            if (h == "ArcSin") { F = en.plus({en.times({u, app("ArcSin", {u})}), en.power(en.plus({one(), en.neg(en.power(u, integer(2)))}), half())}); return true; }
            if (h == "ArcCos") { F = en.plus({en.times({u, app("ArcCos", {u})}), en.neg(en.power(en.plus({one(), en.neg(en.power(u, integer(2)))}), half()))}); return true; }
            if (h == "ArcSinh") { F = en.plus({en.times({u, app("ArcSinh", {u})}), en.neg(en.power(en.plus({en.power(u, integer(2)), one()}), half()))}); return true; }
            return false;
        };
        if (g->is_apply() && g->args.size() == 1 && linear(g->args[0], x, a, b)) {
            Expr F;
            if (func_case(g->name, g->args[0], F)) return en.div(F, a);
        }
        if (g->has_head("Power", 2)) {
            const Expr& base = g->args[0];
            const Expr& ex = g->args[1];
            // E^u, c^u
            if (!depends_on(base, x) && linear(ex, x, a, b)) {
                if (base->is_symbol_named("E")) return en.div(g, a);
                return en.div(g, en.times({a, en.eval(app("Log", {base}))}));
            }
            // u^p
            if (!depends_on(ex, x) && linear(base, x, a, b)) {
                if (ex->is_number() && ex->q == Rational(-1)) return en.div(app("Log", {base}), a);
                if (ex->is_number()) return en.div(en.power(base, en.plus({ex, one()})), en.times({a, en.plus({ex, one()})}));
            }
            // Log(u)^n: by parts, u Log(u)^n - n * integral of Log(u)^(n-1)
            if (base->has_head("Log", 1) && ex->is_integer() && !ex->q.is_negative() && linear(base->args[0], x, a, b)) {
                int64_t nn = 0;
                if (ex->q.num.to_int64(nn) && nn >= 2 && nn <= 30) {
                    Expr inner = integ(en.power(base, integer(nn - 1)), x, 1);
                    Expr u = base->args[0];
                    return en.div(en.plus({en.times({u, g}), en.neg(en.times({integer(nn), en.times({a, inner})}))}), a);
                }
            }
            // Cos(u)^-2, Sin(u)^-2, Cosh(u)^-2
            if (ex->is_number() && ex->q == Rational(-2) && base->is_apply() && base->args.size() == 1 && linear(base->args[0], x, a, b)) {
                const Expr& u = base->args[0];
                if (base->name == "Cos") return en.div(app("Tan", {u}), a);
                if (base->name == "Sin") return en.neg(en.div(app("Cot", {u}), a));
                if (base->name == "Cosh") return en.div(app("Tanh", {u}), a);
            }
            // (c - d u^2)^(-1/2) forms: 1/sqrt(c + d x^2)
            if (ex->is_number() && ex->q == Rational(BigInt(-1), BigInt(2))) {
                std::vector<Expr> co;
                if (en.poly_coefficients(base, x, co) && co.size() == 3 && co[0]->is_number() && co[1]->is_number() && co[1]->q.is_zero() && co[2]->is_number()) {
                    Rational c0 = co[0]->q, d = co[2]->q;
                    if (Rational(0) < c0 && d.is_negative()) {   // 1/sqrt(c - k x^2) = asin(sqrt(k/c) x)/sqrt(k)
                        Expr k = num(-d);
                        return en.div(app("ArcSin", {en.times({en.power(en.div(k, num(c0)), half()), symbol(x)})}), en.power(k, half()));
                    }
                    if (Rational(0) < c0 && Rational(0) < d) {   // 1/sqrt(c + k x^2) = asinh(sqrt(k/c) x)/sqrt(k)
                        Expr k = num(d);
                        return en.div(app("ArcSinh", {en.times({en.power(en.div(k, num(c0)), half()), symbol(x)})}), en.power(k, half()));
                    }
                }
            }
        }
        return nullptr;
    }

    // ---- polynomial times exp/sin/cos/sinh/cosh/log/atan ----
    Expr poly_times_transcendental(const Expr& g, const std::string& x) {
        if (!g->has_head("Times")) return nullptr;
        std::vector<Expr> polyf;
        Expr T;
        for (const auto& f : g->args) {
            Poly tmp;
            if (!depends_on(f, x) || as_polynomial(f, x, tmp)) { polyf.push_back(f); continue; }
            if (T) return nullptr;
            T = f;
        }
        if (!T) return nullptr;
        Expr p_expr = en.expand(en.times(polyf));
        std::vector<Expr> pc;
        if (!en.poly_coefficients(p_expr, x, pc)) return nullptr;
        auto deriv = [&](const std::vector<Expr>& c) {
            std::vector<Expr> r;
            for (size_t i = 1; i < c.size(); ++i) r.push_back(en.times({integer(static_cast<int64_t>(i)), c[i]}));
            return r;
        };
        auto to_e = [&](const std::vector<Expr>& c) {
            std::vector<Expr> t;
            for (size_t i = 0; i < c.size(); ++i) t.push_back(en.times({c[i], en.power(symbol(x), integer(static_cast<int64_t>(i)))}));
            return en.plus(t);
        };
        Expr a, b, u;
        // E^u
        bool is_exp = T->has_head("Power", 2) && T->args[0]->is_symbol_named("E") && linear(T->args[1], x, a, b);
        bool is_trig = T->is_apply() && T->args.size() == 1 && (T->name == "Sin" || T->name == "Cos" || T->name == "Sinh" || T->name == "Cosh") && linear(T->args[0], x, a, b);
        if (is_exp || is_trig) {
            u = is_exp ? T->args[1] : T->args[0];
            // recursive tabular integration on derivatives of p
            std::vector<std::vector<Expr>> derivs{pc};
            while (!derivs.back().empty() && derivs.size() < 200) derivs.push_back(deriv(derivs.back()));
            const size_t n = derivs.size();
            // I[k](kind) = integral of p^(k) * kind(u)
            // exp:  I_e[k] = p^(k) E^u / a - I_e[k+1] / a
            // sin:  I_s[k] = -p^(k) cos u / a + I_c[k+1] / a ;  cos: I_c[k] = p^(k) sin u / a - I_s[k+1] / a
            // sinh: I_sh[k] = p^(k) cosh u / a - I_ch[k+1] / a ; cosh: I_ch[k] = p^(k) sinh u / a - I_sh[k+1] / a
            Expr Eu = T->has_head("Power", 2) ? T : nullptr;
            Expr Sin = app("Sin", {u}), Cos = app("Cos", {u}), Sinh = app("Sinh", {u}), Cosh = app("Cosh", {u});
            auto P = [&](size_t k) { return to_e(derivs[k]); };
            // evaluate from the end
            std::vector<Expr> s(n + 1, zero()), c(n + 1, zero());   // for trig pairs; for exp use s only
            for (size_t k = n; k-- > 0;) {
                Expr pk = P(k);
                if (is_exp) {
                    s[k] = en.div(en.plus({en.times({pk, Eu}), en.neg(s[k + 1])}), a);
                } else if (T->name == "Sin" || T->name == "Cos") {
                    Expr is_new = en.div(en.plus({en.neg(en.times({pk, Cos})), c[k + 1]}), a);
                    Expr ic_new = en.div(en.plus({en.times({pk, Sin}), en.neg(s[k + 1])}), a);
                    s[k] = is_new; c[k] = ic_new;
                } else {
                    Expr ish = en.div(en.plus({en.times({pk, Cosh}), en.neg(c[k + 1])}), a);
                    Expr ich = en.div(en.plus({en.times({pk, Sinh}), en.neg(s[k + 1])}), a);
                    s[k] = ish; c[k] = ich;
                }
            }
            if (is_exp || T->name == "Sin" || T->name == "Sinh") return s[0];
            return c[0];
        }
        // p * Log(u): by parts
        if (T->has_head("Log", 1) && linear(T->args[0], x, a, b)) {
            Expr P_int = integrate_poly(pc, x);
            Expr rest = en.times({P_int, a, en.power(T->args[0], minus_one())});
            return en.plus({en.times({P_int, T}), en.neg(integ(rest, x, 1))});
        }
        if (T->has_head("ArcTan", 1) && linear(T->args[0], x, a, b)) {
            Expr P_int = integrate_poly(pc, x);
            Expr rest = en.times({P_int, a, en.power(en.plus({one(), en.power(T->args[0], integer(2))}), minus_one())});
            return en.plus({en.times({P_int, T}), en.neg(integ(rest, x, 1))});
        }
        return nullptr;
    }

    Expr integrate_poly(const std::vector<Expr>& c, const std::string& x) {
        std::vector<Expr> t;
        for (size_t i = 0; i < c.size(); ++i)
            t.push_back(en.times({c[i], en.power(symbol(x), integer(static_cast<int64_t>(i + 1))), num(Rational(1) / Rational(static_cast<int64_t>(i + 1)))}));
        return en.plus(t);
    }

    // ---- sin^m cos^n of one linear argument, exact Fourier expansion ----
    Expr trig_powers(const Expr& g, const std::string& x) {
        std::vector<Expr> factors = g->has_head("Times") ? g->args : std::vector<Expr>{g};
        int64_t m = 0, n = 0;
        Expr u;
        Expr coeff = one();
        for (const auto& f : factors) {
            Expr base = f, ex = one();
            if (f->has_head("Power", 2)) { base = f->args[0]; ex = f->args[1]; }
            if (!depends_on(f, x)) { coeff = en.times({coeff, f}); continue; }
            if (base->is_apply() && base->args.size() == 1 && (base->name == "Sin" || base->name == "Cos") && ex->is_integer() && !ex->q.is_negative()) {
                int64_t k = 0;
                if (!ex->q.num.to_int64(k) || k > 60) return nullptr;
                if (u && !equal(u, base->args[0])) return nullptr;
                u = base->args[0];
                (base->name == "Sin" ? m : n) += k;
                continue;
            }
            return nullptr;
        }
        if (!u || m + n == 0 || m + n > 60) return nullptr;
        Expr a, b;
        if (!linear(u, x, a, b)) return nullptr;
        // (z - 1/z)^m (z + 1/z)^n / ((2i)^m 2^n), as Laurent polynomial in z with Gaussian rational coefficients
        const int64_t total = m + n;
        std::vector<G> poly(static_cast<size_t>(2 * total + 1));   // exponent e stored at index e + total
        poly[static_cast<size_t>(total)] = {Rational(1), Rational(0)};
        auto mul_factor = [&](bool minus) {
            std::vector<G> next(poly.size());
            for (size_t i = 0; i < poly.size(); ++i) {
                if (poly[i].re.is_zero() && poly[i].im.is_zero()) continue;
                if (i + 1 < next.size()) next[i + 1] = next[i + 1] + poly[i];                 // * z
                if (i >= 1) next[i - 1] = next[i - 1] + G{minus ? -poly[i].re : poly[i].re, minus ? -poly[i].im : poly[i].im};  // * -+1/z
            }
            poly = std::move(next);
        };
        for (int64_t i = 0; i < m; ++i) mul_factor(true);
        for (int64_t i = 0; i < n; ++i) mul_factor(false);
        // divide by (2i)^m 2^n : (2i)^m = 2^m i^m
        Rational denom = Rational(2).pow(total);
        G ipow{Rational(1), Rational(0)};
        for (int64_t i = 0; i < m; ++i) ipow = ipow * G{Rational(0), Rational(1)};
        // 1 / i^m = conj(i^m) since |i^m| = 1
        G inv_i{ipow.re, -ipow.im};
        std::vector<Expr> terms;
        for (int64_t k = 0; k <= total; ++k) {
            G ck = poly[static_cast<size_t>(total + k)] * inv_i;
            ck.re = ck.re / denom;
            ck.im = ck.im / denom;
            if (k == 0) {
                if (!ck.re.is_zero()) terms.push_back(en.times({num(ck.re), u}));
                continue;
            }
            // c_k z^k + conj(c_k) z^-k = 2 Re(c_k) cos(k u) - 2 Im(c_k) sin(k u)
            Expr ku = k == 1 ? u : en.times({integer(k), u});
            Rational cosc = Rational(2) * ck.re, sinc = -(Rational(2) * ck.im);
            if (!cosc.is_zero()) terms.push_back(en.times({num(cosc / Rational(k)), app("Sin", {ku})}));
            if (!sinc.is_zero()) terms.push_back(en.times({num(-sinc / Rational(k)), app("Cos", {ku})}));
        }
        return en.div(en.times({coeff, en.plus(terms)}), a);
    }

    // ---- E^(p x) times sin / cos (q x) ----
    Expr exp_trig(const Expr& g, const std::string& x) {
        if (!g->has_head("Times")) return nullptr;
        Expr Eu, T, coeff = one();
        for (const auto& f : g->args) {
            if (!depends_on(f, x)) { coeff = en.times({coeff, f}); continue; }
            if (f->has_head("Power", 2) && f->args[0]->is_symbol_named("E") && !Eu) { Eu = f; continue; }
            if (f->is_apply() && f->args.size() == 1 && (f->name == "Sin" || f->name == "Cos") && !T) { T = f; continue; }
            return nullptr;
        }
        if (!Eu || !T) return nullptr;
        Expr p, p0, q, q0;
        if (!linear(Eu->args[1], x, p, p0) || !linear(T->args[0], x, q, q0)) return nullptr;
        Expr den = en.plus({en.power(p, integer(2)), en.power(q, integer(2))});
        Expr u2 = T->args[0];
        Expr num_part = T->name == "Sin"
            ? en.plus({en.times({p, app("Sin", {u2})}), en.neg(en.times({q, app("Cos", {u2})}))})
            : en.plus({en.times({p, app("Cos", {u2})}), en.times({q, app("Sin", {u2})})});
        return en.times({coeff, Eu, num_part, en.power(den, minus_one())});
    }

    // ---- u-substitution ----
    Expr substitution(const Expr& g, const std::string& x, int depth) {
        std::vector<Expr> cands;
        collect_candidates(g, x, cands);
        std::stable_sort(cands.begin(), cands.end(), [](const Expr& a, const Expr& b) { return leaf_count(a) > leaf_count(b); });
        if (cands.size() > 12) cands.resize(12);
        const std::string w = "$w" + std::to_string(depth);
        for (const auto& u : cands) {
            try {
                Expr du = en.diff(u, x);
                if (du->is_number() && du->q.is_zero()) continue;
                Expr q = en.simplify(en.div(g, du));
                Expr qw = en.eval(replace_expr(q, u, symbol(w)));
                if (depends_on(qw, x)) continue;
                Expr r = integ(qw, w, depth + 1);
                return en.eval(replace_expr(r, symbol(w), u));
            } catch (const CasUnsupported&) {
                continue;
            } catch (const CasMathError&) {
                continue;
            }
        }
        return nullptr;
    }

    // ---- driver ----
    Expr integ(const Expr& f0, const std::string& x, int depth) {
        if (depth > 6) throw CasUnsupported("integration is too deeply nested");
        en.tick();
        Expr f = en.eval(f0);
        if (!depends_on(f, x)) return en.times({f, symbol(x)});
        if (f->has_head("Plus")) {
            std::vector<Expr> ts;
            for (const auto& t : f->args) ts.push_back(integ(t, x, depth));
            return en.plus(ts);
        }
        if (f->has_head("Times")) {
            std::vector<Expr> constant, rest;
            for (const auto& t : f->args) (depends_on(t, x) ? rest : constant).push_back(t);
            if (!constant.empty()) return en.times({en.times(constant), integ(en.times(rest), x, depth)});
        }
        // rational function in x with rational coefficients
        {
            std::vector<std::string> syms;
            free_symbols(f, syms);
            if (syms.size() == 1 && syms[0] == x) {
                Poly N, D;
                bool ok = false;
                try { ok = en.rational_function(f, x, N, D); } catch (const CasLimitError&) { throw; } catch (const CasError&) { ok = false; }
                if (ok) return rational(N, D, x);
            }
        }
        if (Expr r = table(f, x)) return r;
        if (Expr r = poly_times_transcendental(f, x)) return r;
        if (Expr r = trig_powers(f, x)) return r;
        if (Expr r = exp_trig(f, x)) return r;
        // expanding may expose terms the rules above handle
        Expr ex = en.expand(f);
        if (!equal(ex, f)) return integ(ex, x, depth + 1);
        if (Expr r = substitution(f, x, depth)) return r;
        throw CasUnsupported("no integration rule applies to " + to_string(f));
    }
};

Expr Engine::integrate(const Expr& f0, const std::string& var) {
    Expr f = eval(f0);
    IntegrateOps ops(*this);
    Expr F = ops.integ(f, var, 0);
    F = simplify(F);
    // Verification: D[F] - f must vanish identically (proof) or, when the simplifier cannot show it, numerically at
    // several sample points (evidence). Failing to evaluate anywhere is not evidence.
    Expr residual = simplify(sub(diff(F, var), f));
    if (residual->is_number() && residual->q.is_zero()) return F;
    std::string witness;
    int evaluated = 0;
    if (find_nonzero_witness(residual, witness, &evaluated))
        throw CasUnsupported("internal check failed: derivative of the result differs from the integrand (" + witness + ")");
    if (evaluated < 3) throw CasUnsupported("could not verify the antiderivative numerically");
    return F;
}

// ---------------------------------------------------------------------------------------------------------------
// definite integrals
// ---------------------------------------------------------------------------------------------------------------
namespace {

// Number of sign changes of the Sturm sequence at x (finite) or at +-infinity.
int sturm_variations(const std::vector<Poly>& seq, const Rational* x, int infinity_sign) {
    int var = 0, last = 0;
    for (const auto& p : seq) {
        int sg = 0;
        if (p.empty()) continue;
        if (x) sg = poly_eval(p, *x).sign();
        else {
            sg = p.back().sign();
            if (infinity_sign < 0 && (p.size() - 1) % 2 == 1) sg = -sg;
        }
        if (sg == 0) continue;
        if (last != 0 && sg != last) ++var;
        last = sg;
    }
    return var;
}

// Distinct real roots of the polynomial D in (lo, hi), with +-infinity given by flags. Exact.
int real_roots_in(const Poly& D0, const Rational* lo, const Rational* hi, bool lo_inf, bool hi_inf) {
    Poly D = D0;
    Poly g = poly_gcd(D, poly_deriv(D));
    if (poly_degree(g) > 0) { Poly q, r; poly_divmod(D, g, q, r); D = q; }
    if (poly_degree(D) < 1) return 0;
    std::vector<Poly> seq{D, poly_deriv(D)};
    while (!seq.back().empty()) {
        Poly q, r;
        poly_divmod(seq[seq.size() - 2], seq.back(), q, r);
        if (r.empty()) break;
        seq.push_back(poly_scale(r, Rational(-1)));
    }
    int at_lo = lo_inf ? sturm_variations(seq, nullptr, -1) : sturm_variations(seq, lo, 0);
    int at_hi = hi_inf ? sturm_variations(seq, nullptr, +1) : sturm_variations(seq, hi, 0);
    int count = at_lo - at_hi;   // roots in (lo, hi]
    if (!hi_inf && poly_eval(D, *hi).is_zero()) --count;
    return count;
}

} // namespace

Expr Engine::integrate_definite(const Expr& f0, const std::string& var, const Expr& a_in, const Expr& b_in) {
    Expr f = eval(f0), a = eval(a_in), b = eval(b_in);
    auto is_pinf = [](const Expr& e) { return e->is_symbol_named("Infinity"); };
    auto is_ninf = [](const Expr& e) { return e->has_head("Times", 2) && e->args[0]->is_number() && e->args[0]->q == Rational(-1) && e->args[1]->is_symbol_named("Infinity"); };
    const bool a_inf = is_pinf(a) || is_ninf(a), b_inf = is_pinf(b) || is_ninf(b);
    double av = 0, bv = 0;
    if (!a_inf && !numeric_value(a, av)) throw CasUnsupported("integration limits must be numbers");
    if (!b_inf && !numeric_value(b, bv)) throw CasUnsupported("integration limits must be numbers");
    if (is_pinf(a)) av = INFINITY;
    if (is_ninf(a)) av = -INFINITY;
    if (is_pinf(b)) bv = INFINITY;
    if (is_ninf(b)) bv = -INFINITY;
    if (av == bv) return zero();
    if (av > bv) return neg(integrate_definite(f, var, b, a));

    // singularities strictly inside (a, b)
    std::vector<std::string> syms;
    free_symbols(f, syms);
    Poly rn, rd;
    bool rational_integrand = syms.size() == 1 && syms[0] == var && rational_function(f, var, rn, rd);
    if (rational_integrand) {
        Rational lo, hi;
        auto exact_of = [](const Expr& e, double d) {
            if (e->is_number()) return e->q;
            int ex = 0;
            double m = std::frexp(d, &ex);                    // d = m * 2^ex, 0.5 <= |m| < 1
            BigInt mant(static_cast<int64_t>(std::ldexp(m, 53)));
            int shift = ex - 53;
            Rational r = Rational::from_bigint(mant);
            return shift >= 0 ? r * Rational::from_bigint(BigInt::pow(BigInt(2), static_cast<uint64_t>(shift)))
                              : r / Rational::from_bigint(BigInt::pow(BigInt(2), static_cast<uint64_t>(-shift)));
        };
        if (!a_inf) lo = exact_of(a, av);
        if (!b_inf) hi = exact_of(b, bv);
        if (real_roots_in(rd, &lo, &hi, a_inf, b_inf) > 0) throw CasMathError("the integrand has a pole inside the interval: the integral does not converge");
    } else {
        // sample for non-finite values inside the interval
        const int n = 1500;
        double lo = a_inf ? -50.0 : av, hi = b_inf ? 50.0 : bv;
        for (int i = 1; i < n; ++i) {
            double xv = lo + (hi - lo) * static_cast<double>(i) / n;
            double v = 0.0;
            try {
                if (!numeric_value(substitute(f, {{var, real(xv)}}), v) || !std::isfinite(v))
                    throw CasUnsupported("the integrand is not finite or real inside the interval");
            } catch (const CasMathError&) {
                throw CasUnsupported("the integrand is singular inside the interval");
            }
        }
    }

    Expr F = integrate(f, var);
    auto endpoint = [&](const Expr& e, bool infinite, int side) -> Expr {
        if (infinite) return limit(F, var, e, 0);
        try {
            Expr v = eval(substitute(F, {{var, e}}));
            double d = 0;
            if (numeric_value(v, d) && std::isfinite(d)) return v;
        } catch (const CasMathError&) {
        } catch (const CasUnsupported&) {
        }
        return limit(F, var, e, side);
    };
    Expr Fb = endpoint(b, b_inf, -1), Fa = endpoint(a, a_inf, +1);
    for (const Expr& e : {Fa, Fb}) {
        if (depends_on(e, "Infinity") || e->is_symbol_named("Infinity") || e->is_symbol_named("Indeterminate") || is_ninf(e))
            throw CasMathError("the integral does not converge");
    }
    Expr value = simplify(sub(Fb, Fa));

    // Numerical cross-check on the interior of a finite interval: integrate f numerically between a' and b' (a small
    // distance inside the endpoints, so endpoint singularities do not matter) and compare with F(b') - F(a'). This
    // catches an antiderivative that jumps across a hidden singularity.
    if (!a_inf && !b_inf) {
        static const double gx[] = {-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640};
        static const double gw[] = {0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891};
        const double a2 = av + 1e-3 * (bv - av), b2 = bv - 1e-3 * (bv - av);
        auto Fnum = [&](double xv, double& out) {
            try { return numeric_value(substitute(F, {{var, real(xv)}}), out) && std::isfinite(out); }
            catch (const CasError&) { return false; }
        };
        double fa = 0.0, fb = 0.0;
        if (Fnum(a2, fa) && Fnum(b2, fb)) {
            const int panels = 400;
            double sum = 0.0;
            bool ok = true;
            for (int p = 0; p < panels && ok; ++p) {
                double l = a2 + (b2 - a2) * p / panels, h = a2 + (b2 - a2) * (p + 1) / panels;
                for (int k = 0; k < 5; ++k) {
                    double xv = 0.5 * (l + h) + 0.5 * (h - l) * gx[k];
                    double v = 0.0;
                    try { ok = numeric_value(substitute(f, {{var, real(xv)}}), v) && std::isfinite(v); }
                    catch (const CasError&) { ok = false; }
                    if (!ok) break;
                    sum += 0.5 * (h - l) * gw[k] * v;
                }
            }
            if (ok && std::fabs((fb - fa) - sum) > 1e-6 * (1.0 + std::fabs(sum)))
                throw CasUnsupported("numerical check disagrees with the antiderivative (integrand may be singular)");
        }
    }
    return value;
}

} // namespace strata::math::cas
