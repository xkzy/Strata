// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/cas/series.cpp - truncated Laurent series (the Series builtin of Mathics, implemented natively)
#include "strata/math/cas/engine.hpp"

#include <algorithm>
#include <cmath>

namespace strata::math::cas {

namespace {

const char* kT = "$t";   // shifted variable: x = x0 + t. '$' cannot appear in parsed input.
const char* kS = "$s";   // dummy variable for function derivatives

// Terms with exponent < ord are exact. c[i] multiplies t^(nmin + i); c.size() == ord - nmin (>= 0).
struct Ser {
    int64_t nmin = 0;
    int64_t ord = 0;
    std::vector<Expr> c;
};

} // namespace

struct SeriesOps {
    Engine& en;
    explicit SeriesOps(Engine& e) : en(e) {}

    // Exact decision whether e is zero: true / false, or throws CasUnsupported when it cannot tell.
    bool is_zero(const Expr& e0) {
        Expr e = en.eval(e0);
        if (e->is_number()) return e->q.is_zero();
        if (e->kind == Kind::Real) return e->d == 0.0;
        double v = 0.0;
        if (en.numeric_value(e, v)) {
            if (std::fabs(v) > 1e-9) return false;
            Expr s = en.simplify(e);
            if (s->is_number() && s->q.is_zero()) return true;
            throw CasUnsupported("cannot decide whether a series coefficient is zero");
        }
        Expr s = en.simplify(e);
        if (s->is_number()) return s->q.is_zero();
        std::string witness;
        if (en.find_nonzero_witness(s, witness)) return false;
        throw CasUnsupported("cannot decide whether a series coefficient is zero");
    }

    Ser constant(const Expr& e, int64_t N) {
        Ser s;
        s.nmin = 0;
        s.ord = N;
        s.c.assign(static_cast<size_t>(std::max<int64_t>(N, 0)), zero());
        if (N > 0) s.c[0] = e;
        normalize(s);
        // a non-zero constant keeps its slot; a zero constant is "all zeros up to ord"
        return s;
    }

    void normalize(Ser& s) {
        size_t lead = 0;
        while (lead < s.c.size() && is_zero(s.c[lead])) ++lead;
        if (lead == s.c.size()) { s.nmin = s.ord; s.c.clear(); return; }
        if (lead) { s.c.erase(s.c.begin(), s.c.begin() + static_cast<long>(lead)); s.nmin += static_cast<int64_t>(lead); }
    }

    Ser add(const Ser& a, const Ser& b) {
        Ser r;
        r.ord = std::min(a.ord, b.ord);
        r.nmin = std::min(a.nmin, b.nmin);
        if (r.nmin > r.ord) r.nmin = r.ord;
        r.c.assign(static_cast<size_t>(r.ord - r.nmin), zero());
        auto acc = [&](const Ser& s) {
            for (size_t i = 0; i < s.c.size(); ++i) {
                int64_t e = s.nmin + static_cast<int64_t>(i);
                if (e >= r.ord) break;
                size_t idx = static_cast<size_t>(e - r.nmin);
                r.c[idx] = en.plus({r.c[idx], s.c[i]});
            }
        };
        acc(a); acc(b);
        normalize(r);
        return r;
    }

    Ser scale(const Ser& a, const Expr& k) {
        Ser r = a;
        for (auto& x : r.c) x = en.times({k, x});
        normalize(r);
        return r;
    }

    Ser mul(const Ser& a, const Ser& b) {
        Ser r;
        // relative precisions: a.ord - a.nmin etc; product valid below min(a.ord + b.nmin, b.ord + a.nmin)
        int64_t anmin = a.c.empty() ? a.ord : a.nmin, bnmin = b.c.empty() ? b.ord : b.nmin;
        r.ord = std::min(a.ord + bnmin, b.ord + anmin);
        r.nmin = anmin + bnmin;
        if (r.nmin > r.ord) { r.nmin = r.ord; r.c.clear(); return r; }
        r.c.assign(static_cast<size_t>(r.ord - r.nmin), zero());
        for (size_t i = 0; i < a.c.size(); ++i) {
            for (size_t j = 0; j < b.c.size(); ++j) {
                int64_t e = a.nmin + static_cast<int64_t>(i) + b.nmin + static_cast<int64_t>(j);
                if (e >= r.ord) break;
                size_t idx = static_cast<size_t>(e - r.nmin);
                r.c[idx] = en.plus({r.c[idx], en.times({a.c[i], b.c[j]})});
            }
            en.tick(b.c.size());
        }
        normalize(r);
        return r;
    }

    Ser reciprocal(const Ser& a) {
        if (a.c.empty()) throw CasUnsupported("series is zero to the available order: cannot invert");
        const int64_t v = a.nmin;
        const size_t m = a.c.size();           // relative precision
        Expr c0 = a.c[0];
        std::vector<Expr> s(m), b(m);
        for (size_t i = 0; i < m; ++i) s[i] = en.div(a.c[i], c0);
        b[0] = one();
        for (size_t k = 1; k < m; ++k) {
            std::vector<Expr> terms;
            for (size_t j = 1; j <= k; ++j) terms.push_back(en.times({s[j], b[k - j]}));
            b[k] = en.neg(en.plus(terms));
        }
        Ser r;
        r.nmin = -v;
        r.ord = -v + static_cast<int64_t>(m);
        r.c.resize(m);
        Expr c0inv = en.power(c0, minus_one());
        for (size_t k = 0; k < m; ++k) r.c[k] = en.times({b[k], c0inv});
        normalize(r);
        return r;
    }

    Ser pow_int(const Ser& a, int64_t n, int64_t N) {
        if (n == 0) return constant(one(), N);
        Ser base = n < 0 ? reciprocal(a) : a;
        uint64_t k = static_cast<uint64_t>(n < 0 ? -n : n);
        Ser result = constant(one(), N + 64);
        result.ord = std::max<int64_t>(result.ord, 0);
        bool first = true;
        Ser acc;
        while (k) {
            if (k & 1) { acc = first ? base : mul(acc, base); first = false; }
            k >>= 1;
            if (k) base = mul(base, base);
        }
        return acc;
    }

    // series of f(g) around g0 = constant term of g, via derivatives of f.
    Ser apply_function(const std::string& head, const Ser& g, int64_t N) {
        // Only functions analytic at the expansion point may be expanded through their derivatives. Abs, Sign,
        // Floor and friends have derivatives that vanish at the kink (or are undefined), which would silently
        // produce a wrong zero series.
        static const char* analytic[] = {"Sin", "Cos", "Tan", "Cot", "Sec", "Csc", "ArcSin", "ArcCos", "ArcTan", "Sinh", "Cosh",
                                         "Tanh", "ArcSinh", "ArcCosh", "ArcTanh", "Exp", "Log"};
        bool ok = false;
        for (const char* a : analytic) if (head == a) ok = true;
        if (!ok) throw CasUnsupported("series of " + head + " is not supported");
        Expr g0 = zero();
        Ser h = g;
        if (!g.c.empty() && g.nmin < 0) throw CasUnsupported("function of a series with a pole");
        if (!g.c.empty() && g.nmin == 0) {
            g0 = g.c[0];
            h.c[0] = zero();
            normalize(h);
        }
        const int64_t hv = h.c.empty() ? h.ord : h.nmin;     // valuation of h (>= 1) or "infinite"
        Ser result = constant(zero(), g.ord);
        Expr deriv = app(head, {symbol(kS)});
        Ser hpow = constant(one(), g.ord);
        Expr fact = one();
        for (int64_t k = 0;; ++k) {
            if (k > 0) { hpow = mul(hpow, h); fact = en.times({fact, integer(k)}); }
            if (!h.c.empty() && k * hv >= std::min(g.ord, N + 64) && k > 0) break;
            if (h.c.empty() && k > 0) break;
            if (k > 400) throw CasLimitError("series needs too many terms");
            Expr fk;
            try {
                Expr at = Engine::substitute(deriv, {{kS, g0}});
                fk = en.eval(at);
            } catch (const CasMathError&) {
                throw CasUnsupported("function is singular at the expansion point (non-power-series behaviour)");
            }
            if (k > 0 || true) {
                Ser term = scale(hpow, en.div(fk, fact));
                result = add(result, term);
            }
            deriv = en.diff(deriv, kS);
        }
        return result;
    }

    Ser from_expr(const Expr& e, int64_t N, const Expr& x0) {
        en.tick();
        switch (e->kind) {
            case Kind::Number: case Kind::Real: return constant(e, N);
            case Kind::Symbol: {
                if (e->name == kT) {
                    Ser s;
                    s.nmin = 0; s.ord = N;
                    s.c.assign(static_cast<size_t>(std::max<int64_t>(N, 0)), zero());
                    if (N > 0) s.c[0] = x0;
                    if (N > 1) s.c[1] = one();
                    normalize(s);
                    return s;
                }
                return constant(e, N);
            }
            case Kind::Apply: break;
        }
        if (!depends_on(e, kT)) return constant(e, N);
        const std::string& h = e->name;
        if (h == "Plus") {
            Ser acc = from_expr(e->args[0], N, x0);
            for (size_t i = 1; i < e->args.size(); ++i) acc = add(acc, from_expr(e->args[i], N, x0));
            return acc;
        }
        if (h == "Times") {
            Ser acc = from_expr(e->args[0], N, x0);
            for (size_t i = 1; i < e->args.size(); ++i) acc = mul(acc, from_expr(e->args[i], N, x0));
            return acc;
        }
        if (h == "Power" && e->args.size() == 2) {
            const Expr& b = e->args[0];
            const Expr& p = e->args[1];
            if (!depends_on(p, kT)) {
                Ser bs = from_expr(b, N, x0);
                if (p->is_integer()) {
                    int64_t k = 0;
                    if (!p->q.num.to_int64(k) || std::llabs(k) > 100000) throw CasLimitError("exponent too large");
                    return pow_int(bs, k, N);
                }
                if (p->is_number()) {
                    // b = c t^m (1 + u): b^p = c^p t^(m p) (1+u)^p, needs m*p integer
                    if (bs.c.empty()) throw CasUnsupported("power of a zero series");
                    Rational mp = Rational(bs.nmin) * p->q;
                    if (!mp.is_integer()) throw CasUnsupported("fractional power of a series with a branch point");
                    Expr c0 = bs.c[0];
                    Ser u;
                    u.nmin = 0; u.ord = bs.ord - bs.nmin; u.c.resize(bs.c.size());
                    for (size_t i = 0; i < bs.c.size(); ++i) u.c[i] = en.div(bs.c[i], c0);
                    u.c[0] = zero();
                    normalize(u);
                    // (1+u)^p = sum binom(p,k) u^k
                    Ser res = constant(zero(), u.ord);
                    Ser upow = constant(one(), u.ord);
                    Rational coef(1);
                    const int64_t uv = u.c.empty() ? u.ord : u.nmin;
                    for (int64_t k = 0;; ++k) {
                        if (k > 0) { upow = mul(upow, u); coef = coef * (p->q - Rational(k - 1)) / Rational(k); }
                        res = add(res, scale(upow, num(coef)));
                        if (u.c.empty() || (k + 1) * uv >= u.ord) break;
                        if (k > 400) throw CasLimitError("series needs too many terms");
                    }
                    int64_t shift = 0;
                    mp.num.to_int64(shift);
                    Ser out = res;
                    out.nmin += shift; out.ord += shift;
                    return scale(out, en.power(c0, p));
                }
                // symbolic constant exponent: b^p = exp(p log b)
            }
            // E^g: expand exp around the constant term of g
            if (b->is_symbol_named("E")) return apply_function("Exp", from_expr(p, N, x0), N);
            // general: b^p = E^(p Log b)
            return from_expr(app("Power", {symbol("E"), en.times({p, en.eval(app("Log", {b}))})}), N, x0);
        }
        if (e->args.size() == 1) {
            Ser g = from_expr(e->args[0], N, x0);
            return apply_function(h, g, N);
        }
        throw CasUnsupported("series of " + h + " is not supported");
    }
};

Expr Engine::series(const Expr& f0, const std::string& var, const Expr& x0, int64_t n) {
    if (n < 0 || n > 200) throw CasLimitError("series order out of range");
    Expr f = eval(f0);
    Expr shifted = eval(substitute(f, {{var, plus({x0, symbol(kT)})}}));
    SeriesOps ops(*this);
    // Precision is lost to cancellation (sin(x)/x needs sin to one more order, 1 - cos(x) over x^2 two more):
    // ask for extra terms and retry with more until n+1 terms are exact.
    for (int64_t extra : {4, 8, 16, 32, 64}) {
        const int64_t N = n + 1 + extra;
        Ser s = ops.from_expr(shifted, N, zero());
        // drop terms at or beyond the requested order
        const int64_t target = n + 1;
        if (s.ord < target) continue;
        std::vector<Expr> coeffs;
        int64_t nmin = s.c.empty() ? target : std::min(s.nmin, target);
        for (int64_t e = nmin; e < target; ++e) {
            int64_t idx = e - s.nmin;
            coeffs.push_back(idx >= 0 && idx < static_cast<int64_t>(s.c.size()) ? simplify(s.c[static_cast<size_t>(idx)]) : zero());
        }
        return app("SeriesData", {symbol(var), x0, app("List", coeffs), integer(nmin), integer(target), one()});
    }
    throw CasUnsupported("could not reach the requested series order");
}

void Engine::series_valuation(const Expr& g, int64_t& v, Expr& lead, bool& is_zero) {
    SeriesOps ops(*this);
    for (int64_t N : {6, 12, 24, 48}) {
        Ser s = ops.from_expr(g, N, zero());
        if (!s.c.empty()) {
            v = s.nmin;
            lead = simplify(s.c[0]);
            is_zero = false;
            return;
        }
        if (s.ord > 0) { is_zero = true; v = s.ord; lead = zero(); return; }
    }
    throw CasUnsupported("could not determine the leading behaviour");
}

} // namespace strata::math::cas
