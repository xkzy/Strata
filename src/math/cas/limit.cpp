// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/cas/limit.cpp - limits (the Limit builtin of Mathics delegates to SymPy; implemented natively)
//
// Strategy for the right-hand limit of g(t) as t -> 0+:
//   1. direct substitution when the expression is continuous there,
//   2. Laurent series (valuation and leading coefficient),
//   3. exp/log rewriting of variable powers and extended-real arithmetic on sums and products,
//   4. L'Hopital on quotients and 0*infinity products.
// A two-sided limit is the common value of the two one-sided limits, otherwise Indeterminate.
#include "strata/math/cas/engine.hpp"

#include <algorithm>
#include <cmath>

namespace strata::math::cas {

namespace {

const char* kT = "$t";

enum class LK { Finite, PosInf, NegInf };
struct Lim {
    LK kind = LK::Finite;
    Expr val;
    static Lim finite(Expr v) { Lim l; l.kind = LK::Finite; l.val = std::move(v); return l; }
    static Lim pos_inf() { Lim l; l.kind = LK::PosInf; return l; }
    static Lim neg_inf() { Lim l; l.kind = LK::NegInf; return l; }
};

} // namespace

struct LimitOps {
    Engine& en;
    explicit LimitOps(Engine& e) : en(e) {}

    static int sign_of(Engine& en, const Expr& v) {
        double d = 0.0;
        if (!en.numeric_value(v, d)) throw CasUnsupported("cannot determine the sign of a limit value");
        return d > 0 ? 1 : d < 0 ? -1 : 0;
    }

    bool continuous_at_zero(const Expr& g) {
        // Constructs that are not continuous (or whose value at the point is not the limit) rule out substitution.
        std::vector<const Node*> stack{g.get()};
        while (!stack.empty()) {
            const Node* n = stack.back();
            stack.pop_back();
            if (n->kind != Kind::Apply) continue;
            if (n->name == "Sign" || n->name == "Floor" || n->name == "Ceiling") return false;
            if (n->name == "Power" && n->args.size() == 2 && depends_on(n->args[1], kT)) return false;  // 0^0 hazards
            for (const auto& a : n->args) stack.push_back(a.get());
        }
        return true;
    }

    // Replaces Abs(a) / Sign(a) by +-a / +-1 using the sign of a as t -> 0+ (decided by its leading series term).
    Expr resolve_signs(const Expr& g) {
        if (!g->is_apply()) return g;
        std::vector<Expr> args;
        for (const auto& a : g->args) args.push_back(resolve_signs(a));
        if ((g->name == "Abs" || g->name == "Sign") && args.size() == 1 && depends_on(args[0], kT)) {
            int64_t v = 0; Expr lead; bool z = false;
            en.series_valuation(en.eval(args[0]), v, lead, z);
            if (z) throw CasUnsupported("sign of a vanishing quantity");
            int s = sign_of(en, lead);
            if (s == 0) throw CasUnsupported("sign of a quantity is undetermined");
            if (g->name == "Sign") return integer(s);
            return s > 0 ? args[0] : en.neg(args[0]);
        }
        return app(g->name, std::move(args));
    }

    static bool continuous_function(const std::string& h) {
        return !(h == "Sign" || h == "Floor" || h == "Ceiling");
    }

    Lim right(const Expr& g0, int depth, int lh = 0) {
        if (lh > 8) throw CasUnsupported("limit needs too many L'Hopital steps");
        if (depth > 60) throw CasUnsupported("limit expression is too deeply nested");
        en.tick();
        Expr g = en.eval(g0);
        if (!depends_on(g, kT)) return Lim::finite(g);
        g = en.eval(resolve_signs(g));
        if (!depends_on(g, kT)) return Lim::finite(g);

        // 1. substitution
        if (continuous_at_zero(g)) {
            try {
                Expr v = en.eval(Engine::substitute(g, {{kT, zero()}}));
                if (!depends_on(v, kT) && !v->is_symbol_named("Infinity")) return Lim::finite(v);
            } catch (const CasMathError&) {
            } catch (const CasUnsupported&) {
            }
        }

        // 2. series
        try {
            int64_t v = 0;
            Expr lead;
            bool is_zero = false;
            en.series_valuation(g, v, lead, is_zero);
            if (is_zero || v > 0) return Lim::finite(zero());
            if (v == 0) return Lim::finite(lead);
            int s = sign_of(en, lead);
            if (s == 0) throw CasUnsupported("zero leading coefficient");
            return s > 0 ? Lim::pos_inf() : Lim::neg_inf();
        } catch (const CasUnsupported&) {
        } catch (const CasMathError&) {
        }

        // 3. structural rules
        if (g->has_head("Power", 2) && depends_on(g->args[1], kT)) {
            const Expr& b = g->args[0];
            const Expr& e = g->args[1];
            // b^e = E^(e Log b), valid for b > 0 near the point
            int64_t bv = 0; Expr blead; bool bz = false;
            en.series_valuation(en.eval(b), bv, blead, bz);
            if (bz || sign_of(en, blead) <= 0) throw CasUnsupported("power with a non-positive base");
            Lim L = right(en.times({e, en.eval(app("Log", {b}))}), depth + 1, lh);
            if (L.kind == LK::Finite) return Lim::finite(en.eval(app("Power", {symbol("E"), L.val})));
            return L.kind == LK::PosInf ? Lim::pos_inf() : Lim::finite(zero());
        }
        if (g->has_head("Power", 2) && !depends_on(g->args[1], kT) && g->args[1]->is_number()) {
            // b^p with constant p and b -> infinity
            Lim lb = right(g->args[0], depth + 1, lh);
            if (lb.kind == LK::PosInf) return g->args[1]->q.is_negative() ? Lim::finite(zero()) : Lim::pos_inf();
            if (lb.kind == LK::NegInf && g->args[1]->is_integer()) {
                bool even = g->args[1]->q.num.is_even();
                if (g->args[1]->q.is_negative()) return Lim::finite(zero());
                return even ? Lim::pos_inf() : Lim::neg_inf();
            }
        }
        if (g->is_apply() && g->args.size() == 1 && g->name != "Power") {
            const std::string& h = g->name;
            if (!continuous_function(h)) throw CasUnsupported(h + " is discontinuous: limit not determined");
            Lim la = right(g->args[0], depth + 1, lh);
            if (la.kind == LK::Finite) {
                // Log at 0+ is -infinity; otherwise evaluate (domain errors become "unsupported")
                if (h == "Log" && la.val->is_number() && la.val->q.is_zero()) {
                    int64_t v = 0; Expr lead; bool z = false;
                    en.series_valuation(en.eval(g->args[0]), v, lead, z);
                    if (!z && v > 0 && sign_of(en, lead) > 0) return Lim::neg_inf();
                    throw CasUnsupported("logarithm of a quantity that does not approach 0 from above");
                }
                try { return Lim::finite(en.eval(app(h, {la.val}))); }
                catch (const CasMathError&) { throw CasUnsupported("function is singular at the limit value"); }
            }
            const bool pos = la.kind == LK::PosInf;
            const Expr half_pi = en.times({num(Rational(BigInt(1), BigInt(2))), symbol("Pi")});
            if (h == "ArcTan") return Lim::finite(pos ? half_pi : en.neg(half_pi));
            if (h == "Tanh") return Lim::finite(integer(pos ? 1 : -1));
            if (h == "Sinh" || h == "ArcSinh") return pos ? Lim::pos_inf() : Lim::neg_inf();
            if (h == "Cosh") return Lim::pos_inf();
            if (h == "ArcCosh" && pos) return Lim::pos_inf();
            if (h == "Log" && pos) return Lim::pos_inf();
            throw CasUnsupported("function of an unbounded quantity has no limit rule: " + h);
        }
        if (g->has_head("Plus")) {
            Lim acc = Lim::finite(zero());
            bool failed = false;
            std::vector<Lim> parts;
            for (const auto& a : g->args) {
                Lim x;
                try { x = right(a, depth + 1, lh); } catch (const CasUnsupported&) { failed = true; break; }
                parts.push_back(x);
                if (acc.kind == LK::Finite && x.kind == LK::Finite) acc = Lim::finite(en.plus({acc.val, x.val}));
                else if (acc.kind == LK::Finite) acc = x;
                else if (x.kind == LK::Finite) { /* finite + infinity */ }
                else if (acc.kind != x.kind) { failed = true; break; }   // inf - inf
            }
            if (!failed) return acc;
            // inf - inf: factor out a term a that dominates, a (1 + sum(b/a)); the sum of limits decides the sign.
            for (size_t i = 0; i < g->args.size(); ++i) {
                try {
                    Lim la = right(g->args[i], depth + 1, lh);
                    if (la.kind == LK::Finite) continue;
                    std::vector<Expr> rest;
                    for (size_t j = 0; j < g->args.size(); ++j) if (j != i) rest.push_back(en.div(g->args[j], g->args[i]));
                    Lim lr = right(en.plus(rest), depth + 1, lh);
                    if (lr.kind != LK::Finite) continue;
                    Expr onep = en.eval(en.plus({one(), lr.val}));
                    int sg = sign_of(en, onep);
                    if (sg == 0) continue;
                    bool pos = (la.kind == LK::PosInf) == (sg > 0);
                    return pos ? Lim::pos_inf() : Lim::neg_inf();
                } catch (const CasUnsupported&) {
                    continue;
                }
            }
        }
        if (g->has_head("Times") && g->args.size() >= 2) {
            std::vector<Lim> ls;
            for (const auto& f : g->args) ls.push_back(right(f, depth + 1, lh));
            auto is_zero_lim = [&](const Lim& l) { return l.kind == LK::Finite && l.val->is_number() && l.val->q.is_zero(); };
            bool has_zero = false, has_inf = false;
            for (const auto& l : ls) { has_zero = has_zero || is_zero_lim(l); has_inf = has_inf || l.kind != LK::Finite; }
            if (!(has_zero && has_inf)) {
                if (has_zero) return Lim::finite(zero());
                Expr c = one();
                int sgn = 1;
                bool inf = false;
                for (const auto& l : ls) {
                    if (l.kind == LK::Finite) c = en.times({c, l.val});
                    else { inf = true; sgn *= l.kind == LK::PosInf ? 1 : -1; }
                }
                if (!inf) return Lim::finite(en.eval(c));
                int sc = sign_of(en, c);
                if (sc == 0) throw CasUnsupported("indeterminate product");
                return sgn * sc > 0 ? Lim::pos_inf() : Lim::neg_inf();
            }
            // 0 * infinity -> quotient, then L'Hopital. Two equivalent rewrites; try the natural one first.
            std::vector<Expr> za, ib;
            for (size_t i = 0; i < g->args.size(); ++i) (is_zero_lim(ls[i]) ? za : ib).push_back(g->args[i]);
            Expr A = en.times(za), B = en.times(ib);
            bool b_is_denominator = true;
            for (const auto& f : ib) if (!(f->has_head("Power", 2) && f->args[1]->is_number() && f->args[1]->q.is_negative())) b_is_denominator = false;
            auto zero_over_zero = [&]() { return right(en.div(en.diff(A, kT), en.diff(en.power(B, minus_one()), kT)), depth + 1, lh + 1); };
            auto inf_over_inf = [&]() { return right(en.div(en.diff(B, kT), en.diff(en.power(A, minus_one()), kT)), depth + 1, lh + 1); };
            try {
                return b_is_denominator ? zero_over_zero() : inf_over_inf();
            } catch (const CasUnsupported&) {
            }
            return b_is_denominator ? inf_over_inf() : zero_over_zero();
        }
        throw CasUnsupported("limit could not be determined");
    }
};

Expr Engine::limit(const Expr& f0, const std::string& var, const Expr& x0_in, int dir) {
    Expr f = eval(f0);
    Expr x0 = eval(x0_in);
    LimitOps ops(*this);
    Expr t = symbol(kT);
    auto to_expr = [&](const Lim& l) -> Expr {
        if (l.kind == LK::Finite) return l.val;
        return l.kind == LK::PosInf ? symbol("Infinity") : neg(symbol("Infinity"));
    };
    auto one_sided = [&](int side) -> Lim {
        // side +1: x = x0 + t, side -1: x = x0 - t, with t -> 0+
        Expr sub = side > 0 ? plus({x0, t}) : plus({x0, neg(t)});
        return ops.right(substitute(f, {{var, sub}}), 0);
    };
    bool pos_inf = x0->is_symbol_named("Infinity");
    bool neg_inf = x0->has_head("Times", 2) && x0->args[0]->is_number() && x0->args[0]->q == Rational(-1) && x0->args[1]->is_symbol_named("Infinity");
    if (pos_inf || neg_inf) {
        Expr sub = pos_inf ? power(t, minus_one()) : neg(power(t, minus_one()));
        return to_expr(ops.right(substitute(f, {{var, sub}}), 0));
    }
    if (dir > 0) return to_expr(one_sided(+1));
    if (dir < 0) return to_expr(one_sided(-1));
    Lim a = one_sided(+1), b = one_sided(-1);
    if (a.kind != b.kind) return symbol("Indeterminate");
    if (a.kind == LK::Finite) {
        Expr d = simplify(sub(a.val, b.val));
        if (d->is_number() && d->q.is_zero()) return a.val;
        double dv = 0.0;
        if (numeric_value(d, dv) && std::fabs(dv) > 1e-12) return symbol("Indeterminate");
        throw CasUnsupported("could not compare the one-sided limits");
    }
    return to_expr(a);
}

} // namespace strata::math::cas
