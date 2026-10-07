// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/cas/printer.cpp - Expr -> infix text for the native CAS
#include "strata/math/cas/parser.hpp"

#include <algorithm>
#include <cstdio>
#include <map>

namespace strata::math::cas {

namespace {

const char* print_name(const std::string& head) {
    static const std::map<std::string, const char*> m = {
        {"Sin", "sin"}, {"Cos", "cos"}, {"Tan", "tan"}, {"Cot", "cot"}, {"Sec", "sec"}, {"Csc", "csc"},
        {"ArcSin", "asin"}, {"ArcCos", "acos"}, {"ArcTan", "atan"}, {"Sinh", "sinh"}, {"Cosh", "cosh"},
        {"Tanh", "tanh"}, {"ArcSinh", "asinh"}, {"ArcCosh", "acosh"}, {"ArcTanh", "atanh"}, {"Exp", "exp"},
        {"Log", "log"}, {"Sqrt", "sqrt"}, {"Abs", "abs"}, {"Sign", "sign"}, {"Floor", "floor"},
        {"Ceiling", "ceiling"}, {"Gamma", "gamma"},
    };
    auto it = m.find(head);
    return it == m.end() ? nullptr : it->second;
}

std::string real_to_string(double d) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.15g", d);
    std::string s = buf;
    if (s.find_first_of(".eEn") == std::string::npos) s += ".0";  // keep it a visible real; n/i for nan/inf
    return s;
}

// ---- term ordering for Plus: descending monomial degree ----
struct Atom { Expr e; int64_t exp; };

void collect_atoms(const Expr& t, std::vector<Atom>& out) {
    if (t->is_number() || t->kind == Kind::Real) return;
    if (t->has_head("Times")) { for (const auto& a : t->args) collect_atoms(a, out); return; }
    if (t->has_head("Power", 2) && t->args[1]->is_integer()) {
        int64_t k = 0;
        if (t->args[1]->q.num.to_int64(k) && !t->args[0]->has_head("Times")) {
            out.push_back({t->args[0], k});
            return;
        }
    }
    out.push_back({t, 1});
}

bool is_constant_symbol(const Expr& e) { return e->is_symbol() && (e->name == "Pi" || e->name == "E" || e->name == "I"); }

bool atom_less(const Expr& a, const Expr& b) {  // variables first (alphabetical), then constants (pi, E, I), then other atoms
    int ra = a->is_symbol() ? (is_constant_symbol(a) ? 1 : 0) : 2;
    int rb = b->is_symbol() ? (is_constant_symbol(b) ? 1 : 0) : 2;
    if (ra != rb) return ra < rb;
    return compare(a, b) < 0;
}

int cmp_terms_desc(const Expr& a, const Expr& b) {
    std::vector<Atom> aa, ab;
    collect_atoms(a, aa);
    collect_atoms(b, ab);
    std::vector<Expr> atoms;
    for (auto& x : aa) atoms.push_back(x.e);
    for (auto& x : ab) atoms.push_back(x.e);
    std::sort(atoms.begin(), atoms.end(), atom_less);
    atoms.erase(std::unique(atoms.begin(), atoms.end(), [](const Expr& p, const Expr& q) { return equal(p, q); }), atoms.end());
    auto exp_of = [](const std::vector<Atom>& v, const Expr& at) {
        int64_t s = 0;
        for (const auto& x : v) if (equal(x.e, at)) s += x.exp;
        return s;
    };
    for (const auto& at : atoms) {
        int64_t ea = exp_of(aa, at), eb = exp_of(ab, at);
        if (ea != eb) return ea > eb ? -1 : 1;
    }
    return compare(a, b);
}

std::string print(const Expr& e, int prec);

bool is_negative_number(const Expr& e) { return e->is_number() && e->q.is_negative(); }

// precedence: 1 sum, 2 product, 3 power/unary, 4 atom
std::string paren(const std::string& s, bool need) { return need ? "(" + s + ")" : s; }

// True if s is a single parenthesised group, e.g. "(a + b)" (but not "(a)*(b)").
bool is_wrapped(const std::string& s) {
    if (s.size() < 2 || s.front() != '(' || s.back() != ')') return false;
    int depth = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '(') ++depth;
        else if (s[i] == ')' && --depth == 0 && i + 1 != s.size()) return false;
    }
    return true;
}
std::string wrap(const std::string& s) { return is_wrapped(s) ? s : "(" + s + ")"; }

// Splits a Times into numeric coefficient, numerator factors and denominator factors.
void split_times(const Expr& t, Rational& coeff, std::vector<Expr>& numer, std::vector<Expr>& denom) {
    coeff = Rational(1);
    for (const auto& f : t->args) {
        if (f->is_number()) { coeff = coeff * f->q; continue; }
        if (f->has_head("Power", 2) && f->args[1]->is_number() && f->args[1]->q.is_negative() &&
            !(f->args[1]->q.is_integer() == false)) {
            Expr pos = f->args[1]->q == Rational(-1) ? f->args[0]
                                                      : apply2("Power", f->args[0], num(-f->args[1]->q));
            denom.push_back(pos);
            continue;
        }
        numer.push_back(f);
    }
}

std::string print_times(const Expr& t, int prec) {
    Rational coeff;
    std::vector<Expr> numer, denom;
    split_times(t, coeff, numer, denom);
    bool neg = coeff.is_negative();
    Rational mag = neg ? -coeff : coeff;
    BigInt p = mag.num, q = mag.den;

    std::string body;
    auto join = [&](const std::vector<Expr>& fs) {
        std::string s;
        for (size_t i = 0; i < fs.size(); ++i) {
            if (i) s += "*";
            s += print(fs[i], 2 + (fs[i]->has_head("Plus") ? 1 : 0) + (is_negative_number(fs[i]) ? 1 : 0));
        }
        return s;
    };
    std::string numer_s = join(numer);
    if (!p.is_one() || numer_s.empty()) numer_s = numer_s.empty() ? p.to_string() : p.to_string() + "*" + numer_s;
    std::string den_s;
    if (!q.is_one()) den_s = q.to_string();
    if (!denom.empty()) {
        std::string d = join(denom);
        if (den_s.empty()) {
            den_s = denom.size() == 1 && !denom[0]->has_head("Times") ? d : wrap(d);
        } else {
            den_s = "(" + den_s + "*" + d + ")";
        }
    }
    if (!den_s.empty()) {
        bool numer_needs_paren = numer.size() == 1 && p.is_one() && numer[0]->has_head("Plus");
        body = (numer_needs_paren ? wrap(numer_s) : numer_s) + "/" + den_s;
        if (numer.empty() && p.is_one()) body = "1/" + den_s;
    } else {
        body = numer_s;
    }
    if (neg) return paren("-" + body, prec >= 2);
    return paren(body, false);
}

std::string print_power(const Expr& e, int prec) {
    const Expr& b = e->args[0];
    const Expr& x = e->args[1];
    if (b->is_symbol_named("E")) return "exp(" + print(x, 0) + ")";
    if (x->is_number() && x->q == Rational(BigInt(1), BigInt(2))) return "sqrt(" + print(b, 0) + ")";
    if (x->is_number() && x->q.is_negative()) {
        Expr pos = x->q == Rational(-1) ? b : apply2("Power", b, num(-x->q));
        std::string d = print(pos, 2);
        bool need = pos->has_head("Plus") || pos->has_head("Times");
        return paren("1/" + (need ? wrap(d) : d), prec >= 2);
    }
    std::string bs = print(b, 3 + 1);  // base: parenthesise anything that is not an atom or a call
    if (b->has_head("Power") || is_negative_number(b) || (b->is_number() && !b->q.is_integer())) bs = "(" + print(b, 0) + ")";
    const bool plain_exp = x->is_symbol() || (x->is_integer() && !x->q.is_negative()) ||
                           (x->is_apply() && !x->has_head("Plus") && !x->has_head("Times") && !x->has_head("Power") &&
                            x->name != "Rule" && x->name != "Equal");
    std::string es = plain_exp ? print(x, 4) : "(" + print(x, 0) + ")";
    return bs + "^" + es;
}

std::string print(const Expr& e, int prec) {
    switch (e->kind) {
        case Kind::Number: {
            if (e->q.is_integer()) return paren(e->q.num.to_string(), e->q.is_negative() && prec >= 2);
            return paren(e->q.to_string(), prec >= 2);
        }
        case Kind::Real: return paren(real_to_string(e->d), e->d < 0 && prec >= 2);
        case Kind::Symbol: return e->name == "Pi" ? "pi" : e->name;
        case Kind::Apply: break;
    }
    const std::string& h = e->name;
    if (h == "Plus" && !e->args.empty()) {
        std::vector<Expr> terms = e->args;
        std::stable_sort(terms.begin(), terms.end(), [](const Expr& a, const Expr& b) { return cmp_terms_desc(a, b) < 0; });
        std::string s;
        for (size_t i = 0; i < terms.size(); ++i) {
            const Expr& t = terms[i];
            std::string ts = print(t, 1);
            bool neg = !ts.empty() && ts[0] == '-';
            if (i == 0) { s += ts; continue; }
            if (neg) s += " - " + ts.substr(1);
            else s += " + " + ts;
        }
        return paren(s, prec >= 2);
    }
    if (h == "Times" && !e->args.empty()) return print_times(e, prec);
    if (h == "Power" && e->args.size() == 2) return print_power(e, prec);
    if (h == "List") {
        std::string s = "{";
        for (size_t i = 0; i < e->args.size(); ++i) { if (i) s += ", "; s += print(e->args[i], 0); }
        return s + "}";
    }
    if (h == "Equal" && e->args.size() == 2) return paren(print(e->args[0], 0) + " == " + print(e->args[1], 0), prec >= 1);
    if (h == "Rule" && e->args.size() == 2) return paren(print(e->args[0], 0) + " -> " + print(e->args[1], 0), prec >= 1);
    if (h == "Factorial" && e->args.size() == 1) return print(e->args[0], 4) + "!";
    if (h == "SeriesData" && e->args.size() == 6 && e->args[2]->has_head("List") && e->args[3]->is_integer() && e->args[4]->is_integer()) {
        // Ascending powers of (x - x0), then O[x - x0]^nmax, like Mathics / Mathematica.
        const Expr& var = e->args[0];
        const Expr& x0 = e->args[1];
        Expr base = x0->is_number() && x0->q.is_zero() ? var : app("Plus", {var, num(-x0->q)});
        if (!(x0->is_number())) base = app("Plus", {var, app("Times", {minus_one(), x0})});
        int64_t nmin = 0, nmax = 0;
        e->args[3]->q.num.to_int64(nmin);
        e->args[4]->q.num.to_int64(nmax);
        std::string s;
        for (size_t i = 0; i < e->args[2]->args.size(); ++i) {
            const Expr& c = e->args[2]->args[i];
            if (c->is_number() && c->q.is_zero()) continue;
            int64_t k = nmin + static_cast<int64_t>(i);
            Expr term = k == 0 ? c : (k == 1 ? app("Times", {c, base}) : app("Times", {c, app("Power", {base, integer(k)})}));
            std::string ts = print(term, 1);
            bool neg = !ts.empty() && ts[0] == '-';
            if (s.empty()) s = ts;
            else s += neg ? " - " + ts.substr(1) : " + " + ts;
        }
        std::string o = "O[" + print(base, 0) + "]";
        if (nmax != 1) o += "^" + std::to_string(nmax);
        return (s.empty() ? "" : s + " + ") + o;
    }
    const char* pn = print_name(h);
    std::string s = pn ? pn : h;
    s += "(";
    for (size_t i = 0; i < e->args.size(); ++i) { if (i) s += ", "; s += print(e->args[i], 0); }
    return s + ")";
}

} // namespace

std::string to_string(const Expr& e) { return print(e, 0); }

} // namespace strata::math::cas
