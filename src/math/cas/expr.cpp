// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/cas/expr.cpp - expression tree basics (see expr.hpp)
#include "strata/math/cas/expr.hpp"

#include <algorithm>
#include <cmath>

namespace strata::math::cas {

// ---- Rational ----
Rational::Rational(BigInt n, BigInt d) {
    if (d.is_zero()) throw CasMathError("division by zero");
    if (d.is_negative()) { n = -n; d = -d; }
    BigInt g = BigInt::gcd(n, d);
    if (!g.is_one() && !g.is_zero()) { n = n / g; d = d / g; }
    num = std::move(n);
    den = std::move(d);
}

double Rational::to_double() const {
    if (den.is_one()) return num.to_double();
    // Scale both so the quotient keeps precision even for huge operands.
    size_t nb = num.bit_length(), db = den.bit_length();
    if (nb < 900 && db < 900) return num.to_double() / den.to_double();
    size_t shift = std::max(nb, db) - 900;
    BigInt p = BigInt::pow(BigInt(2), shift);
    return (num / p).to_double() / (den / p).to_double();
}

std::string Rational::to_string() const {
    return den.is_one() ? num.to_string() : num.to_string() + "/" + den.to_string();
}

Rational Rational::operator-() const { Rational r = *this; r.num = -r.num; return r; }
Rational operator+(const Rational& a, const Rational& b) {
    return Rational(a.num * b.den + b.num * a.den, a.den * b.den);
}
Rational operator-(const Rational& a, const Rational& b) { return a + (-b); }
Rational operator*(const Rational& a, const Rational& b) { return Rational(a.num * b.num, a.den * b.den); }
Rational operator/(const Rational& a, const Rational& b) {
    if (b.num.is_zero()) throw CasMathError("division by zero");
    return Rational(a.num * b.den, a.den * b.num);
}
bool operator<(const Rational& a, const Rational& b) { return BigInt::compare(a.num * b.den, b.num * a.den) < 0; }
Rational Rational::reciprocal() const {
    if (num.is_zero()) throw CasMathError("division by zero");
    return Rational(den, num);
}
Rational Rational::pow(int64_t exp) const {
    if (exp == 0) return Rational(1);
    uint64_t e = exp < 0 ? static_cast<uint64_t>(-(exp + 1)) + 1 : static_cast<uint64_t>(exp);
    Rational r;
    r.num = BigInt::pow(num, e);
    r.den = BigInt::pow(den, e);
    if (exp < 0) {
        if (r.num.is_zero()) throw CasMathError("division by zero");
        Rational inv(r.den, r.num);
        return inv;
    }
    return r;
}

// ---- constructors ----
static Expr make(Node n) { return std::make_shared<const Node>(std::move(n)); }

Expr num(const Rational& q) { Node n; n.kind = Kind::Number; n.q = q; return make(std::move(n)); }
Expr integer(int64_t v) { return num(Rational(v)); }
Expr real(double d) { Node n; n.kind = Kind::Real; n.d = d; return make(std::move(n)); }
Expr symbol(const std::string& name) { Node n; n.kind = Kind::Symbol; n.name = name; return make(std::move(n)); }
Expr app(const std::string& head, std::vector<Expr> args) {
    Node n; n.kind = Kind::Apply; n.name = head; n.args = std::move(args); return make(std::move(n));
}
Expr apply1(const std::string& head, Expr a) { return app(head, {std::move(a)}); }
Expr apply2(const std::string& head, Expr a, Expr b) { return app(head, {std::move(a), std::move(b)}); }

Expr zero() { static const Expr z = integer(0); return z; }
Expr one() { static const Expr o = integer(1); return o; }
Expr minus_one() { static const Expr m = integer(-1); return m; }

// ---- equality / order ----
bool equal(const Expr& a, const Expr& b) {
    if (a.get() == b.get()) return true;
    if (a->kind != b->kind) return false;
    switch (a->kind) {
        case Kind::Number: return a->q == b->q;
        case Kind::Real: return a->d == b->d;
        case Kind::Symbol: return a->name == b->name;
        case Kind::Apply:
            if (a->name != b->name || a->args.size() != b->args.size()) return false;
            for (size_t i = 0; i < a->args.size(); ++i) if (!equal(a->args[i], b->args[i])) return false;
            return true;
    }
    return false;
}

static int rank(const Expr& e) {
    switch (e->kind) {
        case Kind::Number: case Kind::Real: return 0;
        case Kind::Symbol: return 1;
        case Kind::Apply: return 2;
    }
    return 3;
}

int compare(const Expr& a, const Expr& b) {
    // A Power orders by its base (then exponent), so x, x^2, y sort as in Mathematica rather than y before x^2.
    if (a->has_head("Power", 2) || b->has_head("Power", 2)) {
        const Expr& ba = a->has_head("Power", 2) ? a->args[0] : a;
        const Expr& bb = b->has_head("Power", 2) ? b->args[0] : b;
        // (numbers still sort first: only symbolic bases are compared this way)
        if (!(a->kind == Kind::Number || a->kind == Kind::Real || b->kind == Kind::Number || b->kind == Kind::Real) &&
            !(a->has_head("Power", 2) && b->has_head("Power", 2) && equal(ba, bb) == false && false)) {
            int c = compare(ba, bb);
            if (c != 0) return c;
            const Expr ea = a->has_head("Power", 2) ? a->args[1] : integer(1);
            const Expr eb = b->has_head("Power", 2) ? b->args[1] : integer(1);
            return compare(ea, eb);
        }
    }
    int ra = rank(a), rb = rank(b);
    if (ra != rb) return ra < rb ? -1 : 1;
    if (ra == 0) {
        if (a->kind == Kind::Number && b->kind == Kind::Number) {
            if (a->q == b->q) return 0;
            return a->q < b->q ? -1 : 1;
        }
        double da = a->kind == Kind::Number ? a->q.to_double() : a->d;
        double db = b->kind == Kind::Number ? b->q.to_double() : b->d;
        if (da == db) return a->kind == b->kind ? 0 : (a->kind == Kind::Number ? -1 : 1);
        return da < db ? -1 : 1;
    }
    if (ra == 1) return a->name == b->name ? 0 : (a->name < b->name ? -1 : 1);
    if (a->name != b->name) return a->name < b->name ? -1 : 1;
    if (a->args.size() != b->args.size()) return a->args.size() < b->args.size() ? -1 : 1;
    for (size_t i = 0; i < a->args.size(); ++i) {
        int c = compare(a->args[i], b->args[i]);
        if (c != 0) return c;
    }
    return 0;
}

size_t leaf_count(const Expr& e) {
    if (!e->is_apply()) return 1;
    size_t n = 1;
    for (const auto& a : e->args) n += leaf_count(a);
    return n;
}

bool depends_on(const Expr& e, const std::string& var) {
    if (e->is_symbol()) return e->name == var;
    if (!e->is_apply()) return false;
    for (const auto& a : e->args) if (depends_on(a, var)) return true;
    return false;
}

static void collect_symbols(const Expr& e, std::vector<std::string>& out) {
    if (e->is_symbol()) {
        if (e->name != "Pi" && e->name != "E" && e->name != "I" && e->name != "Infinity" && e->name != "Indeterminate") out.push_back(e->name);
        return;
    }
    if (e->is_apply()) for (const auto& a : e->args) collect_symbols(a, out);
}

void free_symbols(const Expr& e, std::vector<std::string>& out) {
    std::vector<std::string> tmp;
    collect_symbols(e, tmp);
    std::sort(tmp.begin(), tmp.end());
    tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
    out = std::move(tmp);
}

} // namespace strata::math::cas
