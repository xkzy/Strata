// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/cas/poly.cpp - polynomials over Q and factorization (see poly.hpp)
#include "strata/math/cas/poly.hpp"

#include <algorithm>

namespace strata::math::cas {

// ---------------------------------------------------------------------------------------------------------------
// Q[x]
// ---------------------------------------------------------------------------------------------------------------
int poly_degree(const Poly& p) { return static_cast<int>(p.size()) - 1; }

void poly_trim(Poly& p) { while (!p.empty() && p.back().is_zero()) p.pop_back(); }

Poly poly_add(const Poly& a, const Poly& b) {
    Poly r(std::max(a.size(), b.size()));
    for (size_t i = 0; i < r.size(); ++i) {
        Rational x = i < a.size() ? a[i] : Rational(0);
        Rational y = i < b.size() ? b[i] : Rational(0);
        r[i] = x + y;
    }
    poly_trim(r);
    return r;
}

Poly poly_sub(const Poly& a, const Poly& b) {
    Poly r(std::max(a.size(), b.size()));
    for (size_t i = 0; i < r.size(); ++i) {
        Rational x = i < a.size() ? a[i] : Rational(0);
        Rational y = i < b.size() ? b[i] : Rational(0);
        r[i] = x - y;
    }
    poly_trim(r);
    return r;
}

Poly poly_mul(const Poly& a, const Poly& b) {
    if (a.empty() || b.empty()) return {};
    Poly r(a.size() + b.size() - 1, Rational(0));
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].is_zero()) continue;
        for (size_t j = 0; j < b.size(); ++j) r[i + j] = r[i + j] + a[i] * b[j];
    }
    poly_trim(r);
    return r;
}

Poly poly_scale(const Poly& a, const Rational& k) {
    Poly r;
    r.reserve(a.size());
    for (const auto& c : a) r.push_back(c * k);
    poly_trim(r);
    return r;
}

void poly_divmod(const Poly& a, const Poly& b, Poly& q, Poly& r) {
    if (b.empty()) throw CasMathError("polynomial division by zero");
    r = a;
    q.assign(a.size() >= b.size() ? a.size() - b.size() + 1 : 0, Rational(0));
    const Rational lc_inv = b.back().reciprocal();
    while (r.size() >= b.size()) {
        Rational factor = r.back() * lc_inv;
        size_t shift = r.size() - b.size();
        q[shift] = factor;
        for (size_t i = 0; i < b.size(); ++i) r[shift + i] = r[shift + i] - factor * b[i];
        poly_trim(r);
        if (r.empty()) break;
    }
    poly_trim(q);
}

Poly poly_monic(const Poly& a) {
    if (a.empty()) return a;
    return poly_scale(a, a.back().reciprocal());
}

Poly poly_gcd(const Poly& a, const Poly& b) {
    Poly x = a, y = b;
    while (!y.empty()) {
        Poly q, r;
        poly_divmod(x, y, q, r);
        x = std::move(y);
        y = std::move(r);
    }
    return poly_monic(x);
}

Poly poly_deriv(const Poly& a) {
    Poly r;
    for (size_t i = 1; i < a.size(); ++i) r.push_back(a[i] * Rational(static_cast<int64_t>(i)));
    poly_trim(r);
    return r;
}

Rational poly_eval(const Poly& a, const Rational& x) {
    Rational r(0);
    for (size_t i = a.size(); i-- > 0;) r = r * x + a[i];
    return r;
}

// ---------------------------------------------------------------------------------------------------------------
// Z[x] helpers
// ---------------------------------------------------------------------------------------------------------------
namespace {

void int_trim(IntPoly& p) { while (!p.empty() && p.back().is_zero()) p.pop_back(); }

IntPoly int_mul(const IntPoly& a, const IntPoly& b) {
    if (a.empty() || b.empty()) return {};
    IntPoly r(a.size() + b.size() - 1);
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].is_zero()) continue;
        for (size_t j = 0; j < b.size(); ++j) r[i + j] = r[i + j] + a[i] * b[j];
    }
    int_trim(r);
    return r;
}

BigInt int_content(const IntPoly& p) {
    BigInt g;
    for (const auto& c : p) g = BigInt::gcd(g, c);
    return g;
}

// Clears denominators and returns the primitive integer polynomial plus the rational factor c with f = c * result.
IntPoly to_primitive(const Poly& f, Rational& c) {
    BigInt l(1);
    for (const auto& x : f) l = l / BigInt::gcd(l, x.den) * x.den;
    IntPoly p;
    p.reserve(f.size());
    for (const auto& x : f) p.push_back(x.num * (l / x.den));
    BigInt g = int_content(p);
    if (g.is_zero()) g = BigInt(1);
    for (auto& v : p) v = v / g;
    c = Rational(g, l);
    return p;
}

Poly from_int(const IntPoly& p) {
    Poly r;
    r.reserve(p.size());
    for (const auto& c : p) r.push_back(Rational::from_bigint(c));
    return r;
}

// ---------------------------------------------------------------------------------------------------------------
// Z/p[x]
// ---------------------------------------------------------------------------------------------------------------
using u64 = uint64_t;
using ModPoly = std::vector<u64>;

struct Fp {
    u64 p;
    explicit Fp(u64 prime) : p(prime) {}

    void trim(ModPoly& a) const { while (!a.empty() && a.back() == 0) a.pop_back(); }
    u64 mulm(u64 a, u64 b) const { return (a * b) % p; }
    u64 powm(u64 a, u64 e) const {
        u64 r = 1;
        a %= p;
        while (e) { if (e & 1) r = mulm(r, a); a = mulm(a, a); e >>= 1; }
        return r;
    }
    u64 inv(u64 a) const { return powm(a, p - 2); }

    ModPoly add(const ModPoly& a, const ModPoly& b) const {
        ModPoly r(std::max(a.size(), b.size()), 0);
        for (size_t i = 0; i < r.size(); ++i) r[i] = ((i < a.size() ? a[i] : 0) + (i < b.size() ? b[i] : 0)) % p;
        trim(r);
        return r;
    }
    ModPoly sub(const ModPoly& a, const ModPoly& b) const {
        ModPoly r(std::max(a.size(), b.size()), 0);
        for (size_t i = 0; i < r.size(); ++i) r[i] = ((i < a.size() ? a[i] : 0) + p - (i < b.size() ? b[i] : 0)) % p;
        trim(r);
        return r;
    }
    ModPoly mul(const ModPoly& a, const ModPoly& b) const {
        if (a.empty() || b.empty()) return {};
        ModPoly r(a.size() + b.size() - 1, 0);
        for (size_t i = 0; i < a.size(); ++i) {
            if (!a[i]) continue;
            for (size_t j = 0; j < b.size(); ++j) r[i + j] = (r[i + j] + a[i] * b[j]) % p;
        }
        trim(r);
        return r;
    }
    void divmod(const ModPoly& a, const ModPoly& b, ModPoly& q, ModPoly& r) const {
        r = a;
        q.assign(a.size() >= b.size() ? a.size() - b.size() + 1 : 0, 0);
        const u64 li = inv(b.back());
        while (r.size() >= b.size()) {
            u64 f = mulm(r.back(), li);
            size_t shift = r.size() - b.size();
            q[shift] = f;
            for (size_t i = 0; i < b.size(); ++i) r[shift + i] = (r[shift + i] + p - mulm(f, b[i])) % p;
            trim(r);
            if (r.empty()) break;
        }
        trim(q);
    }
    ModPoly mod(const ModPoly& a, const ModPoly& m) const { ModPoly q, r; divmod(a, m, q, r); return r; }
    ModPoly monic(const ModPoly& a) const {
        if (a.empty()) return a;
        u64 li = inv(a.back());
        ModPoly r = a;
        for (auto& c : r) c = mulm(c, li);
        return r;
    }
    ModPoly gcd(ModPoly a, ModPoly b) const {
        while (!b.empty()) { ModPoly r = mod(a, b); a = std::move(b); b = std::move(r); }
        return monic(a);
    }
    ModPoly deriv(const ModPoly& a) const {
        ModPoly r;
        for (size_t i = 1; i < a.size(); ++i) r.push_back(mulm(a[i], i % p));
        trim(r);
        return r;
    }
    ModPoly powmod_u64(ModPoly base, u64 e, const ModPoly& m) const {
        ModPoly r{1};
        base = mod(base, m);
        while (e) { if (e & 1) r = mod(mul(r, base), m); base = mod(mul(base, base), m); e >>= 1; }
        return r;
    }
    ModPoly powmod_big(ModPoly base, const BigInt& e, const ModPoly& m) const {
        ModPoly r{1};
        base = mod(base, m);
        size_t bits = e.bit_length();
        for (size_t i = 0; i < bits; ++i) {
            if (e.bit(i)) r = mod(mul(r, base), m);
            if (i + 1 < bits) base = mod(mul(base, base), m);
        }
        return r;
    }
    // Extended Euclid: s*a + t*b = gcd (monic) for coprime a, b gives 1.
    void ext_gcd(const ModPoly& a, const ModPoly& b, ModPoly& s, ModPoly& t) const {
        ModPoly r0 = a, r1 = b, s0{1}, s1, t0, t1{1};
        while (!r1.empty()) {
            ModPoly q, r;
            divmod(r0, r1, q, r);
            ModPoly s2 = sub(s0, mul(q, s1));
            ModPoly t2 = sub(t0, mul(q, t1));
            r0 = std::move(r1); r1 = std::move(r);
            s0 = std::move(s1); s1 = std::move(s2);
            t0 = std::move(t1); t1 = std::move(t2);
        }
        u64 li = inv(r0.back());
        for (auto& c : s0) c = mulm(c, li);
        for (auto& c : t0) c = mulm(c, li);
        s = s0; t = t0;
    }
};

struct Rng {
    u64 s = 0x9E3779B97F4A7C15ull;
    u64 next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
};

// Equal-degree splitting: g is a product of irreducibles of degree d.
void equal_degree(const Fp& fp, const ModPoly& g, size_t d, Rng& rng, std::vector<ModPoly>& out, const TickFn& tick) {
    size_t n = g.size() - 1;
    if (n == d) { out.push_back(fp.monic(g)); return; }
    BigInt expo = (BigInt::pow(BigInt(static_cast<int64_t>(fp.p)), d) - BigInt(1)) / BigInt(2);
    for (int attempt = 0; attempt < 200; ++attempt) {
        tick(n * n + 8);
        ModPoly a(n);
        for (auto& c : a) c = rng.next() % fp.p;
        fp.trim(a);
        if (a.size() < 2) continue;
        ModPoly splitter;
        if (fp.p == 2) {
            // characteristic 2: the trace map a + a^2 + a^4 + ... + a^(2^(d-1)) is 0 or 1 modulo each irreducible factor
            ModPoly t = fp.mod(a, g), cur = t;
            for (size_t k = 1; k < d; ++k) { cur = fp.mod(fp.mul(cur, cur), g); t = fp.add(t, cur); }
            splitter = t;
        } else {
            splitter = fp.sub(fp.powmod_big(a, expo, g), ModPoly{1});   // (a^((p^d-1)/2) - 1)
        }
        ModPoly g1 = fp.gcd(g, splitter);
        if (g1.size() > 1 && g1.size() < g.size()) {
            ModPoly q, r;
            fp.divmod(g, g1, q, r);
            equal_degree(fp, g1, d, rng, out, tick);
            equal_degree(fp, fp.monic(q), d, rng, out, tick);
            return;
        }
    }
    throw CasLimitError("modular factorization did not converge");
}

// Factors a monic square-free polynomial mod p.
std::vector<ModPoly> factor_mod_p(const Fp& fp, ModPoly f, const TickFn& tick) {
    std::vector<ModPoly> out;
    Rng rng;
    ModPoly h{0, 1};  // x
    ModPoly fs = f;
    size_t d = 1;
    while (fs.size() > 1 && fs.size() - 1 >= 2 * d) {
        tick(fs.size() * fs.size() * 4 + 8);
        h = fp.powmod_u64(h, fp.p, fs);
        ModPoly g = fp.gcd(fs, fp.sub(h, ModPoly{0, 1}));
        if (g.size() > 1) {
            equal_degree(fp, g, d, rng, out, tick);
            ModPoly q, r;
            fp.divmod(fs, g, q, r);
            fs = fp.monic(q);
            h = fp.mod(h, fs);
        }
        ++d;
    }
    if (fs.size() > 1) out.push_back(fp.monic(fs));
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Hensel lifting
// ---------------------------------------------------------------------------------------------------------------
IntPoly lift_from_mod(const ModPoly& a) {
    IntPoly r;
    for (u64 c : a) r.push_back(BigInt(static_cast<int64_t>(c)));
    return r;
}

ModPoly reduce_mod_p(const IntPoly& a, const Fp& fp) {
    ModPoly r;
    BigInt pb(static_cast<int64_t>(fp.p));
    for (const auto& c : a) {
        BigInt m = c % pb;
        if (m.is_negative()) m = m + pb;
        int64_t v = 0;
        m.to_int64(v);
        r.push_back(static_cast<u64>(v));
    }
    fp.trim(r);
    return r;
}

IntPoly reduce_mod_m(const IntPoly& a, const BigInt& m) {
    IntPoly r;
    for (const auto& c : a) {
        BigInt x = c % m;
        if (x.is_negative()) x = x + m;
        r.push_back(std::move(x));
    }
    int_trim(r);
    return r;
}

// Lifts monic g*h == f (mod p) to g*h == f (mod p^k, M = p^k).
void hensel_pair(const IntPoly& f, const Fp& fp, const ModPoly& g0, const ModPoly& h0, const BigInt& target,
                 IntPoly& g, IntPoly& h, const TickFn& tick) {
    ModPoly s, t;
    fp.ext_gcd(g0, h0, s, t);
    g = lift_from_mod(g0);
    h = lift_from_mod(h0);
    BigInt M(static_cast<int64_t>(fp.p));
    const BigInt pb(static_cast<int64_t>(fp.p));
    while (M < target) {
        tick(f.size() * f.size() + 16);
        IntPoly e = f;
        IntPoly gh = int_mul(g, h);
        // e = f - g*h  (divisible by M)
        IntPoly diff(std::max(e.size(), gh.size()));
        for (size_t i = 0; i < diff.size(); ++i) diff[i] = (i < e.size() ? e[i] : BigInt()) - (i < gh.size() ? gh[i] : BigInt());
        int_trim(diff);
        IntPoly eps_int;
        for (const auto& c : diff) eps_int.push_back(c / M);
        ModPoly eps = reduce_mod_p(eps_int, fp);
        ModPoly q, r;
        fp.divmod(fp.mul(eps, t), g0, q, r);   // eps*t = q*g0 + r
        ModPoly dg = r;
        ModPoly dh = fp.add(fp.mul(eps, s), fp.mul(q, h0));
        // g += M*dg, h += M*dh
        auto bump = [&](IntPoly& a, const ModPoly& d) {
            if (a.size() < d.size()) a.resize(d.size());
            for (size_t i = 0; i < d.size(); ++i) a[i] = a[i] + M * BigInt(static_cast<int64_t>(d[i]));
            int_trim(a);
        };
        bump(g, dg);
        bump(h, dh);
        M = M * pb;
    }
    g = reduce_mod_m(g, M);
    h = reduce_mod_m(h, M);
    (void)target;
}

ModPoly product_mod_p(const Fp& fp, const std::vector<ModPoly>& v, size_t lo, size_t hi) {
    ModPoly r{1};
    for (size_t i = lo; i < hi; ++i) r = fp.mul(r, v[i]);
    return r;
}

// Lifts the modular factorization of the monic polynomial f to modulus >= target; out[i] is monic mod M.
void hensel_tree(const IntPoly& f, const Fp& fp, const std::vector<ModPoly>& facs, size_t lo, size_t hi, const BigInt& target,
                 std::vector<IntPoly>& out, const TickFn& tick) {
    if (hi - lo == 1) { out.push_back(f); return; }
    size_t mid = lo + (hi - lo) / 2;
    ModPoly g0 = product_mod_p(fp, facs, lo, mid);
    ModPoly h0 = product_mod_p(fp, facs, mid, hi);
    IntPoly g, h;
    hensel_pair(f, fp, g0, h0, target, g, h, tick);
    hensel_tree(g, fp, facs, lo, mid, target, out, tick);
    hensel_tree(h, fp, facs, mid, hi, target, out, tick);
}

// ---------------------------------------------------------------------------------------------------------------
// Recombination
// ---------------------------------------------------------------------------------------------------------------
IntPoly symmetric(const IntPoly& a, const BigInt& M) {
    BigInt half = M / BigInt(2);
    IntPoly r;
    for (const auto& c : a) r.push_back(c > half ? c - M : c);
    int_trim(r);
    return r;
}

// Exact division of integer polynomials where the divisor is monic. Returns false if not divisible.
bool int_divide_monic(const IntPoly& a, const IntPoly& b, IntPoly& q) {
    IntPoly r = a;
    if (r.size() < b.size()) return false;
    q.assign(r.size() - b.size() + 1, BigInt());
    while (r.size() >= b.size()) {
        BigInt f = r.back();
        size_t shift = r.size() - b.size();
        q[shift] = f;
        for (size_t i = 0; i < b.size(); ++i) r[shift + i] = r[shift + i] - f * b[i];
        int_trim(r);
        if (r.empty()) break;
        if (r.size() < b.size()) break;
    }
    int_trim(q);
    return r.empty();
}

// Factors a monic square-free primitive F in Z[x] into monic irreducible factors.
std::vector<IntPoly> factor_monic(const IntPoly& F, const TickFn& tick) {
    const size_t n = F.size() - 1;
    if (n <= 1) return {F};

    // choose a prime for which F is square-free mod p, preferring few modular factors
    static const u64 primes[] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
    u64 best_p = 0;
    std::vector<ModPoly> best;
    int tried = 0;
    for (u64 p : primes) {
        Fp fp(p);
        ModPoly fm = reduce_mod_p(F, fp);
        if (fm.size() != F.size()) continue;
        ModPoly g = fp.gcd(fm, fp.deriv(fm));
        if (g.size() != 1) continue;
        std::vector<ModPoly> facs = factor_mod_p(fp, fm, tick);
        if (best.empty() || facs.size() < best.size()) { best = std::move(facs); best_p = p; }
        if (best.size() == 1) return {F};  // irreducible mod p => irreducible over Z
        if (++tried >= 6) break;
    }
    if (best.empty()) throw CasLimitError("no suitable prime for factorization");

    // Mignotte-style bound B = C(n, n/2) * ||F||_1 ; need p^k > 2B
    BigInt norm1;
    for (const auto& c : F) norm1 = norm1 + c.abs();
    BigInt binom(1);
    for (size_t i = 1; i <= n / 2; ++i) binom = binom * BigInt(static_cast<int64_t>(n - n / 2 + i)) / BigInt(static_cast<int64_t>(i));
    BigInt bound = binom * norm1 * BigInt(2) + BigInt(1);
    Fp fp(best_p);
    std::vector<IntPoly> lifted;
    hensel_tree(F, fp, best, 0, best.size(), bound, lifted, tick);
    // all lifted factors share the same final modulus: the smallest power of p >= bound
    BigInt M(static_cast<int64_t>(best_p));
    while (M < bound) M = M * BigInt(static_cast<int64_t>(best_p));
    for (auto& z : lifted) z = reduce_mod_m(z, M);

    // Zassenhaus recombination
    std::vector<IntPoly> result;
    IntPoly rest = F;
    std::vector<size_t> idx(lifted.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    size_t s = 1;
    while (idx.size() >= 2 * s) {
        bool found = false;
        std::vector<size_t> pick(s);
        // iterate combinations of size s over idx
        std::vector<size_t> c(s);
        for (size_t i = 0; i < s; ++i) c[i] = i;
        while (true) {
            tick(rest.size() * rest.size() * s + 16);
            IntPoly prod{BigInt(1)};
            for (size_t i = 0; i < s; ++i) prod = reduce_mod_m(int_mul(prod, lifted[idx[c[i]]]), M);
            IntPoly cand = symmetric(prod, M);
            IntPoly q;
            bool ok = !cand.empty() && cand.size() >= 2;
            if (ok && !rest[0].is_zero() && !cand[0].is_zero()) ok = (rest[0] % cand[0]).is_zero();
            if (ok && int_divide_monic(rest, cand, q)) {
                result.push_back(cand);
                rest = q;
                std::vector<size_t> next;
                for (size_t i = 0, ci = 0; i < idx.size(); ++i) {
                    if (ci < s && c[ci] == i) { ++ci; continue; }
                    next.push_back(idx[i]);
                }
                idx = std::move(next);
                found = true;
                break;
            }
            // next combination
            int i = static_cast<int>(s) - 1;
            while (i >= 0 && c[i] == idx.size() - s + i) --i;
            if (i < 0) break;
            ++c[i];
            for (size_t j = i + 1; j < s; ++j) c[j] = c[j - 1] + 1;
        }
        if (!found) ++s;
    }
    if (rest.size() > 1) result.push_back(rest);
    return result;
}

// Square-free factorization over Q (Yun): f = c * prod g_i^i with g_i square-free and pairwise coprime.
std::vector<std::pair<Poly, int>> squarefree_decomposition(const Poly& f) {
    std::vector<std::pair<Poly, int>> out;
    Poly fp = poly_deriv(f);
    if (fp.empty()) { out.push_back({poly_monic(f), 1}); return out; }
    Poly c = poly_gcd(f, fp);
    Poly q, r;
    Poly w, y;
    poly_divmod(f, c, w, r);
    poly_divmod(fp, c, y, r);
    Poly z = poly_sub(y, poly_deriv(w));
    int i = 1;
    while (poly_degree(w) > 0) {
        Poly g = poly_gcd(w, z);
        if (poly_degree(g) > 0) out.push_back({g, i});
        Poly w2, y2;
        poly_divmod(w, g, w2, r);
        poly_divmod(z, g, y2, r);
        w = w2;
        z = poly_sub(y2, poly_deriv(w));
        ++i;
    }
    return out;
}

} // namespace

Factorization factor_over_q(const Poly& f, const TickFn& tick) {
    if (f.empty()) throw CasMathError("cannot factor the zero polynomial");
    Factorization result;
    Rational c;
    IntPoly prim = to_primitive(f, c);
    // Work with the primitive integer polynomial (sign handled below).
    Poly fq = from_int(prim);
    result.content = c;
    if (poly_degree(fq) <= 0) { result.content = result.content * fq[0]; return result; }

    std::vector<std::pair<Poly, int>> sqf = squarefree_decomposition(fq);
    Rational reconstructed(1);
    (void)reconstructed;
    for (auto& [g, mult] : sqf) {
        Rational gc;
        IntPoly gi = to_primitive(g, gc);          // gi primitive; g = gc * gi
        const size_t n = gi.size() - 1;
        std::vector<IntPoly> irreducibles;
        if (n == 1) {
            irreducibles.push_back(gi);
        } else {
            // monic transform: F(y) = lc^(n-1) g(y/lc)
            BigInt lc = gi.back();
            IntPoly F(gi.size());
            BigInt pw(1);
            for (size_t i = n; i-- > 0;) { /* fill below */ }
            // coefficient of y^i is g_i * lc^(n-1-i)
            std::vector<BigInt> lc_pow(n);
            lc_pow[0] = BigInt(1);
            for (size_t i = 1; i < n; ++i) lc_pow[i] = lc_pow[i - 1] * lc;
            for (size_t i = 0; i < n; ++i) F[i] = gi[i] * lc_pow[n - 1 - i];
            F[n] = BigInt(1);
            std::vector<IntPoly> mf = factor_monic(F, tick);
            for (auto& G : mf) {
                // G(lc * x): coefficient i times lc^i, then primitive part
                IntPoly h(G.size());
                BigInt pwr(1);
                for (size_t i = 0; i < G.size(); ++i) { h[i] = G[i] * pwr; pwr = pwr * lc; }
                BigInt ct = int_content(h);
                if (ct.is_zero()) ct = BigInt(1);
                for (auto& v : h) v = v / ct;
                irreducibles.push_back(std::move(h));
            }
        }
        for (auto& h : irreducibles) {
            if (h.back().is_negative()) {
                for (auto& v : h) v = -v;
                result.content = result.content * Rational(-1);
            }
            result.factors.push_back({from_int(h), mult});
        }
        // gc accounts for scalar between the squarefree part g and its primitive form
        Rational scalar = gc;
        for (int k = 0; k < mult; ++k) result.content = result.content * scalar;
    }
    // squarefree_decomposition returns monic parts, so recompute the exact leftover constant by division
    Poly prod{Rational(1)};
    for (const auto& [h, m] : result.factors)
        for (int k = 0; k < m; ++k) prod = poly_mul(prod, h);
    Poly q, r;
    poly_divmod(f, prod, q, r);
    if (!r.empty() || poly_degree(q) != 0) throw CasLimitError("internal factorization inconsistency");
    result.content = q[0];
    // deterministic order: by degree, then lexicographic on coefficients
    std::sort(result.factors.begin(), result.factors.end(), [](const auto& a, const auto& b) {
        if (a.first.size() != b.first.size()) return a.first.size() < b.first.size();
        for (size_t i = a.first.size(); i-- > 0;) {
            if (a.first[i] != b.first[i]) return a.first[i] < b.first[i];
        }
        return a.second < b.second;
    });
    return result;
}


// ---------------------------------------------------------------------------------------------------------------
// public factorization over GF(p)
// ---------------------------------------------------------------------------------------------------------------
namespace {

ModPoly gfp_derivative(const Fp& fp, const ModPoly& a) {
    ModPoly d;
    for (size_t i = 1; i < a.size(); ++i) d.push_back(fp.mulm(a[i], static_cast<u64>(i) % fp.p));
    fp.trim(d);
    return d;
}

// f = g(x^p): the coefficients at multiples of p
ModPoly gfp_pth_root(const Fp& fp, const ModPoly& a) {
    ModPoly r;
    for (size_t i = 0; i < a.size(); i += fp.p) r.push_back(a[i]);
    fp.trim(r);
    return r;
}

void gfp_squarefree(const Fp& fp, ModPoly f, int mult, std::vector<std::pair<ModPoly, int>>& out, const TickFn& tick, int depth = 0) {
    if (f.size() <= 1) return;
    if (depth > 64) throw CasLimitError("modular factorization: too deep");
    f = fp.monic(f);
    ModPoly d = gfp_derivative(fp, f);
    if (d.empty()) {   // f is a p-th power of g(x)
        gfp_squarefree(fp, gfp_pth_root(fp, f), mult * static_cast<int>(fp.p), out, tick, depth + 1);
        return;
    }
    ModPoly c = fp.gcd(f, d), q, r;
    fp.divmod(f, c, q, r);
    ModPoly w = fp.monic(q);
    int i = 1;
    while (w.size() > 1) {
        tick(w.size() * w.size() + 8);
        ModPoly y = fp.gcd(w, c);
        ModPoly fac, rem;
        fp.divmod(w, y, fac, rem);
        if (fac.size() > 1) out.push_back({fp.monic(fac), i * mult});
        w = y;
        ModPoly c2, r2;
        fp.divmod(c, y, c2, r2);
        c = fp.monic(c2);
        ++i;
    }
    if (c.size() > 1) gfp_squarefree(fp, gfp_pth_root(fp, c), mult * static_cast<int>(fp.p), out, tick, depth + 1);
}

} // namespace

ModFactorization factor_over_gfp(std::vector<uint64_t> f, uint64_t p, const TickFn& tick) {
    Fp fp(p);
    for (auto& c : f) c %= p;
    fp.trim(f);
    ModFactorization res;
    if (f.empty()) return res;
    res.content = f.back();
    if (f.size() == 1) return res;
    std::vector<std::pair<ModPoly, int>> parts;
    gfp_squarefree(fp, f, 1, parts, tick);
    for (auto& [g, m] : parts)
        for (auto& irr : factor_mod_p(fp, g, tick)) res.factors.push_back({irr, m});
    // merge equal factors found in different square-free parts, then order by degree and coefficients
    std::sort(res.factors.begin(), res.factors.end(), [](const auto& a, const auto& b) {
        if (a.first.size() != b.first.size()) return a.first.size() < b.first.size();
        return a.first < b.first;
    });
    std::vector<std::pair<std::vector<uint64_t>, int>> merged;
    for (auto& fm : res.factors) {
        if (!merged.empty() && merged.back().first == fm.first) merged.back().second += fm.second;
        else merged.push_back(fm);
    }
    res.factors = std::move(merged);
    return res;
}

} // namespace strata::math::cas
