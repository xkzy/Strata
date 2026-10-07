// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/cas/numtheory.cpp - number theory, integer sequences and iteration (Range/Table/Sum/Product)
//
// Native C++ implementation of the behaviour of Mathics' number theory / list-iteration builtins
// (mathics/builtin/numbers/numbertheory.py, mathics/builtin/list/ ... Sum/Product/Table/Range); Mathics delegates the
// integer algorithms to SymPy, here they are written directly: deterministic Miller-Rabin, Pollard rho, Faulhaber sums.
#include "strata/math/cas/engine.hpp"

#include <algorithm>
#include <cmath>

namespace strata::math::cas {

namespace {

using B = BigInt;

B powmod(B base, B e, const B& m) {
    B r = B(1) % m;
    base = base % m;
    if (base.is_negative()) base = base + m;
    while (!e.is_zero()) {
        if (!e.is_even()) r = (r * base) % m;
        e = e / B(2);
        if (!e.is_zero()) base = (base * base) % m;
    }
    return r;
}

const int kSmallPrimes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71};

// Deterministic for n < 3.3e24 (first 13 prime bases); beyond that 20 bases (error probability is negligible, but it
// is a probable-prime test and is documented as such).
bool is_probable_prime(const B& n) {
    if (n < B(2)) return false;
    for (int p : kSmallPrimes) {
        if (n == B(p)) return true;
        if ((n % B(p)).is_zero()) return false;
    }
    B d = n - B(1);
    int s = 0;
    while (d.is_even()) { d = d / B(2); ++s; }
    const int nb = n.bit_length() < 81 ? 13 : 20;
    const B nm1 = n - B(1);
    for (int i = 0; i < nb; ++i) {
        B x = powmod(B(kSmallPrimes[i]), d, n);
        if (x.is_one() || x == nm1) continue;
        bool composite = true;
        for (int r = 1; r < s; ++r) {
            x = (x * x) % n;
            if (x == nm1) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

std::vector<uint32_t> sieve(uint64_t limit) {
    std::vector<bool> comp(limit + 1, false);
    std::vector<uint32_t> out;
    for (uint64_t i = 2; i <= limit; ++i) {
        if (comp[i]) continue;
        out.push_back(static_cast<uint32_t>(i));
        for (uint64_t j = i * i; j <= limit; j += i) comp[j] = true;
    }
    return out;
}

B gcd_abs(const B& a, const B& b) { return B::gcd(a.abs(), b.abs()); }

struct Factorizer {
    std::function<void()> tick;
    std::vector<std::pair<B, uint64_t>> out;

    B rho(const B& n) {
        if (n.is_even()) return B(2);
        for (int64_t c = 1;; ++c) {
            B x(2), y(2), d(1), C(c);
            auto f = [&](const B& v) { return (v * v + C) % n; };
            while (d.is_one()) {
                x = f(x);
                y = f(f(y));
                d = B::gcd(x > y ? x - y : y - x, n);
                tick();
            }
            if (d != n) return d;
        }
    }
    void add(const B& p, uint64_t e) {
        for (auto& pr : out) if (pr.first == p) { pr.second += e; return; }
        out.push_back({p, e});
    }
    void rec(B n) {
        if (n.is_one()) return;
        if (is_probable_prime(n)) { add(n, 1); return; }
        B d = rho(n);
        rec(d);
        rec(n / d);
    }
    // n >= 2
    void run(B n) {
        static const std::vector<uint32_t> small = sieve(5000);
        for (uint32_t p : small) {
            if (B(p) * B(p) > n) break;
            uint64_t e = 0;
            while ((n % B(p)).is_zero()) { n = n / B(p); ++e; }
            if (e) add(B(p), e);
        }
        if (n > B(1)) rec(n);
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    }
};

Expr boolean(bool v) { return symbol(v ? "True" : "False"); }
Expr int_expr(const B& v) { return num(Rational::from_bigint(v)); }
Expr list_of(std::vector<Expr> v) { return app("List", std::move(v)); }

bool get_int(const Expr& e, B& out) {
    if (!e->is_integer()) return false;
    out = e->q.num;
    return true;
}
bool get_i64(const Expr& e, int64_t& out) { return e->is_integer() && e->q.num.to_int64(out); }

// Fibonacci pair by fast doubling: returns (F(n), F(n+1))
std::pair<B, B> fib_pair(uint64_t n) {
    if (n == 0) return {B(0), B(1)};
    auto p = fib_pair(n / 2);
    B c = p.first * (p.second * B(2) - p.first);
    B d = p.first * p.first + p.second * p.second;
    if (n % 2 == 0) return {c, d};
    return {d, c + d};
}

// Bernoulli numbers with B1 = +1/2 (the convention of sums of powers)
std::vector<Rational> bernoulli_plus(int n) {
    std::vector<Rational> b(static_cast<size_t>(n) + 1);
    b[0] = Rational(1);
    for (int m = 1; m <= n; ++m) {
        Rational acc(0);
        BigInt binom(1);   // C(m+1, 0)
        for (int j = 0; j < m; ++j) {
            acc = acc + Rational::from_bigint(binom) * b[static_cast<size_t>(j)];
            binom = binom * BigInt(m + 1 - j) / BigInt(j + 1);
        }
        b[static_cast<size_t>(m)] = -(acc / Rational(m + 1));
    }
    if (n >= 1) b[1] = Rational(BigInt(1), BigInt(2));
    return b;
}

bool free_of(const Expr& e, const std::string& v) { return !depends_on(e, v); }

} // namespace

Expr Engine::number_theory(const std::string& head, const std::vector<Expr>& args) {
    auto tk = [this]() { tick(); };
    B n, m;
    int64_t k = 0;

    if (head == "PrimeQ" && args.size() == 1) {
        if (get_int(args[0], n)) return boolean(is_probable_prime(n.abs()));
        if (args[0]->is_number() || args[0]->kind == Kind::Real) return boolean(false);
        return nullptr;
    }
    if (head == "EvenQ" || head == "OddQ") {
        if (args.size() != 1) return nullptr;
        if (get_int(args[0], n)) return boolean(n.is_even() == (head == "EvenQ"));
        if (args[0]->is_number() || args[0]->kind == Kind::Real || args[0]->is_symbol()) return boolean(false);
        return nullptr;
    }
    if (head == "IntegerQ" && args.size() == 1) {
        if (args[0]->is_number() || args[0]->kind == Kind::Real || args[0]->is_symbol()) return boolean(args[0]->is_integer());
        return nullptr;
    }
    if (head == "Prime" && args.size() == 1 && get_i64(args[0], k) && k >= 1) {
        if (k > 5'000'000) throw CasLimitError("Prime: index too large");
        const double x = static_cast<double>(k);
        const uint64_t limit = k < 6 ? 15 : static_cast<uint64_t>(x * (std::log(x) + std::log(std::log(x)))) + 10;
        auto pr = sieve(limit);
        return integer(pr[static_cast<size_t>(k - 1)]);
    }
    if (head == "PrimePi" && args.size() == 1 && get_i64(args[0], k)) {
        if (k < 2) return zero();
        if (k > 20'000'000) throw CasLimitError("PrimePi: argument too large");
        return integer(static_cast<int64_t>(sieve(static_cast<uint64_t>(k)).size()));
    }
    if (head == "NextPrime" && (args.size() == 1 || args.size() == 2) && get_int(args[0], n)) {
        int64_t steps = 1;
        if (args.size() == 2 && !get_i64(args[1], steps)) return nullptr;
        if (steps == 0 || abs_u64(steps) > 1000) return nullptr;
        const int dir = steps > 0 ? 1 : -1;
        B cur = n;
        for (uint64_t s = 0; s < abs_u64(steps); ++s) {
            for (int guard = 0;; ++guard) {
                cur = cur + B(dir);
                tk();
                if (guard > 2'000'000) throw CasLimitError("NextPrime: search too long");
                if (cur < B(2)) { if (dir < 0) return nullptr; continue; }
                if (is_probable_prime(cur)) break;
            }
        }
        return int_expr(cur);
    }
    if ((head == "FactorInteger") && args.size() == 1 && args[0]->is_number()) {
        const Rational& q = args[0]->q;
        if (q.is_zero()) return list_of({list_of({zero(), one()})});
        std::vector<Expr> rows;
        auto emit = [&](const B& v, bool negative_exp) {
            B a = v.abs();
            if (a.is_one()) return;
            Factorizer f; f.tick = tk; f.run(a);
            for (auto& pr : f.out) rows.push_back(list_of({int_expr(pr.first), integer(negative_exp ? -static_cast<int64_t>(pr.second) : static_cast<int64_t>(pr.second))}));
        };
        std::vector<Expr> parts;
        if (q.is_negative()) parts.push_back(list_of({minus_one(), one()}));
        emit(q.num, false);
        std::vector<Expr> dens = rows;
        rows.clear();
        emit(q.den, true);
        // merge: numerator primes first, denominator primes interleaved by prime value
        std::vector<Expr> all = dens;
        all.insert(all.end(), rows.begin(), rows.end());
        std::sort(all.begin(), all.end(), [](const Expr& a, const Expr& b) { return a->args[0]->q.num < b->args[0]->q.num; });
        parts.insert(parts.end(), all.begin(), all.end());
        return list_of(std::move(parts));
    }
    if ((head == "Divisors" || head == "EulerPhi" || head == "MoebiusMu" || head == "DivisorSigma") && !args.empty()) {
        const Expr& target = head == "DivisorSigma" ? (args.size() == 2 ? args[1] : args[0]) : args[0];
        if (head == "DivisorSigma" && (args.size() != 2 || !get_i64(args[0], k) || k < 0 || k > 64)) return nullptr;
        if (args.size() != (head == "DivisorSigma" ? 2u : 1u) || !get_int(target, n) || n.is_zero()) return nullptr;
        B a = n.abs();
        std::vector<std::pair<B, uint64_t>> fac;
        if (!a.is_one()) { Factorizer f; f.tick = tk; f.run(a); fac = f.out; }
        if (head == "EulerPhi") {
            B r(1);
            for (auto& p : fac) r = r * BigInt::pow(p.first, p.second - 1) * (p.first - B(1));
            return int_expr(r);
        }
        if (head == "MoebiusMu") {
            for (auto& p : fac) if (p.second > 1) return zero();
            return integer(fac.size() % 2 ? -1 : 1);
        }
        if (head == "DivisorSigma") {
            B r(1);
            for (auto& p : fac) {
                B s(0);
                for (uint64_t e = 0; e <= p.second; ++e) s = s + BigInt::pow(p.first, static_cast<uint64_t>(k) * e);
                r = r * s;
            }
            return int_expr(r);
        }
        // Divisors
        std::vector<B> divs{B(1)};
        for (auto& p : fac) {
            std::vector<B> next;
            B pw(1);
            for (uint64_t e = 0; e <= p.second; ++e) {
                for (auto& d : divs) next.push_back(d * pw);
                pw = pw * p.first;
            }
            divs = std::move(next);
            if (divs.size() > 200000) throw CasLimitError("Divisors: too many divisors");
        }
        std::sort(divs.begin(), divs.end());
        std::vector<Expr> out;
        for (auto& d : divs) out.push_back(int_expr(d));
        return list_of(std::move(out));
    }
    if (head == "PowerMod" && args.size() == 3 && get_int(args[0], n) && get_int(args[2], m) && !m.is_zero()) {
        B e;
        if (!get_int(args[1], e)) return nullptr;
        m = m.abs();
        if (e.is_negative()) {   // modular inverse via extended Euclid
            B a = n % m; if (a.is_negative()) a = a + m;
            B r0 = m, r1 = a, t0(0), t1(1);
            while (!r1.is_zero()) { B q = r0 / r1; B r2 = r0 - q * r1; r0 = r1; r1 = r2; B t2 = t0 - q * t1; t0 = t1; t1 = t2; }
            if (!r0.is_one()) return nullptr;   // not invertible: stays unevaluated, like Mathics
            if (t0.is_negative()) t0 = t0 + m;
            return int_expr(powmod(t0, e.abs(), m));
        }
        return int_expr(powmod(n, e, m));
    }
    if (head == "ExtendedGCD" && args.size() == 2 && get_int(args[0], n) && get_int(args[1], m)) {
        B r0 = n, r1 = m, s0(1), s1(0), t0(0), t1(1);
        while (!r1.is_zero()) {
            B q = r0 / r1;
            B r2 = r0 - q * r1; r0 = r1; r1 = r2;
            B s2 = s0 - q * s1; s0 = s1; s1 = s2;
            B t2 = t0 - q * t1; t0 = t1; t1 = t2;
        }
        if (r0.is_negative()) { r0 = -r0; s0 = -s0; t0 = -t0; }
        return list_of({int_expr(r0), list_of({int_expr(s0), int_expr(t0)})});
    }
    if (head == "JacobiSymbol" && args.size() == 2 && get_int(args[0], n) && get_int(args[1], m) && m.sign() > 0 && !m.is_even()) {
        B a = n % m; if (a.is_negative()) a = a + m;
        B mm = m;
        int result = 1;
        while (!a.is_zero()) {
            while (a.is_even()) {
                a = a / B(2);
                const int64_t r = (mm % B(8)).sign() == 0 ? 0 : 0;
                (void)r;
                B m8 = mm % B(8);
                if (m8 == B(3) || m8 == B(5)) result = -result;
            }
            std::swap(a, mm);
            if ((a % B(4)) == B(3) && (mm % B(4)) == B(3)) result = -result;
            a = a % mm;
        }
        return integer(mm.is_one() ? result : 0);
    }
    if ((head == "Fibonacci" || head == "LucasL") && args.size() == 1 && get_i64(args[0], k)) {
        const uint64_t a = abs_u64(k);
        if (a > 400000) throw CasLimitError(head + ": index too large");
        B v;
        if (head == "Fibonacci") { v = fib_pair(a).first; if (k < 0 && a % 2 == 0) v = -v; }
        else { auto p = fib_pair(a); v = p.second * B(2) - p.first; if (k < 0 && a % 2 == 1) v = -v; }   // L(n) = 2F(n+1) - F(n)
        if (v.bit_length() > budget_.max_bits) throw CasLimitError(head + ": result too large");
        return int_expr(v);
    }
    if ((head == "IntegerDigits" || head == "IntegerLength") && (args.size() == 1 || args.size() == 2) && get_int(args[0], n)) {
        int64_t base = 10;
        if (args.size() == 2 && (!get_i64(args[1], base) || base < 2)) return nullptr;
        B a = n.abs();
        std::vector<Expr> digs;
        if (a.is_zero()) digs.push_back(zero());
        while (!a.is_zero()) {
            B q, r;
            B::divmod(a, B(base), q, r);
            digs.push_back(int_expr(r));
            a = q;
            tk();
        }
        if (head == "IntegerLength") return integer(static_cast<int64_t>(a.is_zero() && n.is_zero() ? 0 : digs.size()));
        std::reverse(digs.begin(), digs.end());
        return list_of(std::move(digs));
    }
    if (head == "FromDigits" && (args.size() == 1 || args.size() == 2) && args[0]->has_head("List")) {
        int64_t base = 10;
        if (args.size() == 2 && (!get_i64(args[1], base) || base < 2)) return nullptr;
        B v(0);
        for (const auto& d : args[0]->args) { if (!get_int(d, n)) return nullptr; v = v * B(base) + n; tk(); }
        return int_expr(v);
    }
    if (head == "Quotient" && args.size() == 2 && get_int(args[0], n) && get_int(args[1], m) && !m.is_zero()) {
        B q, r;
        B::divmod(n, m, q, r);
        if (!r.is_zero() && (r.is_negative() != m.is_negative())) q = q - B(1);   // floor
        return int_expr(q);
    }
    if (head == "CoprimeQ" && args.size() >= 2) {
        std::vector<B> v;
        for (const auto& a : args) { if (!get_int(a, n)) return nullptr; v.push_back(n); }
        for (size_t i = 0; i < v.size(); ++i) for (size_t j = i + 1; j < v.size(); ++j) if (!gcd_abs(v[i], v[j]).is_one()) return boolean(false);
        return boolean(true);
    }
    if (head == "Divisible" && args.size() == 2 && get_int(args[0], n) && get_int(args[1], m) && !m.is_zero()) return boolean((n % m).is_zero());
    return iteration_function(head, args);
}

// ---- Range / Table / Sum / Product / Total / Length ----
namespace {
struct Iter {
    std::string var;
    Expr lo, hi, step;
    bool counted = false;   // {n}: repeat n times, no variable
};
bool parse_iter(const Expr& spec, Iter& it) {
    if (!spec->has_head("List")) return false;
    const auto& a = spec->args;
    it.step = one();
    if (a.size() == 1) { it.counted = true; it.lo = one(); it.hi = a[0]; return true; }
    if (!a[0]->is_symbol()) return false;
    it.var = a[0]->name;
    if (a.size() == 2) { it.lo = one(); it.hi = a[1]; return true; }
    if (a.size() == 3) { it.lo = a[1]; it.hi = a[2]; return true; }
    if (a.size() == 4) { it.lo = a[1]; it.hi = a[2]; it.step = a[3]; return true; }
    return false;
}
} // namespace

Expr Engine::iteration_function(const std::string& head, const std::vector<Expr>& args) {
    const size_t cap = std::min<size_t>(budget_.max_terms * 5, 100000);
    auto count_of = [&](const Expr& lo, const Expr& hi, const Expr& step, size_t& n) -> bool {
        if (!lo->is_number() || !hi->is_number() || !step->is_number() || step->q.is_zero()) return false;
        Rational span = (hi->q - lo->q) / step->q;
        if (span.is_negative()) { n = 0; return true; }
        B fl = span.num / span.den;
        int64_t c = 0;
        if (!fl.to_int64(c) || static_cast<size_t>(c) + 1 > cap) throw CasLimitError(head + ": too many iterations");
        n = static_cast<size_t>(c) + 1;
        return true;
    };
    if (head == "Range" && args.size() >= 1 && args.size() <= 3) {
        Expr lo = args.size() == 1 ? one() : args[0];
        Expr hi = args.size() == 1 ? args[0] : args[1];
        Expr st = args.size() == 3 ? args[2] : one();
        size_t n = 0;
        if (!count_of(lo, hi, st, n)) return nullptr;
        std::vector<Expr> out;
        for (size_t i = 0; i < n; ++i) out.push_back(num(lo->q + st->q * Rational(static_cast<int64_t>(i))));
        return list_of(std::move(out));
    }
    if (head == "Length" && args.size() == 1 && args[0]->has_head("List")) return integer(static_cast<int64_t>(args[0]->args.size()));
    if (head == "Total" && args.size() == 1 && args[0]->has_head("List")) return plus(args[0]->args);

    if ((head == "Table" || head == "Sum" || head == "Product") && args.size() == 2) {
        Iter it;
        if (!parse_iter(args[1], it)) return nullptr;
        size_t n = 0;
        if (count_of(it.lo, it.hi, it.step, n)) {
            std::vector<Expr> vals;
            for (size_t i = 0; i < n; ++i) {
                tick();
                if (it.counted) { vals.push_back(args[0]); continue; }
                Expr v = num(it.lo->q + it.step->q * Rational(static_cast<int64_t>(i)));
                vals.push_back(eval(substitute(args[0], {{it.var, v}})));
            }
            if (head == "Table") return list_of(std::move(vals));
            return head == "Sum" ? plus(std::move(vals)) : times(std::move(vals));
        }
        if (head == "Table" || it.counted) return nullptr;

        // ---- symbolic bound: closed forms only where they are exact ----
        const Expr &lo = it.lo, &hi = it.hi;
        if (!it.step->is_number() || !it.step->q.is_one()) return nullptr;
        if (head == "Sum") {
            std::vector<Expr> c;
            if (poly_coefficients(args[0], it.var, c) && c.size() <= 40) {
                auto B = bernoulli_plus(static_cast<int>(c.size()) + 1);
                auto S = [&](const Expr& mexp) {   // sum_{i=1}^{m} f(i)
                    std::vector<Expr> terms;
                    for (size_t kk = 0; kk < c.size(); ++kk) {
                        if (c[kk]->is_number() && c[kk]->q.is_zero()) continue;
                        std::vector<Expr> pk;
                        BigInt binom(1);
                        for (size_t j = 0; j <= kk; ++j) {
                            Rational coef = Rational::from_bigint(binom) * B[j] / Rational(static_cast<int64_t>(kk + 1));
                            pk.push_back(times({num(coef), power(mexp, integer(static_cast<int64_t>(kk + 1 - j)))}));
                            binom = binom * BigInt(static_cast<int64_t>(kk + 1 - j)) / BigInt(static_cast<int64_t>(j + 1));
                        }
                        terms.push_back(times({c[kk], plus(std::move(pk))}));
                    }
                    return plus(std::move(terms));
                };
                Expr r = plus({S(hi), neg(S(plus({lo, minus_one()})))});
                return simplify(expand(r));
            }
            // geometric: c * r^i with r numeric and != 1
            Expr body = args[0];
            std::vector<Expr> factors = body->has_head("Times") ? body->args : std::vector<Expr>{body};
            std::vector<Expr> coef;
            Expr ratio;
            for (const auto& f : factors) {
                if (f->has_head("Power", 2) && f->args[1]->is_symbol_named(it.var.c_str()) && free_of(f->args[0], it.var) && !ratio) ratio = f->args[0];
                else if (free_of(f, it.var)) coef.push_back(f);
                else return nullptr;
            }
            if (ratio && ratio->is_number() && !ratio->q.is_one()) {
                Expr r = times({times(coef), plus({power(ratio, plus({hi, one()})), neg(power(ratio, lo))}), power(plus({ratio, minus_one()}), minus_one())});
                return simplify(r);
            }
            return nullptr;
        }
        // Product
        if (free_of(args[0], it.var)) return power(args[0], plus({hi, neg(lo), one()}));
        if (args[0]->is_symbol_named(it.var.c_str()) && lo->is_integer() && !lo->q.is_negative() && !lo->q.is_zero()) {
            Expr denom = eval(app("Factorial", {num(lo->q - Rational(1))}));
            return div(app("Factorial", {hi}), denom);
        }
    }
    return nullptr;
}

} // namespace strata::math::cas
