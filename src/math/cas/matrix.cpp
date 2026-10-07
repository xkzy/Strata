// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/cas/matrix.cpp - Det, Inverse, Transpose, Dot and exact combinatorics on top of the evaluator
#include "strata/math/cas/engine.hpp"
#include "matrix_util.hpp"

#include <algorithm>

namespace strata::math::cas {

// Determinant by cofactor expansion for symbolic entries; Gauss elimination over Q for numeric ones.
static Expr det_impl(Engine& en, const Matrix& m) {
    const size_t n = m.size();
    for (const auto& r : m) if (r.size() != n) throw CasMathError("determinant needs a square matrix");
    if (all_numbers(m)) {
        std::vector<std::vector<Rational>> a(n, std::vector<Rational>(n));
        for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) a[i][j] = m[i][j]->q;
        Rational det(1);
        for (size_t c = 0; c < n; ++c) {
            size_t p = c;
            while (p < n && a[p][c].is_zero()) ++p;
            if (p == n) return zero();
            if (p != c) { std::swap(a[p], a[c]); det = -det; }
            det = det * a[c][c];
            Rational inv = a[c][c].reciprocal();
            for (size_t r = c + 1; r < n; ++r) {
                if (a[r][c].is_zero()) continue;
                Rational f = a[r][c] * inv;
                for (size_t k = c; k < n; ++k) a[r][k] = a[r][k] - f * a[c][k];
                en.tick(n);
            }
        }
        return num(det);
    }
    if (n > 6) throw CasUnsupported("symbolic determinants are limited to 6x6");
    if (n == 1) return m[0][0];
    std::vector<Expr> terms;
    for (size_t j = 0; j < n; ++j) {
        Matrix minor;
        for (size_t i = 1; i < n; ++i) {
            std::vector<Expr> row;
            for (size_t k = 0; k < n; ++k) if (k != j) row.push_back(m[i][k]);
            minor.push_back(row);
        }
        Expr cof = det_impl(en, minor);
        terms.push_back(en.times({integer(j % 2 ? -1 : 1), m[0][j], cof}));
    }
    return en.expand(en.plus(terms));
}

Expr Engine::matrix_function(const std::string& head, const std::vector<Expr>& args) {
    if (head == "Permutations" && args.size() == 2 && args[0]->is_integer() && args[1]->is_integer()) {
        // n! / (n-k)!
        int64_t n = 0, k = 0;
        if (!args[0]->q.num.to_int64(n) || !args[1]->q.num.to_int64(k) || n < 0 || k < 0) throw CasMathError("Permutations needs non-negative integers");
        if (k > n) return zero();
        if (k > 20000) throw CasLimitError("Permutations argument too large");
        BigInt r(1);
        for (int64_t i = 0; i < k; ++i) { r = r * BigInt(n - i); tick(); if (r.bit_length() > budget_.max_bits) throw CasLimitError("result too large"); }
        return num(Rational::from_bigint(r));
    }
    if (head == "Multinomial" && !args.empty()) {
        BigInt total, denom(1);
        BigInt numer(1);
        int64_t sum = 0;
        for (const auto& a : args) {
            int64_t k = 0;
            if (!a->is_integer() || !a->q.num.to_int64(k) || k < 0) throw CasMathError("Multinomial needs non-negative integers");
            sum += k;
        }
        if (sum > 20000) throw CasLimitError("Multinomial argument too large");
        for (int64_t i = 2; i <= sum; ++i) numer = numer * BigInt(i);
        for (const auto& a : args) { int64_t k = 0; a->q.num.to_int64(k); for (int64_t i = 2; i <= k; ++i) denom = denom * BigInt(i); }
        return num(Rational::from_bigint(numer / denom));
    }
    if ((head == "GCD" || head == "LCM") && args.size() >= 1) {
        BigInt acc;
        bool first = true;
        for (const auto& a : args) {
            if (!a->is_integer()) return nullptr;
            BigInt v = a->q.num.abs();
            if (first) { acc = v; first = false; }
            else if (head == "GCD") acc = BigInt::gcd(acc, v);
            else acc = acc.is_zero() || v.is_zero() ? BigInt() : acc / BigInt::gcd(acc, v) * v;
        }
        return num(Rational::from_bigint(acc));
    }
    if (head == "Mod" && args.size() == 2 && args[0]->is_integer() && args[1]->is_integer() && !args[1]->q.is_zero()) {
        BigInt q, r;
        BigInt::divmod(args[0]->q.num, args[1]->q.num, q, r);
        if (!r.is_zero() && (r.is_negative() != args[1]->q.num.is_negative())) r = r + args[1]->q.num;
        return num(Rational::from_bigint(r));
    }
    Matrix m;
    if (head == "Det" && args.size() == 1 && as_matrix(args[0], m)) return det_impl(*this, m);
    if (head == "Transpose" && args.size() == 1 && as_matrix(args[0], m)) {
        Matrix t(m[0].size(), std::vector<Expr>(m.size()));
        for (size_t i = 0; i < m.size(); ++i) for (size_t j = 0; j < m[0].size(); ++j) t[j][i] = m[i][j];
        return from_matrix(t);
    }
    if (head == "Inverse" && args.size() == 1 && as_matrix(args[0], m)) {
        const size_t n = m.size();
        if (m[0].size() != n) throw CasMathError("only square matrices can be inverted");
        Expr d = det_impl(*this, m);
        if (d->is_number() && d->q.is_zero()) throw CasMathError("the matrix is singular");
        if (all_numbers(m)) {
            std::vector<std::vector<Rational>> a(n, std::vector<Rational>(2 * n, Rational(0)));
            for (size_t i = 0; i < n; ++i) { for (size_t j = 0; j < n; ++j) a[i][j] = m[i][j]->q; a[i][n + i] = Rational(1); }
            for (size_t c = 0; c < n; ++c) {
                size_t p = c;
                while (a[p][c].is_zero()) ++p;
                std::swap(a[p], a[c]);
                Rational inv = a[c][c].reciprocal();
                for (auto& v : a[c]) v = v * inv;
                for (size_t r = 0; r < n; ++r) {
                    if (r == c || a[r][c].is_zero()) continue;
                    Rational f = a[r][c];
                    for (size_t k = 0; k < 2 * n; ++k) a[r][k] = a[r][k] - f * a[c][k];
                    tick(n);
                }
            }
            Matrix inv(n, std::vector<Expr>(n));
            for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) inv[i][j] = num(a[i][n + j]);
            return from_matrix(inv);
        }
        if (n > 4) throw CasUnsupported("symbolic inverses are limited to 4x4");
        // adjugate / det
        Matrix inv(n, std::vector<Expr>(n));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                Matrix minor;
                for (size_t r = 0; r < n; ++r) {
                    if (r == i) continue;
                    std::vector<Expr> row;
                    for (size_t c = 0; c < n; ++c) if (c != j) row.push_back(m[r][c]);
                    minor.push_back(row);
                }
                Expr cof = n == 1 ? one() : det_impl(*this, minor);
                if ((i + j) % 2) cof = neg(cof);
                inv[j][i] = simplify(div(cof, d));
            }
        }
        return from_matrix(inv);
    }
    if (head == "Dot" && args.size() == 2) {
        Matrix a, b;
        bool am = as_matrix(args[0], a), bm = as_matrix(args[1], b);
        if (am && bm) {
            if (a[0].size() != b.size()) throw CasMathError("matrix dimensions do not match");
            Matrix r(a.size(), std::vector<Expr>(b[0].size()));
            for (size_t i = 0; i < a.size(); ++i)
                for (size_t j = 0; j < b[0].size(); ++j) {
                    std::vector<Expr> t;
                    for (size_t k = 0; k < b.size(); ++k) t.push_back(times({a[i][k], b[k][j]}));
                    r[i][j] = expand(plus(t));
                }
            return from_matrix(r);
        }
        // vector . vector
        if (args[0]->has_head("List") && args[1]->has_head("List") && args[0]->args.size() == args[1]->args.size() && !am && !bm) {
            std::vector<Expr> t;
            for (size_t k = 0; k < args[0]->args.size(); ++k) t.push_back(times({args[0]->args[k], args[1]->args[k]}));
            return expand(plus(t));
        }
        // matrix . vector
        if (am && args[1]->has_head("List") && a[0].size() == args[1]->args.size()) {
            std::vector<Expr> out;
            for (size_t i = 0; i < a.size(); ++i) {
                std::vector<Expr> t;
                for (size_t k = 0; k < a[i].size(); ++k) t.push_back(times({a[i][k], args[1]->args[k]}));
                out.push_back(expand(plus(t)));
            }
            return app("List", out);
        }
        throw CasMathError("Dot: incompatible arguments");
    }
    return nullptr;
}

} // namespace strata::math::cas
