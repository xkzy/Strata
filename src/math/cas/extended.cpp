// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/cas/extended.cpp - linear algebra beyond Det/Inverse, vector calculus, partial fractions, inequalities,
// modular arithmetic and partitions (clean-room, exact; behaviour follows Mathics / Sage)
//
// Everything is exact: rational input gives rational output, or a CasError / CasUnsupported - never a guess.
#include "strata/math/cas/engine.hpp"
#include "matrix_util.hpp"

#include <algorithm>
#include <cmath>

namespace strata::math::cas {

namespace {

using RMatrix = std::vector<std::vector<Rational>>;

RMatrix to_rational(const Matrix& m) {
    RMatrix r(m.size(), std::vector<Rational>(m[0].size()));
    for (size_t i = 0; i < m.size(); ++i)
        for (size_t j = 0; j < m[i].size(); ++j) r[i][j] = m[i][j]->q;
    return r;
}

Expr from_rational(const RMatrix& r) {
    Matrix m(r.size(), std::vector<Expr>(r.empty() ? 0 : r[0].size()));
    for (size_t i = 0; i < r.size(); ++i) for (size_t j = 0; j < r[i].size(); ++j) m[i][j] = num(r[i][j]);
    return from_matrix(m);
}

// Reduced row echelon form over Q; returns the pivot columns.
std::vector<size_t> rref(RMatrix& a, Engine& en) {
    std::vector<size_t> pivots;
    const size_t rows = a.size(), cols = a[0].size();
    size_t r = 0;
    for (size_t c = 0; c < cols && r < rows; ++c) {
        size_t p = r;
        while (p < rows && a[p][c].is_zero()) ++p;
        if (p == rows) continue;
        std::swap(a[p], a[r]);
        const Rational inv = a[r][c].reciprocal();
        for (auto& v : a[r]) v = v * inv;
        for (size_t i = 0; i < rows; ++i) {
            if (i == r || a[i][c].is_zero()) continue;
            const Rational f = a[i][c];
            for (size_t k = 0; k < cols; ++k) a[i][k] = a[i][k] - f * a[r][k];
            en.tick(cols);
        }
        pivots.push_back(c);
        ++r;
    }
    return pivots;
}

std::vector<std::vector<Rational>> nullspace(RMatrix a, Engine& en) {
    const size_t cols = a[0].size();
    const std::vector<size_t> piv = rref(a, en);
    std::vector<bool> is_pivot(cols, false);
    for (size_t c : piv) is_pivot[c] = true;
    std::vector<std::vector<Rational>> basis;
    for (size_t f = 0; f < cols; ++f) {
        if (is_pivot[f]) continue;
        std::vector<Rational> v(cols, Rational(0));
        v[f] = Rational(1);
        for (size_t k = 0; k < piv.size(); ++k) v[piv[k]] = -a[k][f];
        basis.push_back(v);
    }
    return basis;
}

bool contains_symbol(const Expr& e, const char* name) {
    if (e->is_symbol()) return e->name == name;
    for (const auto& a : e->args) if (contains_symbol(a, name)) return true;
    return false;
}

// extended Euclid on BigInt: g = a x + b y
void egcd(const BigInt& a, const BigInt& b, BigInt& g, BigInt& x, BigInt& y) {
    BigInt r0 = a, r1 = b, s0(1), s1(0), t0(0), t1(1);
    while (!r1.is_zero()) {
        BigInt q = r0 / r1;   // truncated division is fine here: the invariants hold for any quotient
        BigInt r2 = r0 - q * r1, s2 = s0 - q * s1, t2 = t0 - q * t1;
        r0 = r1; r1 = r2; s0 = s1; s1 = s2; t0 = t1; t1 = t2;
    }
    g = r0; x = s0; y = t0;
    if (g.is_negative()) { g = -g; x = -x; y = -y; }
}

BigInt floor_mod(const BigInt& a, const BigInt& m) {
    BigInt r = a % m;
    if (r.is_negative()) r = r + m;
    return r;
}

using u64 = uint64_t;
using u128 = unsigned __int128;
u64 mulmod(u64 a, u64 b, u64 m) { return static_cast<u64>(static_cast<u128>(a) * b % m); }
u64 powmod(u64 a, u64 e, u64 m) { u64 r = 1 % m; a %= m; while (e) { if (e & 1) r = mulmod(r, a, m); a = mulmod(a, a, m); e >>= 1; } return r; }
u64 gcd64(u64 a, u64 b) { while (b) { u64 t = a % b; a = b; b = t; } return a; }

bool to_u64(const Expr& e, u64& out) {
    int64_t v = 0;
    if (!e->is_integer() || !e->q.num.to_int64(v) || v < 0) return false;
    out = static_cast<u64>(v);
    return true;
}

// distinct prime factors of n by trial division (n < 2^62; bounded work)
bool prime_factors(u64 n, std::vector<u64>& out) {
    for (u64 d = 2; d * d <= n && d < 4'000'000; ++d) {
        if (n % d == 0) { out.push_back(d); while (n % d == 0) n /= d; }
    }
    if (n > 1) {
        // n has no factor below 4e6: prime if n < 1.6e13, otherwise we cannot tell by trial division
        if (n >= 16'000'000'000'000ULL) return false;
        out.push_back(n);
    }
    return true;
}

// Euler phi from the factorization
u64 phi_of(u64 n, const std::vector<u64>& primes) { u64 r = n; for (u64 p : primes) r = r / p * (p - 1); return r; }

Rational rational_from_double(double v) {
    const int64_t scaled = static_cast<int64_t>(std::llround(v * 1073741824.0));   // k / 2^30
    return Rational::from_bigint(BigInt(scaled)) / Rational::from_bigint(BigInt(1073741824));
}

} // namespace

Expr Engine::extended_math(const std::string& head, const std::vector<Expr>& args) {
    Matrix m;
    // size guard for everything below that eliminates or factors: exact O(n^3) work on a matrix or polynomial this large is refused up front
    if (!args.empty() && args[0]->has_head("List") && args[0]->args.size() > 200) throw CasLimitError(head + ": input larger than 200 rows / elements");

    // ---------------------------------------------------------------- comparisons of numbers
    if ((head == "Less" || head == "Greater" || head == "LessEqual" || head == "GreaterEqual") && args.size() == 2 && args[0]->is_number() && args[1]->is_number()) {
        const Rational& a = args[0]->q;
        const Rational& b = args[1]->q;
        const bool lt = a < b, eq = a == b;
        const bool r = head == "Less" ? lt : head == "Greater" ? (!lt && !eq) : head == "LessEqual" ? (lt || eq) : !lt;
        return symbol(r ? "True" : "False");
    }
    if (head == "Unequal" && args.size() == 2 && args[0]->is_number() && args[1]->is_number()) return symbol(args[0]->q == args[1]->q ? "False" : "True");

    // ---------------------------------------------------------------- matrices
    if (head == "IdentityMatrix" && args.size() == 1 && args[0]->is_integer()) {
        int64_t n = 0;
        if (!args[0]->q.num.to_int64(n) || n < 1 || n > 512) throw CasMathError("IdentityMatrix needs a size between 1 and 512");
        RMatrix r(n, std::vector<Rational>(n, Rational(0)));
        for (int64_t i = 0; i < n; ++i) r[i][i] = Rational(1);
        return from_rational(r);
    }
    if ((head == "Tr" || head == "Trace") && args.size() == 1 && as_matrix(args[0], m)) {
        if (m.size() != m[0].size()) throw CasMathError("the trace needs a square matrix");
        std::vector<Expr> d;
        for (size_t i = 0; i < m.size(); ++i) d.push_back(m[i][i]);
        return expand(plus(d));
    }
    if ((head == "Rank" || head == "MatrixRank") && args.size() == 1 && as_matrix(args[0], m)) {
        if (!all_numbers(m)) throw CasUnsupported("the rank of a symbolic matrix is not implemented");
        RMatrix a = to_rational(m);
        return integer(static_cast<int64_t>(rref(a, *this).size()));
    }
    if (head == "NullSpace" && args.size() == 1 && as_matrix(args[0], m)) {
        if (!all_numbers(m)) throw CasUnsupported("the null space of a symbolic matrix is not implemented");
        const auto basis = nullspace(to_rational(m), *this);
        std::vector<Expr> out;
        for (const auto& v : basis) { std::vector<Expr> row; for (const auto& x : v) row.push_back(num(x)); out.push_back(app("List", row)); }
        return app("List", out);
    }
    if ((head == "CharacteristicPolynomial" || head == "CharPoly") && args.size() == 2 && as_matrix(args[0], m) && args[1]->is_symbol()) {
        if (m.size() != m[0].size()) throw CasMathError("the characteristic polynomial needs a square matrix");
        Matrix b = m;
        for (size_t i = 0; i < b.size(); ++i) b[i][i] = plus({m[i][i], neg(args[1])});   // M - x I
        return expand(matrix_function("Det", {from_matrix(b)}));
    }
    if ((head == "Eigenvalues" || head == "Eigenvectors") && args.size() == 1 && as_matrix(args[0], m)) {
        if (m.size() != m[0].size()) throw CasMathError("eigenvalues need a square matrix");
        const std::string x = "$lambda";
        Expr cp = extended_math("CharacteristicPolynomial", {args[0], symbol(x)});
        Expr sol = solve(app("Equal", {cp, zero()}), {x});
        std::vector<Expr> values;
        for (const auto& set : sol->args) {
            if (!set->has_head("List") || set->args.size() != 1 || !set->args[0]->has_head("Rule", 2)) throw CasUnsupported("the eigenvalues are not in a simple form");
            values.push_back(set->args[0]->args[1]);
        }
        // algebraic multiplicity: how many derivatives of the characteristic polynomial vanish at the root
        std::vector<Expr> all, vectors;
        std::vector<std::pair<Expr, std::vector<Expr>>> by_value;   // eigenvalue -> its eigenvectors
        size_t found = 0;
        for (const auto& r : values) {
            int mult = 0;
            Expr d = cp;
            while (mult <= static_cast<int>(m.size())) {
                Expr at = eval(substitute(d, {{x, r}}));
                Expr z = expand(at);
                if (!(z->is_number() && z->q.is_zero())) { if (simplify(z)->is_number() && simplify(z)->q.is_zero()) { } else break; }
                ++mult;
                d = diff(d, x);
            }
            if (mult == 0) throw CasUnsupported("could not determine the multiplicity of an eigenvalue");
            found += mult;
            for (int k = 0; k < mult; ++k) all.push_back(r);
            by_value.push_back({r, {}});
            if (head == "Eigenvectors") {
                Matrix b = m;
                for (size_t i = 0; i < b.size(); ++i) b[i][i] = plus({m[i][i], neg(r)});
                Matrix be = b;
                for (auto& row : be) for (auto& e : row) e = eval(e);
                if (!all_numbers(be)) throw CasUnsupported("eigenvectors for irrational eigenvalues are not implemented");
                const auto basis = nullspace(to_rational(be), *this);
                for (const auto& v : basis) { std::vector<Expr> row; for (const auto& c : v) row.push_back(num(c)); by_value.back().second.push_back(app("List", row)); }
            }
        }
        if (found != m.size()) throw CasUnsupported("some eigenvalues are complex or not expressible exactly");
        std::sort(all.begin(), all.end(), [&](const Expr& a, const Expr& b) { double da = 0, db = 0; numeric_value(a, da); numeric_value(b, db); return da > db; });
        std::sort(by_value.begin(), by_value.end(), [&](const auto& a, const auto& b) { double da = 0, db = 0; numeric_value(a.first, da); numeric_value(b.first, db); return da > db; });
        for (const auto& bv : by_value) for (const auto& v : bv.second) vectors.push_back(v);   // same order as the eigenvalues
        return head == "Eigenvalues" ? app("List", all) : app("List", vectors);
    }
    if (head == "LU" && args.size() == 1 && as_matrix(args[0], m)) {   // Sage's convention: P, L, U with P A = L U
        if (!all_numbers(m) || m.size() != m[0].size()) throw CasUnsupported("LU needs a square matrix of exact numbers");
        const size_t n = m.size();
        RMatrix u = to_rational(m), l(n, std::vector<Rational>(n, Rational(0)));
        std::vector<size_t> perm(n);
        for (size_t i = 0; i < n; ++i) perm[i] = i;
        for (size_t c = 0; c < n; ++c) {
            size_t p = c;
            while (p < n && u[p][c].is_zero()) ++p;
            if (p == n) continue;
            if (p != c) { std::swap(u[p], u[c]); std::swap(perm[p], perm[c]); for (size_t k = 0; k < c; ++k) std::swap(l[p][k], l[c][k]); }
            for (size_t r = c + 1; r < n; ++r) {
                if (u[r][c].is_zero()) continue;
                const Rational f = u[r][c] / u[c][c];
                l[r][c] = f;
                for (size_t k = c; k < n; ++k) u[r][k] = u[r][k] - f * u[c][k];
                tick(n);
            }
        }
        for (size_t i = 0; i < n; ++i) l[i][i] = Rational(1);
        RMatrix pm(n, std::vector<Rational>(n, Rational(0)));
        for (size_t i = 0; i < n; ++i) pm[i][perm[i]] = Rational(1);
        return app("List", {from_rational(pm), from_rational(l), from_rational(u)});
    }
    if ((head == "LinearSolve" || head == "MatrixSolve") && args.size() == 2 && as_matrix(args[0], m)) {
        if (!all_numbers(m)) throw CasUnsupported("LinearSolve for symbolic matrices is not implemented");
        const size_t rows = m.size(), cols = m[0].size();
        Matrix bm;
        bool is_mat_b = as_matrix(args[1], bm);
        bool is_vec_b = args[1]->has_head("List") && !is_mat_b;
        if (!is_mat_b && !is_vec_b) throw CasMathError("LinearSolve: second argument must be a vector or matrix");
        if (is_vec_b && args[1]->args.size() != rows) throw CasMathError("LinearSolve: vector dimension must match matrix rows");
        if (is_mat_b && (bm.size() != rows || !all_numbers(bm))) throw CasMathError("LinearSolve: matrix dimension must match and contain numbers");
        const size_t bcols = is_mat_b ? bm[0].size() : 1;
        RMatrix aug(rows, std::vector<Rational>(cols + bcols, Rational(0)));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) aug[i][j] = m[i][j]->q;
            if (is_vec_b) {
                if (!args[1]->args[i]->is_number()) throw CasUnsupported("LinearSolve: numerical vector entries required");
                aug[i][cols] = args[1]->args[i]->q;
            } else {
                for (size_t k = 0; k < bcols; ++k) aug[i][cols + k] = bm[i][k]->q;
            }
        }
        rref(aug, *this);
        // Check for inconsistency
        for (size_t i = 0; i < rows; ++i) {
            bool all_zero = true;
            for (size_t j = 0; j < cols; ++j) if (!aug[i][j].is_zero()) { all_zero = false; break; }
            if (all_zero) {
                for (size_t k = 0; k < bcols; ++k) {
                    if (!aug[i][cols + k].is_zero()) throw CasMathError("LinearSolve: system has no solution");
                }
            }
        }
        if (rows < cols) throw CasUnsupported("LinearSolve: underdetermined system");
        if (is_vec_b) {
            std::vector<Expr> sol(cols, zero());
            for (size_t i = 0; i < cols; ++i) {
                if (i >= rows || aug[i][i].is_zero()) throw CasMathError("LinearSolve: matrix is singular");
                sol[i] = num(aug[i][cols]);
            }
            return app("List", sol);
        } else {
            Matrix sol(cols, std::vector<Expr>(bcols, zero()));
            for (size_t i = 0; i < cols; ++i) {
                if (i >= rows || aug[i][i].is_zero()) throw CasMathError("LinearSolve: matrix is singular");
                for (size_t k = 0; k < bcols; ++k) sol[i][k] = num(aug[i][cols + k]);
            }
            return from_matrix(sol);
        }
    }
    if ((head == "QR" || head == "QRDecomposition") && args.size() == 1 && as_matrix(args[0], m)) {
        if (!all_numbers(m)) throw CasUnsupported("QR decomposition needs numerical matrix entries");
        const size_t rows = m.size(), cols = m[0].size();
        if (rows < cols) throw CasMathError("QR decomposition requires rows >= cols");
        // Gram-Schmidt orthogonalization
        Matrix q(rows, std::vector<Expr>(cols, zero()));
        Matrix r(cols, std::vector<Expr>(cols, zero()));
        for (size_t j = 0; j < cols; ++j) {
            std::vector<Expr> v(rows);
            for (size_t i = 0; i < rows; ++i) v[i] = m[i][j];
            for (size_t k = 0; k < j; ++k) {
                std::vector<Expr> dot_terms;
                for (size_t i = 0; i < rows; ++i) dot_terms.push_back(times({m[i][j], q[i][k]}));
                Expr r_kj = eval(plus(dot_terms));
                r[k][j] = r_kj;
                for (size_t i = 0; i < rows; ++i) v[i] = eval(sub(v[i], times({r_kj, q[i][k]})));
            }
            std::vector<Expr> norm_terms;
            for (size_t i = 0; i < rows; ++i) norm_terms.push_back(power(v[i], integer(2)));
            Expr norm_sq = eval(plus(norm_terms));
            if (norm_sq->is_number() && norm_sq->q.is_zero()) throw CasMathError("QR decomposition: linearly dependent columns");
            Expr norm = eval(power(norm_sq, num(Rational(1) / Rational(2))));
            r[j][j] = norm;
            for (size_t i = 0; i < rows; ++i) q[i][j] = eval(div(v[i], norm));
            tick(rows);
        }
        return app("List", {from_matrix(q), from_matrix(r)});
    }
    if ((head == "Cholesky" || head == "CholeskyDecomposition") && args.size() == 1 && as_matrix(args[0], m)) {
        if (!all_numbers(m) || m.size() != m[0].size()) throw CasMathError("Cholesky decomposition requires a square numerical matrix");
        const size_t n = m.size();
        // Check symmetry
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                if (m[i][j]->q != m[j][i]->q) throw CasMathError("Cholesky decomposition: matrix must be symmetric");
            }
        }
        Matrix l(n, std::vector<Expr>(n, zero()));
        for (size_t j = 0; j < n; ++j) {
            std::vector<Expr> sum_diag;
            for (size_t k = 0; k < j; ++k) sum_diag.push_back(power(l[j][k], integer(2)));
            Expr diag_diff = eval(sub(m[j][j], plus(sum_diag)));
            double d_val = 0;
            if (!numeric_value(diag_diff, d_val) || d_val <= 0.0) throw CasMathError("Cholesky decomposition: matrix is not positive-definite");
            l[j][j] = eval(power(diag_diff, num(Rational(1) / Rational(2))));
            for (size_t i = j + 1; i < n; ++i) {
                std::vector<Expr> sum_ij;
                for (size_t k = 0; k < j; ++k) sum_ij.push_back(times({l[i][k], l[j][k]}));
                Expr off_diff = eval(sub(m[i][j], plus(sum_ij)));
                l[i][j] = eval(div(off_diff, l[j][j]));
            }
            tick(n);
        }
        return from_matrix(l);
    }

    // ---------------------------------------------------------------- vector calculus
    if ((head == "Grad" || head == "Gradient") && args.size() == 2 && args[1]->has_head("List")) {
        std::vector<Expr> g;
        for (const auto& v : args[1]->args) { if (!v->is_symbol()) throw CasMathError("Grad: the variables must be symbols"); g.push_back(diff(args[0], v->name)); }
        return app("List", g);
    }
    if (head == "Jacobian" && args.size() == 2 && args[0]->has_head("List") && args[1]->has_head("List")) {
        Matrix j;
        for (const auto& f : args[0]->args) {
            std::vector<Expr> row;
            for (const auto& v : args[1]->args) { if (!v->is_symbol()) throw CasMathError("Jacobian: the variables must be symbols"); row.push_back(diff(f, v->name)); }
            j.push_back(row);
        }
        if (j.empty() || j[0].empty()) throw CasMathError("Jacobian needs at least one function and one variable");
        return from_matrix(j);
    }
    if (head == "Hessian" && args.size() == 2 && args[1]->has_head("List")) {
        Matrix h;
        for (const auto& a : args[1]->args) {
            std::vector<Expr> row;
            if (!a->is_symbol()) throw CasMathError("Hessian: the variables must be symbols");
            Expr da = diff(args[0], a->name);
            for (const auto& b : args[1]->args) { if (!b->is_symbol()) throw CasMathError("Hessian: the variables must be symbols"); row.push_back(diff(da, b->name)); }
            h.push_back(row);
        }
        if (h.empty()) throw CasMathError("Hessian needs at least one variable");
        return from_matrix(h);
    }

    // ---------------------------------------------------------------- polynomial forms
    if (head == "Collect" && args.size() == 2 && args[1]->is_symbol()) {
        std::vector<Expr> coeffs;
        if (!poly_coefficients(args[0], args[1]->name, coeffs)) throw CasUnsupported("Collect needs a polynomial in the variable");
        std::vector<Expr> terms;
        for (size_t k = coeffs.size(); k-- > 0;) {
            Expr c = expand(coeffs[k]);
            if (c->is_number() && c->q.is_zero()) continue;
            terms.push_back(k == 0 ? c : times({c, k == 1 ? args[1] : power(args[1], integer(static_cast<int64_t>(k)))}));
        }
        return terms.empty() ? zero() : terms.size() == 1 ? terms[0] : app("Plus", terms);
    }
    if (head == "Apart" && args.size() >= 1) {
        Expr f = args[0];
        std::string var = args.size() == 2 && args[1]->is_symbol() ? args[1]->name : "";
        if (var.empty()) { std::vector<std::string> s; free_symbols(f, s); if (s.size() != 1) throw CasUnsupported("Apart needs one variable"); var = s[0]; }
        Poly n, d;
        if (!rational_function(f, var, n, d)) throw CasUnsupported("Apart needs a rational function of one variable");
        const TickFn tk = [this](uint64_t k) { tick(k); };
        Poly q, r;
        poly_divmod(n, d, q, r);
        std::vector<Expr> terms;
        if (!q.empty()) terms.push_back(poly_to_expr(q, var));
        if (!r.empty()) {
            Factorization fz = factor_over_q(d, tk);
            // D = product of h_i^m_i (no content); unknowns: for each factor power h^j a numerator of degree < deg h
            Poly D{Rational(1)};
            for (const auto& [h, mult] : fz.factors) for (int k = 0; k < mult; ++k) D = poly_mul(D, h);
            struct Unknown { size_t factor; int j; int k; Poly basis; };
            std::vector<Unknown> unk;
            for (size_t i = 0; i < fz.factors.size(); ++i) {
                const auto& [h, mult] = fz.factors[i];
                for (int j = 1; j <= mult; ++j) {
                    Poly other{Rational(1)};   // D / h^j
                    for (size_t t = 0; t < fz.factors.size(); ++t) {
                        const int e = t == i ? mult - j : fz.factors[t].second;
                        for (int k = 0; k < e; ++k) other = poly_mul(other, fz.factors[t].first);
                    }
                    for (int k = 0; k < poly_degree(h); ++k) {
                        Poly xk(static_cast<size_t>(k) + 1, Rational(0));
                        xk[k] = Rational(1);
                        unk.push_back({i, j, k, poly_mul(xk, other)});
                    }
                }
            }
            const size_t nunk = unk.size();
            if (static_cast<int>(nunk) != poly_degree(D)) throw CasUnsupported("partial fractions: unexpected denominator structure");
            RMatrix a(nunk, std::vector<Rational>(nunk + 1, Rational(0)));
            const Rational inv_content = fz.content.reciprocal();
            for (size_t row = 0; row < nunk; ++row) {
                for (size_t col = 0; col < nunk; ++col) a[row][col] = row < unk[col].basis.size() ? unk[col].basis[row] : Rational(0);
                a[row][nunk] = row < r.size() ? r[row] * inv_content : Rational(0);
            }
            const auto piv = rref(a, *this);
            if (piv.size() != nunk) throw CasUnsupported("partial fractions: singular system");
            std::vector<Poly> numer(unk.size());   // by (factor, j) accumulate
            std::vector<std::vector<Poly>> parts(fz.factors.size());
            for (size_t i = 0; i < fz.factors.size(); ++i) parts[i].assign(fz.factors[i].second + 1, Poly());
            for (size_t u = 0; u < nunk; ++u) {
                Poly& p = parts[unk[u].factor][unk[u].j];
                if (p.size() <= static_cast<size_t>(unk[u].k)) p.resize(unk[u].k + 1, Rational(0));
                p[unk[u].k] = a[u][nunk];
            }
            for (size_t i = 0; i < fz.factors.size(); ++i)
                for (int j = 1; j <= fz.factors[i].second; ++j) {
                    Poly p = parts[i][j];
                    poly_trim(p);
                    if (p.empty()) continue;
                    Expr hb = poly_to_expr(fz.factors[i].first, var);
                    terms.push_back(div(poly_to_expr(p, var), j == 1 ? hb : power(hb, integer(j))));
                }
        }
        return terms.empty() ? zero() : terms.size() == 1 ? terms[0] : app("Plus", terms);
    }

    // ---------------------------------------------------------------- inequalities in one variable
    if ((head == "Reduce" || head == "Solve" ) && args.size() >= 1 &&
        (args[0]->has_head("Less", 2) || args[0]->has_head("Greater", 2) || args[0]->has_head("LessEqual", 2) || args[0]->has_head("GreaterEqual", 2))) {
        const Expr& rel = args[0];
        std::string var = args.size() >= 2 && args[1]->is_symbol() ? args[1]->name : "";
        Expr f = eval(sub(rel->args[0], rel->args[1]));
        if (var.empty()) { std::vector<std::string> s; free_symbols(f, s); if (s.size() != 1) throw CasUnsupported("an inequality needs exactly one variable"); var = s[0]; }
        const bool greater = rel->has_head("Greater", 2) || rel->has_head("GreaterEqual", 2);
        const bool strict = rel->has_head("Less", 2) || rel->has_head("Greater", 2);
        Poly n, d;
        if (!rational_function(f, var, n, d)) throw CasUnsupported("inequalities are supported for rational functions of one variable");
        if (n.empty()) return symbol(strict ? "False" : "True");
        // real roots of numerator and denominator
        auto real_roots = [&](const Poly& p, std::vector<std::pair<double, Expr>>& out) {
            if (poly_degree(p) < 1) return;
            Expr sol = solve(app("Equal", {poly_to_expr(p, var), zero()}), {var});
            for (const auto& set : sol->args) {
                if (!set->has_head("List") || set->args.size() != 1 || !set->args[0]->has_head("Rule", 2)) throw CasUnsupported("roots not in a simple form");
                const Expr& v = set->args[0]->args[1];
                if (contains_symbol(v, "I")) continue;   // complex root: never a boundary of a real interval
                double dv = 0;
                if (!numeric_value(v, dv)) throw CasUnsupported("could not locate a root numerically");
                out.push_back({dv, v});
            }
        };
        std::vector<std::pair<double, Expr>> nr, dr;
        real_roots(n, nr);
        real_roots(d, dr);
        // the solver may omit roots it cannot express; check the count against the degree when everything is real and simple
        std::vector<std::pair<double, Expr>> pts;   // (value, expr) with kind in parallel vector
        std::vector<bool> is_pole;
        for (auto& r : nr) { pts.push_back(r); is_pole.push_back(false); }
        for (auto& r : dr) { pts.push_back(r); is_pole.push_back(true); }
        std::vector<size_t> order(pts.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return pts[a].first < pts[b].first; });
        // merge numerically equal points (a root shared by numerator and denominator is a pole of nothing: it cancels in rational_function)
        struct Pt { double v; Expr e; bool pole; };
        std::vector<Pt> P;
        for (size_t i : order) {
            if (!P.empty() && std::fabs(P.back().v - pts[i].first) < 1e-12 * (1 + std::fabs(pts[i].first))) { P.back().pole = P.back().pole || is_pole[i]; continue; }
            P.push_back({pts[i].first, pts[i].second, is_pole[i]});
        }
        auto sign_at = [&](double s) -> int {
            const Rational q = rational_from_double(s);
            const Rational nv = poly_eval(n, q), dv = poly_eval(d, q);
            if (dv.is_zero()) throw CasUnsupported("sample point hit a pole");
            const bool neg = nv.is_negative() != dv.is_negative();
            return nv.is_zero() ? 0 : neg ? -1 : 1;
        };
        const int want = greater ? 1 : -1;
        struct Seg { bool lo_inf, hi_inf; Expr lo, hi; bool lo_closed, hi_closed; };
        std::vector<Seg> segs;
        const size_t k = P.size();
        for (size_t i = 0; i <= k; ++i) {   // open interval i: (P[i-1], P[i])
            double lo = i == 0 ? 0 : P[i - 1].v, hi = i == k ? 0 : P[i].v;
            double sample = i == 0 ? (k ? std::floor(hi) - 1 : 0) : i == k ? std::ceil(lo) + 1 : (lo + hi) / 2;
            if (i > 0 && i < k && !(sample > lo && sample < hi)) throw CasUnsupported("roots too close to separate");
            if (i == 0 && k && !(sample < hi)) sample = hi - 1;
            if (i == k && k && !(sample > lo)) sample = lo + 1;
            if (sign_at(sample) == want) segs.push_back({i == 0, i == k, i == 0 ? nullptr : P[i - 1].e, i == k ? nullptr : P[i].e, false, false});
        }
        if (!strict) {   // zeros of the numerator that are not poles belong to the solution set of <= / >=
            for (size_t i = 0; i < k; ++i) {
                if (P[i].pole) continue;
                bool absorbed = false;
                for (auto& s : segs) {
                    if (!s.lo_inf && s.lo && equal(s.lo, P[i].e)) { s.lo_closed = true; absorbed = true; }
                    if (!s.hi_inf && s.hi && equal(s.hi, P[i].e)) { s.hi_closed = true; absorbed = true; }
                }
                if (!absorbed) segs.push_back({false, false, P[i].e, P[i].e, true, true});
            }
        }
        // merge neighbouring segments that touch at a closed or an included point
        std::vector<Expr> clauses;
        const Expr x = symbol(var);
        for (const auto& s : segs) {
            if (!s.lo_inf && !s.hi_inf && s.lo && s.hi && equal(s.lo, s.hi)) { clauses.push_back(app("Equal", {x, s.lo})); continue; }
            std::vector<Expr> parts;
            if (!s.lo_inf) parts.push_back(app(s.lo_closed ? "GreaterEqual" : "Greater", {x, s.lo}));
            if (!s.hi_inf) parts.push_back(app(s.hi_closed ? "LessEqual" : "Less", {x, s.hi}));
            clauses.push_back(parts.empty() ? symbol("True") : parts.size() == 1 ? parts[0] : app("And", parts));
        }
        if (clauses.empty()) return symbol("False");
        return clauses.size() == 1 ? clauses[0] : app("Or", clauses);
    }

    // ---------------------------------------------------------------- modular arithmetic and partitions
    if ((head == "ChineseRemainder" || head == "CRT") && args.size() == 2 && args[0]->has_head("List") && args[1]->has_head("List") &&
        args[0]->args.size() == args[1]->args.size() && !args[0]->args.empty()) {
        BigInt a, mod;
        bool first = true;
        for (size_t i = 0; i < args[0]->args.size(); ++i) {
            if (!args[0]->args[i]->is_integer() || !args[1]->args[i]->is_integer()) throw CasMathError("ChineseRemainder needs integers");
            BigInt ai = args[0]->args[i]->q.num, mi = args[1]->args[i]->q.num.abs();
            if (mi.is_zero()) throw CasMathError("ChineseRemainder: a modulus is zero");
            ai = floor_mod(ai, mi);
            if (first) { a = ai; mod = mi; first = false; continue; }
            BigInt g, x, y;
            egcd(mod, mi, g, x, y);
            const BigInt diff_v = ai - a;
            if (!(diff_v % g).is_zero()) throw CasMathError("the congruences have no common solution");
            const BigInt lcm = mod / g * mi;
            BigInt t = floor_mod((diff_v / g) * x, mi / g);
            a = floor_mod(a + mod * t, lcm);
            mod = lcm;
            tick();
            if (mod.bit_length() > budget_.max_bits) throw CasLimitError("result too large");
        }
        return num(Rational::from_bigint(a));
    }
    if ((head == "ModularInverse" || head == "ModInverse") && args.size() == 2 && args[0]->is_integer() && args[1]->is_integer()) {
        const BigInt mi = args[1]->q.num.abs();
        if (mi.is_zero()) throw CasMathError("ModularInverse: the modulus is zero");
        BigInt g, x, y;
        egcd(floor_mod(args[0]->q.num, mi), mi, g, x, y);
        if (!g.is_one()) throw CasMathError("the number has no inverse modulo m (not coprime)");
        return num(Rational::from_bigint(floor_mod(x, mi)));
    }
    if (head == "PartitionsP" && args.size() == 1 && args[0]->is_integer()) {
        int64_t n = 0;
        if (!args[0]->q.num.to_int64(n) || n < 0) throw CasMathError("PartitionsP needs a non-negative integer");
        if (n > 20000) throw CasLimitError("PartitionsP argument too large");
        std::vector<BigInt> p(static_cast<size_t>(n) + 1);
        p[0] = BigInt(1);
        for (int64_t i = 1; i <= n; ++i) {
            BigInt s(0);
            for (int64_t k = 1;; ++k) {
                const int64_t g1 = k * (3 * k - 1) / 2, g2 = k * (3 * k + 1) / 2;
                if (g1 > i) break;
                if (k % 2) { s = s + p[i - g1]; if (g2 <= i) s = s + p[i - g2]; }
                else { s = s - p[i - g1]; if (g2 <= i) s = s - p[i - g2]; }
            }
            p[i] = s;
            tick(static_cast<uint64_t>(i));
        }
        return num(Rational::from_bigint(p[n]));
    }
    // ---------------------------------------------------------------- finite fields GF(p): polynomials and multiplicative structure
    if (head == "PolynomialMod" && (args.size() == 2 || args.size() == 3) && args.back()->is_integer() && (args.size() == 2 || args[1]->is_symbol())) {
        Expr var_e;
        if (args.size() == 3) var_e = args[1];
        else {   // PolynomialMod[f, m]: the one variable of f
            std::vector<std::string> fs;
            free_symbols(args[0], fs);
            std::sort(fs.begin(), fs.end());
            fs.erase(std::unique(fs.begin(), fs.end()), fs.end());
            if (fs.size() > 1) throw CasUnsupported("PolynomialMod: several variables; name the variable");
            var_e = symbol(fs.empty() ? "x" : fs[0]);
        }
        const std::vector<Expr> margs = {args[0], var_e, args.back()};
        u64 p = 0;
        if (!to_u64(margs[2], p) || p < 2 || p >= (1ULL << 62)) throw CasMathError("PolynomialMod: the modulus must be an integer >= 2");
        std::vector<Expr> co;
        if (!poly_coefficients(margs[0], margs[1]->name, co)) throw CasUnsupported("PolynomialMod needs a polynomial in the variable");
        Poly out;
        for (const auto& c : co) {
            if (!c->is_integer()) throw CasUnsupported("PolynomialMod needs integer coefficients");
            BigInt r = floor_mod(c->q.num, BigInt(static_cast<int64_t>(p)));
            out.push_back(Rational::from_bigint(r));
        }
        poly_trim(out);
        return out.empty() ? zero() : poly_to_expr(out, margs[1]->name);
    }
    if (head == "PolynomialGCDMod" && args.size() == 4 && args[2]->is_symbol() && args[3]->is_integer()) {   // gcd in GF(p)[x], monic
        u64 p = 0;
        if (!to_u64(args[3], p) || p < 2 || p >= (1ULL << 31)) throw CasMathError("PolynomialGCDMod: the modulus must be a prime below 2^31");
        { std::vector<u64> pf; if (!prime_factors(p, pf) || pf.size() != 1 || pf[0] != p) throw CasMathError("PolynomialGCDMod needs a prime modulus"); }
        auto load = [&](const Expr& f, std::vector<u64>& v) {
            std::vector<Expr> co;
            if (!poly_coefficients(f, args[2]->name, co)) throw CasUnsupported("PolynomialGCDMod needs polynomials in the variable");
            for (const auto& c : co) { if (!c->is_integer()) throw CasUnsupported("integer coefficients only"); v.push_back(static_cast<u64>(floor_mod(c->q.num, BigInt(static_cast<int64_t>(p))).to_double())); }
            while (!v.empty() && v.back() == 0) v.pop_back();
        };
        std::vector<u64> a, b;
        load(args[0], a);
        load(args[1], b);
        while (!b.empty()) {   // Euclid: a mod b
            const u64 inv = powmod(b.back(), p - 2, p);
            while (a.size() >= b.size() && !a.empty()) {
                const u64 f = mulmod(a.back(), inv, p);
                const size_t shift = a.size() - b.size();
                for (size_t i = 0; i < b.size(); ++i) a[shift + i] = (a[shift + i] + p - mulmod(f, b[i], p)) % p;
                while (!a.empty() && a.back() == 0) a.pop_back();
                tick(b.size());
            }
            std::swap(a, b);
        }
        if (a.empty()) return zero();
        const u64 inv = powmod(a.back(), p - 2, p);
        Poly out;
        for (u64 c : a) out.push_back(Rational::from_bigint(BigInt(static_cast<int64_t>(mulmod(c, inv, p)))));
        return poly_to_expr(out, args[2]->name);
    }
    if (head == "FactorMod" && args.size() == 3 && args[1]->is_symbol() && args[2]->is_integer()) {   // FactorMod[f, x, p]: factorization in GF(p)[x]
        u64 p = 0;
        if (!to_u64(args[2], p) || p < 2 || p >= (1ULL << 31)) throw CasMathError("FactorMod: the modulus must be a prime below 2^31");
        { std::vector<u64> pf; if (!prime_factors(p, pf) || pf.size() != 1 || pf[0] != p) throw CasMathError("FactorMod needs a prime modulus"); }
        std::vector<Expr> co;
        if (!poly_coefficients(args[0], args[1]->name, co)) throw CasUnsupported("FactorMod needs a polynomial in the variable");
        std::vector<u64> f;
        for (const auto& c : co) { if (!c->is_integer()) throw CasUnsupported("FactorMod needs integer coefficients"); f.push_back(static_cast<u64>(floor_mod(c->q.num, BigInt(static_cast<int64_t>(p))).to_double())); }
        const TickFn tk = [this](uint64_t k) { tick(k); };
        ModFactorization fz = factor_over_gfp(f, p, tk);
        if (fz.content == 0) return zero();
        std::vector<Expr> parts;
        if (fz.content != 1) parts.push_back(integer(static_cast<int64_t>(fz.content)));
        for (const auto& [g, m] : fz.factors) {
            Poly pg;
            for (u64 c : g) pg.push_back(Rational::from_bigint(BigInt(static_cast<int64_t>(c))));
            Expr base = poly_to_expr(pg, args[1]->name);
            parts.push_back(m == 1 ? base : app("Power", {base, integer(m)}));
        }
        return parts.empty() ? one() : parts.size() == 1 ? parts[0] : app("Times", parts);
    }
    if (head == "MultiplicativeOrder" && args.size() == 2) {
        u64 a = 0, n = 0;
        if (!to_u64(args[0], a) || !to_u64(args[1], n) || n < 2 || n >= (1ULL << 62)) throw CasMathError("MultiplicativeOrder needs integers a >= 0 and n >= 2");
        if (gcd64(a % n, n) != 1) throw CasMathError("the order is undefined: a and n are not coprime");
        std::vector<u64> pn;
        if (!prime_factors(n, pn)) throw CasLimitError("n has a prime factor too large to factor by trial division");
        // phi(n): need the full factorization with multiplicities for phi
        u64 phi = n, m = n;
        for (u64 p : pn) { phi = phi / p * (p - 1); while (m % p == 0) m /= p; }
        std::vector<u64> pp;
        if (!prime_factors(phi, pp)) throw CasLimitError("phi(n) has a prime factor too large to factor by trial division");
        u64 ord = phi;
        for (u64 q : pp) while (ord % q == 0 && powmod(a, ord / q, n) == 1) ord /= q;
        return integer(static_cast<int64_t>(ord));
    }
    if (head == "PrimitiveRoot" && args.size() == 1) {
        u64 n = 0;
        if (!to_u64(args[0], n) || n < 2 || n >= (1ULL << 62)) throw CasMathError("PrimitiveRoot needs an integer n >= 2");
        if (n == 2) return integer(1);
        std::vector<u64> pn;
        if (!prime_factors(n, pn)) throw CasLimitError("n has a prime factor too large to factor by trial division");
        u64 phi = n;
        for (u64 p : pn) phi = phi / p * (p - 1);
        std::vector<u64> pp;
        if (!prime_factors(phi, pp)) throw CasLimitError("phi(n) has a prime factor too large to factor by trial division");
        for (u64 g = 2; g < n && g < 10'000'000; ++g) {
            if (gcd64(g, n) != 1) continue;
            bool root = true;
            for (u64 q : pp) if (powmod(g, phi / q, n) == 1) { root = false; break; }
            if (root) {
                // a primitive root needs the group to be cyclic: order of g must be phi(n); verified above by the prime-divisor test
                return integer(static_cast<int64_t>(g));
            }
        }
        throw CasMathError("no primitive root exists for this n");
    }
    return nullptr;
}

} // namespace strata::math::cas
