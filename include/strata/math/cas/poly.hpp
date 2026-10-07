// SPDX-License-Identifier: GPL-3.0-or-later
// include/strata/math/cas/poly.hpp - univariate polynomials over Q and their factorization
//
// Dense polynomials, coefficients low to high. Factorization over Q is exact: square-free decomposition,
// factorization modulo a prime (Cantor-Zassenhaus), Hensel lifting and recombination (Zassenhaus).
#pragma once

#include "strata/math/cas/expr.hpp"

#include <functional>
#include <utility>
#include <vector>

namespace strata::math::cas {

using Poly = std::vector<Rational>;     // c[i] * x^i, no trailing zeros; the zero polynomial is empty
using IntPoly = std::vector<BigInt>;    // same layout over Z

using TickFn = std::function<void(uint64_t)>;

int poly_degree(const Poly& p);         // -1 for zero
void poly_trim(Poly& p);
Poly poly_add(const Poly& a, const Poly& b);
Poly poly_sub(const Poly& a, const Poly& b);
Poly poly_mul(const Poly& a, const Poly& b);
Poly poly_scale(const Poly& a, const Rational& k);
void poly_divmod(const Poly& a, const Poly& b, Poly& q, Poly& r);   // b != 0
Poly poly_gcd(const Poly& a, const Poly& b);                          // monic, or empty if both zero
Poly poly_deriv(const Poly& a);
Rational poly_eval(const Poly& a, const Rational& x);
Poly poly_monic(const Poly& a);

struct Factorization {
    Rational content = Rational(1);                  // overall constant factor (carries the sign)
    std::vector<std::pair<Poly, int>> factors;       // irreducible, integer coefficients, positive leading coeff
};

// Factors a non-zero polynomial over Q into irreducibles. Throws CasLimitError when the work exceeds the budget.
Factorization factor_over_q(const Poly& f, const TickFn& tick);

struct ModFactorization {
    uint64_t content = 1;                                           // leading coefficient
    std::vector<std::pair<std::vector<uint64_t>, int>> factors;     // monic irreducibles (coefficients low to high) with multiplicity
};
// Factors a non-zero polynomial over GF(p) (p prime, p < 2^31): square-free decomposition (including p-th roots), distinct-degree and
// equal-degree factorization. Coefficients are reduced mod p, low to high.
ModFactorization factor_over_gfp(std::vector<uint64_t> f, uint64_t p, const TickFn& tick);

} // namespace strata::math::cas
