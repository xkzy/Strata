// SPDX-License-Identifier: GPL-3.0-or-later
// include/strata/math/cas/parser.hpp - text <-> Expr for the native CAS
#pragma once

#include "strata/math/cas/expr.hpp"

#include <string>

namespace strata::math::cas {

// Parses infix math: + - * / ^ **, implicit multiplication (5x, 2 sin(x), (x+1)(x-1)), function calls f(x) and
// f[x], lists {a, b}, equations ==, rules ->, factorial !, constants pi/Pi, E, I. Function names are
// case-insensitive for the known set (sin, Sin, SIN). Throws CasError on bad input.
Expr parse(const std::string& text, size_t max_depth = 200);

// Prints in the same infix syntax the parser reads (lower-case function names, "*" without spaces, polynomials
// in descending degree, "x^2/3" style rational coefficients).
std::string to_string(const Expr& e);

} // namespace strata::math::cas
