// SPDX-License-Identifier: GPL-3.0-or-later
// include/strata/math/cas/expr.hpp - symbolic expression tree for the native CAS
//
// A Wolfram-language style expression: exact rational numbers, machine reals, symbols and applied heads
// (Plus, Times, Power, Sin, List, ...). Port of the expression model of Mathics3 (GPL-3.0-or-later) to C++.
#pragma once

#include "strata/math/cas/bigint.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::math::cas {

// Thrown when a request exceeds its resource budget (terms, degree, steps, depth, time). The backend maps it to
// kResourceLimitExceeded; it is never a wrong answer.
struct CasLimitError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Base of every "no answer" condition. The backend maps each subclass to an error status, never to a made-up result.
struct CasError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
// The text is not valid input.
struct CasParseError : CasError {
    using CasError::CasError;
};
// The question has no (real, finite) answer: division by zero, domain errors.
struct CasMathError : CasError {
    using CasError::CasError;
};
// The CAS does not implement this case. The verifier turns it into UNKNOWN, never into FAIL.
struct CasUnsupported : CasError {
    using CasError::CasError;
};

struct Rational {
    BigInt num;
    BigInt den = BigInt(1);  // always > 0, gcd(num, den) == 1

    Rational() = default;
    Rational(int64_t n) : num(n) {}  // NOLINT
    Rational(BigInt n, BigInt d);
    static Rational from_bigint(const BigInt& n) { Rational r; r.num = n; return r; }

    bool is_integer() const { return den.is_one(); }
    bool is_zero() const { return num.is_zero(); }
    bool is_one() const { return num.is_one() && den.is_one(); }
    bool is_negative() const { return num.is_negative(); }
    int sign() const { return num.sign(); }
    double to_double() const;
    std::string to_string() const;

    Rational operator-() const;
    friend Rational operator+(const Rational& a, const Rational& b);
    friend Rational operator-(const Rational& a, const Rational& b);
    friend Rational operator*(const Rational& a, const Rational& b);
    friend Rational operator/(const Rational& a, const Rational& b);  // b != 0
    friend bool operator==(const Rational& a, const Rational& b) { return a.num == b.num && a.den == b.den; }
    friend bool operator!=(const Rational& a, const Rational& b) { return !(a == b); }
    friend bool operator<(const Rational& a, const Rational& b);
    Rational reciprocal() const;  // non-zero
    // Integer power (exp may be negative); the result's size is the caller's responsibility to cap.
    Rational pow(int64_t exp) const;
};

enum class Kind { Number, Real, Symbol, Apply };

class Node;
using Expr = std::shared_ptr<const Node>;

class Node {
public:
    Kind kind;
    Rational q;                  // Number
    double d = 0.0;              // Real
    std::string name;            // Symbol name, or the head symbol of an Apply
    std::vector<Expr> args;      // Apply

    bool is_number() const { return kind == Kind::Number; }
    bool is_integer() const { return kind == Kind::Number && q.is_integer(); }
    bool is_symbol() const { return kind == Kind::Symbol; }
    bool is_apply() const { return kind == Kind::Apply; }
    bool has_head(const char* h) const { return kind == Kind::Apply && name == h; }
    bool has_head(const char* h, size_t nargs) const { return kind == Kind::Apply && name == h && args.size() == nargs; }
    bool is_symbol_named(const char* s) const { return kind == Kind::Symbol && name == s; }
};

// Constructors (no evaluation).
Expr num(const Rational& q);
Expr integer(int64_t v);
Expr real(double d);
Expr symbol(const std::string& name);
Expr app(const std::string& head, std::vector<Expr> args);
Expr apply1(const std::string& head, Expr a);
Expr apply2(const std::string& head, Expr a, Expr b);

// Frequently used constants.
Expr zero();
Expr one();
Expr minus_one();

// Structural equality and a deterministic total order (used to sort Plus/Times arguments).
bool equal(const Expr& a, const Expr& b);
int compare(const Expr& a, const Expr& b);
// Number of nodes (used for "simplest form" selection and budgets).
size_t leaf_count(const Expr& e);
// True if `e` contains the symbol `var` anywhere.
bool depends_on(const Expr& e, const std::string& var);
// Collects the free symbol names (excluding the constants Pi, E, I) in sorted order.
void free_symbols(const Expr& e, std::vector<std::string>& out);

} // namespace strata::math::cas
