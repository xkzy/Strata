// SPDX-License-Identifier: GPL-3.0-or-later
// include/strata/math/verify.hpp - deterministic mathematical verification with explicit assumptions
//
// Three outcomes only: VERIFIED, CONTRADICTED, UNKNOWN. UNKNOWN is never turned into false, and a missing assumption is
// never assumed: sqrt(x^2) = x is UNKNOWN / CONTRADICTED until x >= 0 is given. Every result carries its provenance.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace strata::math {

enum class VerifyState { kVerified, kContradicted, kUnknown };
inline const char* to_string(VerifyState s) { return s == VerifyState::kVerified ? "VERIFIED" : s == VerifyState::kContradicted ? "CONTRADICTED" : "UNKNOWN"; }

// What is known about the variables. Text forms accepted by parse(): "x > 0", "x >= 0", "x < 5", "x != 0", "x in Reals",
// "x ∈ Real", "n integer", "n in Integers", several separated by "," / ";" / "and".
struct VarFacts {
    bool real = false, integer = false, nonzero = false;
    bool has_lo = false, has_hi = false;
    double lo = 0, hi = 0;
    bool lo_strict = false, hi_strict = false;
    bool admits(double v) const;   // is the sample value allowed?
};

class Assumptions {
public:
    static bool parse(const std::string& text, Assumptions& out, std::string* error = nullptr);
    bool add(const std::string& clause, std::string* error = nullptr);
    const VarFacts* find(const std::string& var) const;
    VarFacts& get(const std::string& var) { return facts_[var]; }
    bool empty() const { return facts_.empty(); }
    std::string to_string() const;   // canonical text, stable ordering (part of cache keys)
    const std::map<std::string, VarFacts>& all() const { return facts_; }
private:
    std::map<std::string, VarFacts> facts_;
};

struct Provenance {
    std::string expression_hash;           // of the normalized input
    std::string normalized_expression;
    std::string backend = "StrataCAS";
    std::string backend_version;
    std::string algorithm;                 // e.g. "identity: expand/together/simplify + witness search"
    std::string assumptions;               // canonical text
    std::vector<std::string> input_hashes;
    int precision_digits = 0;              // 0: exact
    std::string rounding = "n/a";
    bool exact = true;
    std::string ir_version;
    int64_t timestamp_ms = 0;
};

struct VerifyResult {
    VerifyState state = VerifyState::kUnknown;
    std::string explanation;
    std::vector<std::string> assumptions_used;   // assumptions the verdict depends on
    std::vector<std::string> conditions;         // e.g. "x - 1 != 0": where both sides are defined
    std::string witness;                         // a counterexample or the point that decided it
    Provenance provenance;
    std::string diagnostic;                      // structured reason for UNKNOWN / unsupported
};

class MathVerifier {
public:
    // "a" and "b" are expressions in the CAS syntax (Mathematica / Sage-like). Variables without assumptions are unconstrained.
    // verify_equal: the claim a = b. With free variables it is VERIFIED only when it is an identity; a statement that
    // holds for some values only is UNKNOWN (it may have been meant as an equation to solve).
    VerifyResult verify_equal(const std::string& a, const std::string& b, const std::string& assumptions = "") const;
    VerifyResult verify_not_equal(const std::string& a, const std::string& b, const std::string& assumptions = "") const;
    // verify_identity: a = b for ALL admissible values; a counterexample is CONTRADICTED.
    VerifyResult verify_identity(const std::string& a, const std::string& b, const std::string& assumptions = "") const;
    // "lhs == rhs" or "lhs = rhs"
    VerifyResult verify_equation(const std::string& equation, const std::string& assumptions = "") const;
    // "lhs < rhs", "<=", ">", ">=": for all admissible values of the variables
    VerifyResult verify_inequality(const std::string& relation, const std::string& assumptions = "") const;
    // any of the above, chosen by the top-level relation
    VerifyResult verify(const std::string& statement, const std::string& assumptions = "") const;

    static const char* version() { return "cas-0.3.0"; }
    static const char* ir_version() { return "ir-1"; }
};

// Units and dimensions: m, km, s, Hz, N, J, W, Pa, V, A, Ohm/Ω, F, C, bytes, bits ... and products / quotients / powers.
struct Dimension {
    int e[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};   // m, kg, s, A, K, mol, cd, bit, (currency-like extension slot)
    bool operator==(const Dimension& o) const { for (int i = 0; i < 9; ++i) if (e[i] != o.e[i]) return false; return true; }
    std::string to_string() const;
};

struct UnitValue {
    Dimension dim;
    double scale = 1.0;   // to SI base units (bits for information)
    bool ok = false;
    std::string error;
};

class UnitEngine {
public:
    // "kg*m/s^2", "V*A", "N*m", "km/h" ... (no numbers other than exponents and scale factors)
    static UnitValue parse(const std::string& unit_expression);
    // "V*A = W": same dimension -> VERIFIED (scale also compared when both sides are fully convertible); different -> CONTRADICTED;
    // an unknown unit symbol -> UNKNOWN (never guessed)
    static VerifyResult verify_equation(const std::string& equation);
    static VerifyResult verify_convertible(const std::string& from, const std::string& to);
};

} // namespace strata::math
