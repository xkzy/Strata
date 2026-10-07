// SPDX-License-Identifier: GPL-3.0-or-later
// include/strata/math/cas/bigint.hpp - arbitrary-precision integers for the exact CAS
//
// Part of the native CAS (src/math/cas), a C++ port of the evaluation model of Mathics3 (GPL-3.0-or-later,
// https://github.com/Mathics3/mathics-core). See docs/MATH_RUNTIME.md for the licensing note.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::math::cas {

// Sign-magnitude integer, base 2^32 limbs (least significant first). Never overflows; size is capped by callers.
class BigInt {
public:
    BigInt() = default;
    BigInt(int64_t v);  // NOLINT: implicit on purpose, mirrors built-in integers

    static bool from_string(const std::string& s, BigInt& out);

    bool is_zero() const { return mag_.empty(); }
    bool is_negative() const { return neg_; }
    int sign() const { return mag_.empty() ? 0 : (neg_ ? -1 : 1); }
    bool is_one() const { return !neg_ && mag_.size() == 1 && mag_[0] == 1; }
    bool is_even() const { return mag_.empty() || (mag_[0] & 1u) == 0; }
    size_t limb_count() const { return mag_.size(); }
    size_t bit_length() const;
    bool bit(size_t i) const { return i / 32 < mag_.size() && ((mag_[i / 32] >> (i % 32)) & 1u); }

    // Exact conversions; return false if the value does not fit.
    bool to_int64(int64_t& out) const;
    double to_double() const;

    std::string to_string() const;

    BigInt operator-() const;
    BigInt abs() const;
    friend BigInt operator+(const BigInt& a, const BigInt& b);
    friend BigInt operator-(const BigInt& a, const BigInt& b);
    friend BigInt operator*(const BigInt& a, const BigInt& b);
    // Truncated division (quotient rounds toward zero, remainder has the dividend's sign). b must be non-zero.
    static void divmod(const BigInt& a, const BigInt& b, BigInt& q, BigInt& r);
    friend BigInt operator/(const BigInt& a, const BigInt& b);
    friend BigInt operator%(const BigInt& a, const BigInt& b);

    static int compare(const BigInt& a, const BigInt& b);
    friend bool operator==(const BigInt& a, const BigInt& b) { return a.neg_ == b.neg_ && a.mag_ == b.mag_; }
    friend bool operator!=(const BigInt& a, const BigInt& b) { return !(a == b); }
    friend bool operator<(const BigInt& a, const BigInt& b) { return compare(a, b) < 0; }
    friend bool operator<=(const BigInt& a, const BigInt& b) { return compare(a, b) <= 0; }
    friend bool operator>(const BigInt& a, const BigInt& b) { return compare(a, b) > 0; }
    friend bool operator>=(const BigInt& a, const BigInt& b) { return compare(a, b) >= 0; }

    static BigInt gcd(BigInt a, BigInt b);
    // base^exp for exp >= 0.
    static BigInt pow(const BigInt& base, uint64_t exp);
    // floor(sqrt(n)) for n >= 0.
    static BigInt isqrt(const BigInt& n);

private:
    bool neg_ = false;
    std::vector<uint32_t> mag_;

    void trim();
    static int cmp_mag(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b);
    static std::vector<uint32_t> add_mag(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b);
    static std::vector<uint32_t> sub_mag(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b);  // a >= b
    static void divmod_mag(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b,
                           std::vector<uint32_t>& q, std::vector<uint32_t>& r);
};

} // namespace strata::math::cas
