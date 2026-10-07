// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/cas/bigint.cpp - arbitrary-precision integers (see bigint.hpp)
#include "strata/math/cas/bigint.hpp"

#include <algorithm>
#include <cmath>

namespace strata::math::cas {

BigInt::BigInt(int64_t v) {
    if (v == 0) return;
    neg_ = v < 0;
    uint64_t u = neg_ ? (~static_cast<uint64_t>(v) + 1) : static_cast<uint64_t>(v);
    while (u) {
        mag_.push_back(static_cast<uint32_t>(u & 0xFFFFFFFFu));
        u >>= 32;
    }
}

void BigInt::trim() {
    while (!mag_.empty() && mag_.back() == 0) mag_.pop_back();
    if (mag_.empty()) neg_ = false;
}

size_t BigInt::bit_length() const {
    if (mag_.empty()) return 0;
    size_t bits = (mag_.size() - 1) * 32;
    uint32_t top = mag_.back();
    while (top) { ++bits; top >>= 1; }
    return bits;
}

int BigInt::cmp_mag(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
    for (size_t i = a.size(); i-- > 0;) {
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    }
    return 0;
}

std::vector<uint32_t> BigInt::add_mag(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    const auto& big = a.size() >= b.size() ? a : b;
    const auto& small = a.size() >= b.size() ? b : a;
    std::vector<uint32_t> r;
    r.reserve(big.size() + 1);
    uint64_t carry = 0;
    for (size_t i = 0; i < big.size(); ++i) {
        uint64_t s = static_cast<uint64_t>(big[i]) + (i < small.size() ? small[i] : 0) + carry;
        r.push_back(static_cast<uint32_t>(s & 0xFFFFFFFFu));
        carry = s >> 32;
    }
    if (carry) r.push_back(static_cast<uint32_t>(carry));
    return r;
}

std::vector<uint32_t> BigInt::sub_mag(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    std::vector<uint32_t> r;
    r.reserve(a.size());
    int64_t borrow = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        int64_t d = static_cast<int64_t>(a[i]) - (i < b.size() ? b[i] : 0) - borrow;
        if (d < 0) { d += (int64_t{1} << 32); borrow = 1; } else { borrow = 0; }
        r.push_back(static_cast<uint32_t>(d));
    }
    while (!r.empty() && r.back() == 0) r.pop_back();
    return r;
}

BigInt BigInt::operator-() const {
    BigInt r = *this;
    if (!r.mag_.empty()) r.neg_ = !r.neg_;
    return r;
}

BigInt BigInt::abs() const {
    BigInt r = *this;
    r.neg_ = false;
    return r;
}

BigInt operator+(const BigInt& a, const BigInt& b) {
    BigInt r;
    if (a.neg_ == b.neg_) {
        r.mag_ = BigInt::add_mag(a.mag_, b.mag_);
        r.neg_ = a.neg_;
    } else {
        int c = BigInt::cmp_mag(a.mag_, b.mag_);
        if (c == 0) return BigInt();
        if (c > 0) { r.mag_ = BigInt::sub_mag(a.mag_, b.mag_); r.neg_ = a.neg_; }
        else       { r.mag_ = BigInt::sub_mag(b.mag_, a.mag_); r.neg_ = b.neg_; }
    }
    r.trim();
    return r;
}

BigInt operator-(const BigInt& a, const BigInt& b) { return a + (-b); }

BigInt operator*(const BigInt& a, const BigInt& b) {
    BigInt r;
    if (a.mag_.empty() || b.mag_.empty()) return r;
    r.mag_.assign(a.mag_.size() + b.mag_.size(), 0);
    for (size_t i = 0; i < a.mag_.size(); ++i) {
        uint64_t carry = 0;
        for (size_t j = 0; j < b.mag_.size(); ++j) {
            uint64_t cur = static_cast<uint64_t>(a.mag_[i]) * b.mag_[j] + r.mag_[i + j] + carry;
            r.mag_[i + j] = static_cast<uint32_t>(cur & 0xFFFFFFFFu);
            carry = cur >> 32;
        }
        size_t k = i + b.mag_.size();
        while (carry) {
            uint64_t cur = static_cast<uint64_t>(r.mag_[k]) + carry;
            r.mag_[k] = static_cast<uint32_t>(cur & 0xFFFFFFFFu);
            carry = cur >> 32;
            ++k;
        }
    }
    r.neg_ = a.neg_ != b.neg_;
    r.trim();
    return r;
}

// Knuth algorithm D on base-2^32 limbs.
void BigInt::divmod_mag(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b,
                        std::vector<uint32_t>& q, std::vector<uint32_t>& r) {
    q.clear();
    r.clear();
    if (cmp_mag(a, b) < 0) { r = a; return; }
    if (b.size() == 1) {
        uint64_t rem = 0;
        q.assign(a.size(), 0);
        for (size_t i = a.size(); i-- > 0;) {
            uint64_t cur = (rem << 32) | a[i];
            q[i] = static_cast<uint32_t>(cur / b[0]);
            rem = cur % b[0];
        }
        while (!q.empty() && q.back() == 0) q.pop_back();
        if (rem) r.push_back(static_cast<uint32_t>(rem));
        return;
    }
    const unsigned shift = static_cast<unsigned>(__builtin_clz(b.back()));
    auto shl = [&](const std::vector<uint32_t>& v, size_t extra) {
        std::vector<uint32_t> out(v.size() + extra, 0);
        uint32_t carry = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            uint64_t cur = (static_cast<uint64_t>(v[i]) << shift) | carry;
            out[i] = static_cast<uint32_t>(cur & 0xFFFFFFFFu);
            carry = shift ? static_cast<uint32_t>(cur >> 32) : 0;
        }
        if (extra) out[v.size()] = carry;
        return out;
    };
    std::vector<uint32_t> v = shl(b, 0);
    std::vector<uint32_t> u = shl(a, 1);
    const size_t n = v.size();
    const size_t m = u.size() - n - 1;
    q.assign(m + 1, 0);
    const uint64_t base = uint64_t{1} << 32;
    for (size_t j = m + 1; j-- > 0;) {
        uint64_t num = (static_cast<uint64_t>(u[j + n]) << 32) | u[j + n - 1];
        uint64_t qhat = num / v[n - 1];
        uint64_t rhat = num % v[n - 1];
        while (qhat >= base || qhat * v[n - 2] > ((rhat << 32) | u[j + n - 2])) {
            --qhat;
            rhat += v[n - 1];
            if (rhat >= base) break;
        }
        int64_t borrow = 0;
        uint64_t carry = 0;
        for (size_t i = 0; i < n; ++i) {
            uint64_t p = qhat * v[i] + carry;
            carry = p >> 32;
            int64_t t = static_cast<int64_t>(u[i + j]) - borrow - static_cast<int64_t>(p & 0xFFFFFFFFu);
            u[i + j] = static_cast<uint32_t>(t);
            borrow = (t < 0) ? 1 : 0;
        }
        int64_t t = static_cast<int64_t>(u[j + n]) - borrow - static_cast<int64_t>(carry);
        u[j + n] = static_cast<uint32_t>(t);
        if (t < 0) {
            --qhat;
            uint64_t c = 0;
            for (size_t i = 0; i < n; ++i) {
                uint64_t s = static_cast<uint64_t>(u[i + j]) + v[i] + c;
                u[i + j] = static_cast<uint32_t>(s & 0xFFFFFFFFu);
                c = s >> 32;
            }
            u[j + n] = static_cast<uint32_t>(u[j + n] + c);
        }
        q[j] = static_cast<uint32_t>(qhat);
    }
    while (!q.empty() && q.back() == 0) q.pop_back();
    // remainder = u[0..n) >> shift
    r.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        uint64_t cur = u[i] >> shift;
        if (shift && i + 1 < u.size()) cur |= (static_cast<uint64_t>(u[i + 1]) << (32 - shift)) & 0xFFFFFFFFu;
        r[i] = static_cast<uint32_t>(cur);
    }
    while (!r.empty() && r.back() == 0) r.pop_back();
}

void BigInt::divmod(const BigInt& a, const BigInt& b, BigInt& q, BigInt& r) {
    BigInt qq, rr;
    divmod_mag(a.mag_, b.mag_, qq.mag_, rr.mag_);
    qq.neg_ = a.neg_ != b.neg_;
    rr.neg_ = a.neg_;
    qq.trim();
    rr.trim();
    q = std::move(qq);
    r = std::move(rr);
}

BigInt operator/(const BigInt& a, const BigInt& b) { BigInt q, r; BigInt::divmod(a, b, q, r); return q; }
BigInt operator%(const BigInt& a, const BigInt& b) { BigInt q, r; BigInt::divmod(a, b, q, r); return r; }

int BigInt::compare(const BigInt& a, const BigInt& b) {
    if (a.neg_ != b.neg_) return a.neg_ ? -1 : 1;
    int c = cmp_mag(a.mag_, b.mag_);
    return a.neg_ ? -c : c;
}

BigInt BigInt::gcd(BigInt a, BigInt b) {
    a.neg_ = false;
    b.neg_ = false;
    while (!b.is_zero()) {
        BigInt r = a % b;
        a = std::move(b);
        b = std::move(r);
    }
    return a;
}

BigInt BigInt::pow(const BigInt& base, uint64_t exp) {
    BigInt result(1), b = base;
    while (exp) {
        if (exp & 1) result = result * b;
        exp >>= 1;
        if (exp) b = b * b;
    }
    return result;
}

BigInt BigInt::isqrt(const BigInt& n) {
    if (n.is_zero() || n.is_negative()) return BigInt();
    // Newton iteration from an over-estimate.
    BigInt x = BigInt::pow(BigInt(2), (n.bit_length() + 1) / 2);
    while (true) {
        BigInt y = (x + n / x) / BigInt(2);
        if (y >= x) return x;
        x = std::move(y);
    }
}

bool BigInt::to_int64(int64_t& out) const {
    if (mag_.size() > 2) return false;
    uint64_t u = 0;
    if (mag_.size() >= 1) u |= mag_[0];
    if (mag_.size() == 2) u |= static_cast<uint64_t>(mag_[1]) << 32;
    if (neg_) {
        if (u > (uint64_t{1} << 63)) return false;
        out = static_cast<int64_t>(~u + 1);
    } else {
        if (u > static_cast<uint64_t>(INT64_MAX)) return false;
        out = static_cast<int64_t>(u);
    }
    return true;
}

double BigInt::to_double() const {
    double r = 0.0;
    for (size_t i = mag_.size(); i-- > 0;) r = r * 4294967296.0 + mag_[i];
    return neg_ ? -r : r;
}

std::string BigInt::to_string() const {
    if (mag_.empty()) return "0";
    std::vector<uint32_t> cur = mag_;
    std::string digits;
    while (!cur.empty()) {
        uint64_t rem = 0;
        for (size_t i = cur.size(); i-- > 0;) {
            uint64_t v = (rem << 32) | cur[i];
            cur[i] = static_cast<uint32_t>(v / 1000000000u);
            rem = v % 1000000000u;
        }
        while (!cur.empty() && cur.back() == 0) cur.pop_back();
        for (int k = 0; k < 9; ++k) {
            digits.push_back(static_cast<char>('0' + rem % 10));
            rem /= 10;
            if (cur.empty() && rem == 0) break;
        }
    }
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    if (neg_) digits.push_back('-');
    std::reverse(digits.begin(), digits.end());
    return digits;
}

bool BigInt::from_string(const std::string& s, BigInt& out) {
    size_t i = 0;
    bool neg = false;
    if (i < s.size() && (s[i] == '-' || s[i] == '+')) { neg = s[i] == '-'; ++i; }
    if (i >= s.size()) return false;
    BigInt r;
    const BigInt ten(10);
    for (; i < s.size(); ++i) {
        if (s[i] < '0' || s[i] > '9') return false;
        r = r * ten + BigInt(s[i] - '0');
    }
    if (neg) r = -r;
    out = std::move(r);
    return true;
}

} // namespace strata::math::cas
