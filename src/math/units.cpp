// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/units.cpp - dimensional analysis: units as exponent vectors over the SI base dimensions (+ information bits)
#include "strata/math/verify.hpp"

#include <cctype>
#include <cmath>
#include <map>
#include <sstream>

namespace strata::math {

namespace {

const char* kDimNames[9] = {"m", "kg", "s", "A", "K", "mol", "cd", "bit", "?"};

struct Def { Dimension dim; double scale; bool prefixable; };

Dimension D(int m = 0, int kg = 0, int s = 0, int A = 0, int K = 0, int mol = 0, int cd = 0, int bit = 0) {
    Dimension d;
    d.e[0] = m; d.e[1] = kg; d.e[2] = s; d.e[3] = A; d.e[4] = K; d.e[5] = mol; d.e[6] = cd; d.e[7] = bit;
    return d;
}

const std::map<std::string, Def>& table() {
    static const std::map<std::string, Def> t = {
        {"m", {D(1), 1, true}}, {"g", {D(0, 1), 1e-3, true}}, {"s", {D(0, 0, 1), 1, true}}, {"A", {D(0, 0, 0, 1), 1, true}},
        {"K", {D(0, 0, 0, 0, 1), 1, true}}, {"mol", {D(0, 0, 0, 0, 0, 1), 1, true}}, {"cd", {D(0, 0, 0, 0, 0, 0, 1), 1, true}},
        {"Hz", {D(0, 0, -1), 1, true}}, {"N", {D(1, 1, -2), 1, true}}, {"J", {D(2, 1, -2), 1, true}}, {"W", {D(2, 1, -3), 1, true}},
        {"Pa", {D(-1, 1, -2), 1, true}}, {"V", {D(2, 1, -3, -1), 1, true}}, {"Ohm", {D(2, 1, -3, -2), 1, true}},
        {"\xCE\xA9", {D(2, 1, -3, -2), 1, true}}, {"F", {D(-2, -1, 4, 2), 1, true}}, {"C", {D(0, 0, 1, 1), 1, true}},
        {"H", {D(2, 1, -2, -2), 1, true}}, {"T", {D(0, 1, -2, -1), 1, true}}, {"Wb", {D(2, 1, -2, -1), 1, true}}, {"S", {D(-2, -1, 3, 2), 1, true}},
        {"L", {D(3), 1e-3, true}}, {"min", {D(0, 0, 1), 60, false}}, {"h", {D(0, 0, 1), 3600, false}}, {"day", {D(0, 0, 1), 86400, false}},
        {"bit", {D(0, 0, 0, 0, 0, 0, 0, 1), 1, true}}, {"bits", {D(0, 0, 0, 0, 0, 0, 0, 1), 1, true}},
        {"B", {D(0, 0, 0, 0, 0, 0, 0, 1), 8, true}}, {"byte", {D(0, 0, 0, 0, 0, 0, 0, 1), 8, true}}, {"bytes", {D(0, 0, 0, 0, 0, 0, 0, 1), 8, true}},
        {"rad", {D(), 1, false}}, {"eV", {D(2, 1, -2), 1.602176634e-19, true}},
    };
    return t;
}

bool prefix_scale(const std::string& p, double& s) {
    static const std::map<std::string, double> m = {
        {"k", 1e3}, {"M", 1e6}, {"G", 1e9}, {"T", 1e12}, {"P", 1e15}, {"m", 1e-3}, {"u", 1e-6}, {"\xC2\xB5", 1e-6}, {"n", 1e-9}, {"p", 1e-12},
        {"c", 1e-2}, {"d", 1e-1}, {"Ki", 1024.0}, {"Mi", 1048576.0}, {"Gi", 1073741824.0}, {"Ti", 1099511627776.0},
    };
    auto it = m.find(p);
    if (it == m.end()) return false;
    s = it->second;
    return true;
}

bool lookup(const std::string& name, UnitValue& out) {
    const auto& t = table();
    auto it = t.find(name);
    if (it != t.end()) { out.dim = it->second.dim; out.scale = it->second.scale; return true; }
    for (size_t k = 1; k <= 2 && k < name.size(); ++k) {   // prefix + unit
        double ps = 0;
        auto u = t.find(name.substr(k));
        if (u != t.end() && u->second.prefixable && prefix_scale(name.substr(0, k), ps)) {
            // a binary prefix only makes sense for information units
            if (k == 2 && !(u->second.dim == D(0, 0, 0, 0, 0, 0, 0, 1))) continue;
            out.dim = u->second.dim; out.scale = u->second.scale * ps; return true;
        }
    }
    return false;
}

Dimension combine(const Dimension& a, const Dimension& b, int sign) {
    Dimension r;
    for (int i = 0; i < 9; ++i) r.e[i] = a.e[i] + sign * b.e[i];
    return r;
}

struct UnitParser {
    const std::string& s;
    size_t i = 0;
    std::string error;
    explicit UnitParser(const std::string& t) : s(t) {}
    void ws() { while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i; }
    bool parse_expr(UnitValue& out) {
        UnitValue v;
        if (!parse_power(v)) return false;
        while (true) {
            ws();
            if (i >= s.size() || s[i] == ')') break;
            int sign = 1;
            if (s[i] == '*') { ++i; }
            else if (s[i] == '/') { sign = -1; ++i; }
            else if (std::isalpha(static_cast<unsigned char>(s[i])) || static_cast<unsigned char>(s[i]) >= 0x80 || std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '(') {}   // juxtaposition
            else { error = std::string("unexpected '") + s[i] + "'"; return false; }
            UnitValue r;
            if (!parse_power(r)) return false;
            v.dim = combine(v.dim, r.dim, sign);
            v.scale = sign > 0 ? v.scale * r.scale : v.scale / r.scale;
        }
        out = v;
        return true;
    }
    bool parse_power(UnitValue& out) {
        UnitValue base;
        if (!parse_atom(base)) return false;
        ws();
        if (i < s.size() && (s[i] == '^' || (s[i] == '*' && i + 1 < s.size() && s[i + 1] == '*'))) {
            i += s[i] == '^' ? 1 : 2;
            ws();
            bool neg = false;
            if (i < s.size() && s[i] == '-') { neg = true; ++i; }
            else if (i < s.size() && s[i] == '+') ++i;
            if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) { error = "an exponent must be an integer"; return false; }
            int e = 0;
            while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) { e = e * 10 + (s[i] - '0'); ++i; if (e > 64) { error = "exponent too large"; return false; } }
            if (neg) e = -e;
            for (int k = 0; k < 9; ++k) base.dim.e[k] *= e;
            base.scale = std::pow(base.scale, e);
        }
        out = base;
        return true;
    }
    bool parse_atom(UnitValue& out) {
        ws();
        if (i >= s.size()) { error = "unexpected end"; return false; }
        if (s[i] == '(') {
            ++i;
            UnitValue v;
            if (!parse_expr(v)) return false;
            ws();
            if (i >= s.size() || s[i] != ')') { error = "missing ')'"; return false; }
            ++i;
            out = v;
            return true;
        }
        if (std::isdigit(static_cast<unsigned char>(s[i]))) {   // a pure number is a scale factor ("1000 m")
            size_t j = i;
            while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])) || s[j] == '.')) ++j;
            out.scale = std::strtod(s.substr(i, j - i).c_str(), nullptr);
            out.dim = Dimension();
            i = j;
            return true;
        }
        size_t j = i;
        while (j < s.size() && (std::isalpha(static_cast<unsigned char>(s[j])) || static_cast<unsigned char>(s[j]) >= 0x80)) ++j;
        if (j == i) { error = std::string("unexpected '") + s[i] + "'"; return false; }
        const std::string name = s.substr(i, j - i);
        UnitValue u;
        if (!lookup(name, u)) { error = "unknown unit '" + name + "'"; return false; }
        out = u;
        i = j;
        return true;
    }
};

} // namespace

std::string Dimension::to_string() const {
    std::ostringstream o;
    bool first = true;
    for (int i = 0; i < 8; ++i) {
        if (e[i] == 0) continue;
        o << (first ? "" : " ") << kDimNames[i];
        if (e[i] != 1) o << "^" << e[i];
        first = false;
    }
    return first ? "1 (dimensionless)" : o.str();
}

UnitValue UnitEngine::parse(const std::string& text) {
    UnitValue out;
    if (text.size() > 512) { out.error = "unit expression too long"; return out; }
    UnitParser p(text);
    UnitValue v;
    if (!p.parse_expr(v)) { out.error = p.error; return out; }
    p.ws();
    if (p.i < text.size()) { out.error = "unexpected trailing text"; return out; }
    v.ok = true;
    return v;
}

namespace {
VerifyResult unit_result(VerifyState st, const std::string& expl, const std::string& norm) {
    VerifyResult r;
    r.state = st;
    r.explanation = expl;
    if (st == VerifyState::kUnknown) r.diagnostic = expl;
    r.provenance.normalized_expression = norm;
    r.provenance.algorithm = "dimensional analysis (SI base exponents + bits)";
    r.provenance.backend_version = MathVerifier::version();
    r.provenance.ir_version = MathVerifier::ir_version();
    return r;
}
} // namespace

VerifyResult UnitEngine::verify_equation(const std::string& eq) {
    size_t p = eq.find("==");
    size_t len = 2;
    if (p == std::string::npos) { p = eq.find('='); len = 1; }
    if (p == std::string::npos) return unit_result(VerifyState::kUnknown, "not an equation", eq);
    const UnitValue a = parse(eq.substr(0, p)), b = parse(eq.substr(p + len));
    if (!a.ok) return unit_result(VerifyState::kUnknown, "left side: " + a.error + " (units are never guessed)", eq);
    if (!b.ok) return unit_result(VerifyState::kUnknown, "right side: " + b.error + " (units are never guessed)", eq);
    const std::string norm = a.dim.to_string() + " == " + b.dim.to_string();
    if (!(a.dim == b.dim)) return unit_result(VerifyState::kContradicted, "dimensionally inconsistent: [" + a.dim.to_string() + "] versus [" + b.dim.to_string() + "]", norm);
    if (std::fabs(a.scale - b.scale) > 1e-9 * std::fabs(a.scale)) return unit_result(VerifyState::kContradicted, "same dimension, different magnitude (factor " + std::to_string(a.scale / b.scale) + ")", norm);
    return unit_result(VerifyState::kVerified, "dimensions and magnitudes agree: [" + a.dim.to_string() + "]", norm);
}

VerifyResult UnitEngine::verify_convertible(const std::string& from, const std::string& to) {
    const UnitValue a = parse(from), b = parse(to);
    if (!a.ok) return unit_result(VerifyState::kUnknown, "from: " + a.error, from + " -> " + to);
    if (!b.ok) return unit_result(VerifyState::kUnknown, "to: " + b.error, from + " -> " + to);
    if (a.dim == b.dim) return unit_result(VerifyState::kVerified, "convertible (factor " + std::to_string(a.scale / b.scale) + ")", from + " -> " + to);
    return unit_result(VerifyState::kContradicted, "not convertible: [" + a.dim.to_string() + "] versus [" + b.dim.to_string() + "]", from + " -> " + to);
}

} // namespace strata::math
