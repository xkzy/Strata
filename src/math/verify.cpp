// SPDX-License-Identifier: GPL-3.0-or-later
// src/math/verify.cpp - deterministic verification with explicit assumptions; units and dimensions (see verify.hpp)
#include "strata/math/verify.hpp"

#include "strata/math/cas/engine.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <set>
#include <sstream>

namespace strata::math {

using namespace cas;

namespace {

std::string lower_s(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
std::string trim_s(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::string content_hash(const std::string& s) {   // 128-bit FNV-style address (not cryptographic)
    uint64_t a = 14695981039346656037ULL, b = 0x9E3779B97F4A7C15ULL;
    for (unsigned char c : s) { a = (a ^ c) * 1099511628211ULL; b = (b ^ c) * 0x100000001B3ULL + 0x9E3779B97F4A7C15ULL; }
    char buf[40];
    std::snprintf(buf, sizeof buf, "%016llx%016llx", static_cast<unsigned long long>(a), static_cast<unsigned long long>(b ^ (a >> 7)));
    return buf;
}

int64_t now_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }

bool is_constant_name(const std::string& n) { return n == "Pi" || n == "E" || n == "I" || n == "Infinity"; }

std::vector<std::string> variables_of(const Expr& e) {
    std::vector<std::string> s;
    free_symbols(e, s);
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());
    s.erase(std::remove_if(s.begin(), s.end(), is_constant_name), s.end());
    return s;
}

// "a = b" with a single '=' becomes "==" (the CAS reads == as equality)
std::string normalize_equals(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '=') {
            const bool prev_op = i > 0 && (s[i - 1] == '<' || s[i - 1] == '>' || s[i - 1] == '!' || s[i - 1] == '=');
            const bool next_eq = i + 1 < s.size() && s[i + 1] == '=';
            if (!prev_op && !next_eq) { out += "=="; continue; }
        }
        out.push_back(c);
    }
    return out;
}

} // namespace

// =====================================================================================================================
// assumptions
// =====================================================================================================================
bool VarFacts::admits(double v) const {
    if (integer && std::floor(v) != v) return false;
    if (nonzero && v == 0) return false;
    if (has_lo && (lo_strict ? !(v > lo) : !(v >= lo))) return false;
    if (has_hi && (hi_strict ? !(v < hi) : !(v <= hi))) return false;
    return true;
}

const VarFacts* Assumptions::find(const std::string& var) const {
    auto it = facts_.find(var);
    return it == facts_.end() ? nullptr : &it->second;
}

bool Assumptions::add(const std::string& clause_in, std::string* error) {
    const std::string clause = trim_s(clause_in);
    if (clause.empty()) return true;
    auto fail = [&](const std::string& m) { if (error) *error = m + ": '" + clause + "'"; return false; };
    // set membership: "x in Reals", "x ∈ Integer", "x is real", "n integer"
    const std::string lc = lower_s(clause);
    static const char* kSeps[] = {" in ", " is ", "\xE2\x88\x88", " : "};
    std::string var, what;
    for (const char* sep : kSeps) {
        const size_t p = lc.find(sep);
        if (p != std::string::npos) { var = trim_s(clause.substr(0, p)); what = trim_s(lc.substr(p + std::string(sep).size())); break; }
    }
    if (var.empty()) {   // "n integer", "x real", "x positive"
        const size_t sp = lc.find_last_of(' ');
        if (sp != std::string::npos && lc.find_first_of("<>=!") == std::string::npos) { var = trim_s(clause.substr(0, sp)); what = trim_s(lc.substr(sp + 1)); }
    }
    if (!var.empty() && !what.empty() && lc.find_first_of("<>=!") == std::string::npos) {
        if (var.empty() || !std::isalpha(static_cast<unsigned char>(var[0]))) return fail("assumption names no variable");
        VarFacts& f = facts_[var];
        if (what == "real" || what == "reals" || what == "\xE2\x84\x9D") f.real = true;
        else if (what == "integer" || what == "integers" || what == "int" || what == "\xE2\x84\xA4") { f.integer = true; f.real = true; }
        else if (what == "positive") { f.real = true; f.has_lo = true; f.lo = 0; f.lo_strict = true; }
        else if (what == "nonnegative" || what == "non-negative") { f.real = true; f.has_lo = true; f.lo = 0; f.lo_strict = false; }
        else if (what == "negative") { f.real = true; f.has_hi = true; f.hi = 0; f.hi_strict = true; }
        else if (what == "nonzero") f.nonzero = true;
        else return fail("unknown assumption");
        return true;
    }
    // relation: "x > 0", "0 < x", "x != 0", "x >= 1/2"
    static const char* kOps[] = {">=", "<=", "!=", ">", "<", "=="};
    for (const char* op : kOps) {
        const size_t p = clause.find(op);
        if (p == std::string::npos) continue;
        std::string l = trim_s(clause.substr(0, p)), r = trim_s(clause.substr(p + std::string(op).size()));
        std::string o = op;
        auto numeric = [](const std::string& t, double& v) { try { Engine en{Budget{}}; return en.numeric_value(en.eval(cas::parse(t)), v); } catch (const std::exception&) { return false; } };
        double val = 0;
        bool var_left = !l.empty() && std::isalpha(static_cast<unsigned char>(l[0])) && numeric(r, val) && l.find_first_of(" +-*/^()") == std::string::npos;
        if (!var_left) {
            if (!r.empty() && std::isalpha(static_cast<unsigned char>(r[0])) && r.find_first_of(" +-*/^()") == std::string::npos && numeric(l, val)) {
                std::swap(l, r);
                o = o == ">=" ? "<=" : o == "<=" ? ">=" : o == ">" ? "<" : o == "<" ? ">" : o;
            } else return fail("only simple bounds on one variable are supported");
        }
        VarFacts& f = facts_[l];
        f.real = true;
        if (o == ">" || o == ">=") {
            if (!f.has_lo || val > f.lo || (val == f.lo && o == ">")) { f.has_lo = true; f.lo = val; f.lo_strict = o == ">"; }
        } else if (o == "<" || o == "<=") {
            if (!f.has_hi || val < f.hi || (val == f.hi && o == "<")) { f.has_hi = true; f.hi = val; f.hi_strict = o == "<"; }
        } else if (o == "!=") { if (val == 0) f.nonzero = true; else return fail("only x != 0 is supported"); }
        else { f.has_lo = f.has_hi = true; f.lo = f.hi = val; f.lo_strict = f.hi_strict = false; }
        return true;
    }
    return fail("unrecognized assumption");
}

bool Assumptions::parse(const std::string& text, Assumptions& out, std::string* error) {
    out = Assumptions();
    std::string t = text;
    // split on ',' ';' and the word "and"
    std::vector<std::string> parts;
    std::string cur;
    for (size_t i = 0; i < t.size(); ++i) {
        if (t[i] == ',' || t[i] == ';') { parts.push_back(cur); cur.clear(); continue; }
        if (i + 5 <= t.size() && lower_s(t.substr(i, 5)) == " and " ) { parts.push_back(cur); cur.clear(); i += 4; continue; }
        cur.push_back(t[i]);
    }
    parts.push_back(cur);
    for (const auto& p : parts) if (!out.add(p, error)) return false;
    return true;
}

std::string Assumptions::to_string() const {
    std::ostringstream o;
    bool first = true;
    for (const auto& [v, f] : facts_) {
        o << (first ? "" : "; ") << v << ":";
        first = false;
        if (f.integer) o << " integer";
        else if (f.real) o << " real";
        if (f.nonzero) o << " nonzero";
        if (f.has_lo) o << (f.lo_strict ? " >" : " >=") << f.lo;
        if (f.has_hi) o << (f.hi_strict ? " <" : " <=") << f.hi;
    }
    return o.str();
}

namespace {

// ---------------------------------------------------------------------------------------------------------------------
// assumption-aware rewriting: sqrt(x^2) is |x|, and x only when x >= 0 is known
// ---------------------------------------------------------------------------------------------------------------------
struct Rewriter {
    const Assumptions& as;
    std::set<std::string>& used;
    Expr go(const Expr& e) {
        if (!e->is_apply()) return e;
        std::vector<Expr> args;
        bool changed = false;
        for (const auto& a : e->args) { Expr n = go(a); changed = changed || n != a; args.push_back(n); }
        Expr cur = changed ? app(e->name, args) : e;
        if (cur->has_head("Sqrt", 1)) cur = app("Power", {cur->args[0], num(Rational(1) / Rational(2))});   // sqrt(f) is f^(1/2)
        // (s^k)^r with k even and r not an integer: |s|^(k r) for real s
        if (cur->has_head("Power", 2) && cur->args[0]->has_head("Power", 2) && cur->args[0]->args[0]->is_symbol() && cur->args[0]->args[1]->is_integer() &&
            cur->args[1]->is_number() && !cur->args[1]->q.is_integer()) {
            int64_t k = 0;
            if (cur->args[0]->args[1]->q.num.to_int64(k) && k % 2 == 0) {
                const std::string& v = cur->args[0]->args[0]->name;
                const VarFacts* f = as.find(v);
                if (f && f->real) {
                    const Rational ke = Rational(k) * cur->args[1]->q;
                    if (f->has_lo && f->lo >= 0) { used.insert(v + " >= 0"); return app("Power", {symbol(v), num(ke)}); }
                    if (f->has_hi && f->hi <= 0) { used.insert(v + " <= 0"); return app("Power", {app("Times", {minus_one(), symbol(v)}), num(ke)}); }
                    used.insert(v + " real");
                    return app("Power", {app("Abs", {symbol(v)}), num(ke)});
                }
            }
        }
        return cur;
    }
};

void denominators(const Expr& e, std::set<std::string>& out) {
    if (e->has_head("Power", 2) && e->args[1]->is_number() && e->args[1]->q.is_negative() && !variables_of(e->args[0]).empty()) {
        out.insert(to_string(e->args[0]) + " != 0");
    }
    for (const auto& a : e->args) denominators(a, out);
}

bool is_zero_expr(const Expr& e) { return e->is_number() && e->q.is_zero(); }

// F - G identically zero?  -1 unknown, 1 yes
bool provably_zero(Engine& en, const Expr& d) {
    if (is_zero_expr(d)) return true;
    try { if (is_zero_expr(en.expand(d))) return true; } catch (const CasError&) {}
    try { if (is_zero_expr(en.together(d))) return true; } catch (const CasError&) {}
    try { if (is_zero_expr(en.simplify(d))) return true; } catch (const CasError&) {}
    return false;
}

const std::vector<std::pair<int, int>>& sample_values() {
    static const std::vector<std::pair<int, int>> v = {{2, 1}, {-3, 1}, {1, 2}, {3, 1}, {-1, 2}, {5, 1}, {-2, 1}, {7, 3}, {1, 1}, {-1, 1}, {10, 1}, {3, 2}, {0, 1}, {2, 3}, {13, 1}, {-7, 1}, {5, 2}};
    return v;
}

// A point (assignment of admissible values) where `d` is clearly non-zero.
bool find_witness(Engine& en, const Expr& d, const std::vector<std::string>& vars, const Assumptions& as, std::string& witness, int& evaluated, double* out_value = nullptr) {
    evaluated = 0;
    const auto& vals = sample_values();
    std::vector<size_t> idx(vars.size(), 0);
    int tried = 0;
    // enumerate combinations in a fixed order (deterministic), bounded
    std::vector<size_t> counter(vars.size(), 0);
    while (tried < 300) {
        std::vector<std::pair<std::string, Expr>> sub;
        bool ok = true;
        std::string desc;
        for (size_t i = 0; i < vars.size(); ++i) {
            const auto& pv = vals[(counter[i] + i * 3) % vals.size()];
            const double v = static_cast<double>(pv.first) / pv.second;
            const VarFacts* f = as.find(vars[i]);
            if (f && !f->admits(v)) { ok = false; break; }
            sub.push_back({vars[i], num(Rational(pv.first) / Rational(pv.second))});
            desc += (desc.empty() ? "" : ", ") + vars[i] + " = " + (pv.second == 1 ? std::to_string(pv.first) : std::to_string(pv.first) + "/" + std::to_string(pv.second));
        }
        if (ok) {
            ++tried;
            try {
                Expr val = en.eval(Engine::substitute(d, sub));
                double dv = 0;
                bool nonzero = false;
                if (val->is_number()) { nonzero = !val->q.is_zero(); dv = val->q.num.to_double() / val->q.den.to_double(); }
                else if (en.numeric_value(val, dv)) nonzero = std::fabs(dv) > 1e-9 * (1.0 + std::fabs(dv));
                else continue;   // undefined here (pole, complex): not a usable point
                ++evaluated;
                if (nonzero) { witness = desc + " gives " + to_string(val); if (out_value) *out_value = dv; return true; }
            } catch (const CasError&) {}
        }
        // advance the counter
        size_t k = 0;
        while (k < counter.size() && ++counter[k] >= vals.size()) { counter[k] = 0; ++k; }
        if (k == counter.size()) break;
        if (vars.empty()) break;
    }
    return false;
}

void fill_provenance(VerifyResult& r, const std::string& normalized, const std::string& algorithm, const Assumptions& as, bool exact = true) {
    r.provenance.normalized_expression = normalized;
    r.provenance.expression_hash = content_hash(normalized + "|" + as.to_string());
    r.provenance.backend_version = MathVerifier::version();
    r.provenance.algorithm = algorithm;
    r.provenance.assumptions = as.to_string();
    r.provenance.input_hashes = {content_hash(normalized)};
    r.provenance.exact = exact;
    r.provenance.ir_version = MathVerifier::ir_version();
    r.provenance.timestamp_ms = now_ms();
}

VerifyResult unknown(const std::string& why, const std::string& normalized, const Assumptions& as) {
    VerifyResult r;
    r.state = VerifyState::kUnknown;
    r.explanation = why;
    r.diagnostic = why;
    fill_provenance(r, normalized, "none", as);
    return r;
}

// the core: compare two expressions. `universal` selects identity semantics (counterexample -> CONTRADICTED)
VerifyResult compare(const std::string& sa, const std::string& sb, const std::string& assumption_text, bool universal, bool want_not_equal) {
    Assumptions as;
    std::string err;
    if (!Assumptions::parse(assumption_text, as, &err)) return unknown("could not read the assumptions: " + err, sa + " = " + sb, as);
    const std::string norm_in = sa + " == " + sb;
    try {
        Engine en{Budget{}};
        Expr a = parse(sa), b = parse(sb);
        std::set<std::string> used;
        Rewriter rw{as, used};
        Expr ra = rw.go(a), rb = rw.go(b);
        Expr d = en.eval(app("Plus", {ra, app("Times", {minus_one(), rb})}));
        std::vector<std::string> vars = variables_of(d);
        const std::vector<std::string> vars_all = [&] { auto v1 = variables_of(a), v2 = variables_of(b); v1.insert(v1.end(), v2.begin(), v2.end()); std::sort(v1.begin(), v1.end()); v1.erase(std::unique(v1.begin(), v1.end()), v1.end()); return v1; }();
        VerifyResult r;
        const std::string normalized = to_string(en.eval(ra)) + " == " + to_string(en.eval(rb));
        std::set<std::string> conds;
        denominators(ra, conds);
        denominators(rb, conds);
        auto finish = [&](VerifyState st, const std::string& expl, const std::string& alg, const std::string& witness = "") {
            r.state = st;
            r.explanation = expl;
            r.witness = witness;
            r.assumptions_used.assign(used.begin(), used.end());
            if (st == VerifyState::kVerified) r.conditions.assign(conds.begin(), conds.end());
            fill_provenance(r, normalized, alg, as);
            return r;
        };
        auto flip = [&](VerifyState s) { return want_not_equal ? (s == VerifyState::kVerified ? VerifyState::kContradicted : s == VerifyState::kContradicted ? VerifyState::kVerified : s) : s; };

        if (is_zero_expr(d) || (!d->is_number() && provably_zero(en, d))) {
            return finish(flip(VerifyState::kVerified), std::string(want_not_equal ? "the two sides are identical" : "the two sides are identical") + (conds.empty() ? "" : " where both are defined"), "exact: expand / together / simplify");
        }
        if (d->is_number()) return finish(flip(VerifyState::kContradicted), "the two sides differ by the non-zero constant " + to_string(d), "exact: difference is a non-zero constant");
        if (vars.empty()) {   // no free variables: decide numerically only when the gap is far beyond rounding
            double dv = 0;
            if (en.numeric_value(d, dv)) {
                if (std::fabs(dv) > 1e-9) return finish(flip(VerifyState::kContradicted), "the two sides differ by about " + std::to_string(dv), "numeric evaluation of the exact difference (double precision)");
                return finish(VerifyState::kUnknown, "the sides agree to double precision but equality could not be proven exactly", "numeric evaluation (not a proof)");
            }
            return finish(VerifyState::kUnknown, "could not evaluate the difference", "none");
        }
        // free variables: look for an admissible counterexample
        std::string witness;
        int evaluated = 0;
        if (find_witness(en, d, vars_all.empty() ? vars : vars_all, as, witness, evaluated)) {
            if (universal) return finish(flip(VerifyState::kContradicted), "not an identity: the sides differ at a point " + std::string(as.empty() ? "" : "that satisfies the assumptions"), "counterexample search", witness);
            // equality as a claim: holds for some values only
            return finish(want_not_equal ? VerifyState::kUnknown : VerifyState::kUnknown, "the sides are not identical (they differ at some points); this holds only for particular values, so it is not verified as an identity", "counterexample search", witness);
        }
        if (evaluated == 0) return finish(VerifyState::kUnknown, "no admissible sample point could be evaluated (assumptions may exclude every sample)", "none");
        return finish(VerifyState::kUnknown, "agrees at every sample point but could not be proven symbolically; with unmet assumptions this stays unknown", "counterexample search (not a proof)");
    } catch (const CasLimitError& e) {
        return unknown(std::string("resource limit: ") + e.what(), norm_in, as);
    } catch (const CasError& e) {
        return unknown(std::string("unsupported or invalid: ") + e.what(), norm_in, as);
    }
}

// ---- inequalities ----
struct Seg { double lo, hi; bool lo_closed, hi_closed; };   // -inf/inf as +-HUGE_VAL

bool seg_from(const Expr& e, Engine& en, std::vector<Seg>& out) {
    auto val = [&](const Expr& v, double& x) { return en.numeric_value(v, x); };
    if (e->is_symbol_named("True")) { out.push_back({-HUGE_VAL, HUGE_VAL, false, false}); return true; }
    if (e->is_symbol_named("False")) return true;
    if (e->has_head("Or")) { for (const auto& a : e->args) if (!seg_from(a, en, out)) return false; return true; }
    Seg s{-HUGE_VAL, HUGE_VAL, false, false};
    auto one = [&](const Expr& c) -> bool {
        double x = 0;
        if (c->args.size() != 2 || !val(c->args[1], x)) return false;
        if (c->has_head("Less")) { s.hi = std::min(s.hi, x); s.hi_closed = false; }
        else if (c->has_head("LessEqual")) { if (x < s.hi) { s.hi = x; s.hi_closed = true; } }
        else if (c->has_head("Greater")) { s.lo = std::max(s.lo, x); s.lo_closed = false; }
        else if (c->has_head("GreaterEqual")) { if (x > s.lo) { s.lo = x; s.lo_closed = true; } }
        else if (c->has_head("Equal")) { s.lo = s.hi = x; s.lo_closed = s.hi_closed = true; }
        else return false;
        return true;
    };
    if (e->has_head("And")) { for (const auto& a : e->args) if (!one(a)) return false; }
    else if (!one(e)) return false;
    out.push_back(s);
    return true;
}

// is p (numerically) a root of the denominator?
bool poly_eval_double_pole(const Poly& d, double p) {
    double v = 0;
    for (size_t i = d.size(); i-- > 0;) v = v * p + d[i].num.to_double() / d[i].den.to_double();
    return std::fabs(v) < 1e-12;
}

bool in_segs(const std::vector<Seg>& segs, double p) {
    for (const auto& s : segs) {
        const bool lo_ok = s.lo == -HUGE_VAL || p > s.lo || (s.lo_closed && p == s.lo);
        const bool hi_ok = s.hi == HUGE_VAL || p < s.hi || (s.hi_closed && p == s.hi);
        if (lo_ok && hi_ok) return true;
    }
    return false;
}

} // namespace

// =====================================================================================================================
// public verification API
// =====================================================================================================================
VerifyResult MathVerifier::verify_equal(const std::string& a, const std::string& b, const std::string& as) const { return compare(a, b, as, false, false); }
VerifyResult MathVerifier::verify_not_equal(const std::string& a, const std::string& b, const std::string& as) const { return compare(a, b, as, false, true); }
VerifyResult MathVerifier::verify_identity(const std::string& a, const std::string& b, const std::string& as) const { return compare(a, b, as, true, false); }

VerifyResult MathVerifier::verify_equation(const std::string& equation, const std::string& as) const {
    const std::string eq = normalize_equals(equation);
    const size_t p = eq.find("==");
    Assumptions dummy;
    if (p == std::string::npos) return unknown("not an equation", equation, dummy);
    return compare(eq.substr(0, p), eq.substr(p + 2), as, false, false);
}

VerifyResult MathVerifier::verify_inequality(const std::string& relation, const std::string& assumption_text) const {
    Assumptions as;
    std::string err;
    if (!Assumptions::parse(assumption_text, as, &err)) return unknown("could not read the assumptions: " + err, relation, as);
    try {
        Engine en{Budget{}};
        Expr rel = parse(relation);
        const bool lt = rel->has_head("Less", 2), gt = rel->has_head("Greater", 2), le = rel->has_head("LessEqual", 2), ge = rel->has_head("GreaterEqual", 2);
        if (!(lt || gt || le || ge)) return unknown("not an inequality (use <, <=, >, >=)", relation, as);
        std::set<std::string> used;
        Rewriter rw{as, used};
        Expr f = en.eval(app("Plus", {rw.go(rel->args[0]), app("Times", {minus_one(), rw.go(rel->args[1])})}));   // f op 0
        VerifyResult r;
        const std::string normalized = to_string(f) + (lt ? " < 0" : gt ? " > 0" : le ? " <= 0" : " >= 0");
        auto finish = [&](VerifyState st, const std::string& expl, const std::string& alg, const std::string& witness = "") {
            r.state = st; r.explanation = expl; r.witness = witness;
            r.assumptions_used.assign(used.begin(), used.end());
            fill_provenance(r, normalized, alg, as);
            return r;
        };
        const std::vector<std::string> vars = variables_of(f);
        auto holds = [&](double v) { return lt ? v < 0 : gt ? v > 0 : le ? v <= 0 : v >= 0; };
        if (vars.empty()) {
            double v = 0;
            if (f->is_number()) { const bool ok = holds(f->q.num.to_double() / f->q.den.to_double()) && (f->q.is_zero() ? (le || ge) : true); if (f->q.is_zero()) return finish((le || ge) ? VerifyState::kVerified : VerifyState::kContradicted, "exact comparison", "exact"); return finish(ok ? VerifyState::kVerified : VerifyState::kContradicted, "exact comparison of rational numbers", "exact"); }
            if (en.numeric_value(f, v) && std::fabs(v) > 1e-9) return finish(holds(v) ? VerifyState::kVerified : VerifyState::kContradicted, "numeric comparison; the gap is far beyond rounding", "numeric (double precision)");
            return finish(VerifyState::kUnknown, "too close to call numerically", "numeric (not a proof)");
        }
        if (vars.size() == 1) {
            const std::string& x = vars[0];
            const VarFacts* fx = as.find(x);
            Poly n, d;
            if (en.rational_function(f, x, n, d)) {
                Expr red = en.eval(app("Reduce", {app(lt ? "Less" : gt ? "Greater" : le ? "LessEqual" : "GreaterEqual", {f, zero()}), symbol(x)}));
                std::vector<Seg> segs;
                if (seg_from(red, en, segs)) {
                    // candidate points: every boundary of the solution set and of the domain, midpoints, and points beyond the ends
                    std::vector<double> bounds;
                    for (const auto& s : segs) { if (s.lo != -HUGE_VAL) bounds.push_back(s.lo); if (s.hi != HUGE_VAL) bounds.push_back(s.hi); }
                    if (fx && fx->has_lo) bounds.push_back(fx->lo);
                    if (fx && fx->has_hi) bounds.push_back(fx->hi);
                    std::sort(bounds.begin(), bounds.end());
                    bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
                    std::vector<double> cands = bounds;
                    for (size_t i = 0; i + 1 < bounds.size(); ++i) cands.push_back((bounds[i] + bounds[i + 1]) / 2);
                    cands.push_back(bounds.empty() ? 0.0 : bounds.front() - 1);
                    cands.push_back(bounds.empty() ? 1.0 : bounds.back() + 1);
                    if (fx && fx->integer) {   // integers: test integer points around the boundaries
                        for (double b : bounds) for (int k = -2; k <= 2; ++k) cands.push_back(std::floor(b) + k);
                    }
                    for (double p : cands) {
                        if (fx && !fx->admits(p)) continue;
                        if (poly_eval_double_pole(d, p)) continue;
                        if (!in_segs(segs, p)) {
                            char buf[64];
                            std::snprintf(buf, sizeof buf, "%s = %g violates it", x.c_str(), p);
                            return finish(VerifyState::kContradicted, "the inequality fails at an admissible point", "interval analysis of the exact solution set", buf);
                        }
                    }
                    if (fx && fx->integer) return finish(VerifyState::kVerified, "holds at every admissible integer point (interval analysis)", "interval analysis of the exact solution set");
                    return finish(VerifyState::kVerified, "holds for every admissible " + x + " (exact interval analysis)", "interval analysis of the exact solution set");
                }
            }
        }
        // other shapes: only a counterexample can be established
        std::string witness;
        int evaluated = 0;
        double dv = 0;
        const auto& vals = sample_values();
        std::vector<std::pair<std::string, Expr>> sub;
        for (size_t t = 0; t < vals.size() * 4; ++t) {
            sub.clear();
            bool ok = true;
            std::string desc;
            for (size_t i = 0; i < vars.size(); ++i) {
                const auto& pv = vals[(t / (i + 1) + i * 5) % vals.size()];
                const double v = static_cast<double>(pv.first) / pv.second;
                const VarFacts* fv = as.find(vars[i]);
                if (fv && !fv->admits(v)) { ok = false; break; }
                sub.push_back({vars[i], num(Rational(pv.first) / Rational(pv.second))});
                desc += (desc.empty() ? "" : ", ") + vars[i] + " = " + std::to_string(v);
            }
            if (!ok) continue;
            try {
                Expr val = en.eval(Engine::substitute(f, sub));
                if (!en.numeric_value(val, dv)) continue;
                ++evaluated;
                if (!holds(dv)) return finish(VerifyState::kContradicted, "the inequality fails at an admissible point", "counterexample search", desc);
            } catch (const CasError&) {}
        }
        return finish(VerifyState::kUnknown, evaluated ? "holds at every sample point but was not proven for all values" : "no admissible sample point could be evaluated", "counterexample search (not a proof)");
    } catch (const CasLimitError& e) {
        return unknown(std::string("resource limit: ") + e.what(), relation, as);
    } catch (const CasError& e) {
        return unknown(std::string("unsupported or invalid: ") + e.what(), relation, as);
    }
}

VerifyResult MathVerifier::verify(const std::string& statement, const std::string& as) const {
    const std::string s = normalize_equals(statement);
    Assumptions dummy;
    try {
        Expr e = parse(s);
        if (e->has_head("Equal", 2)) return verify_equation(s, as);
        if (e->has_head("Unequal", 2)) { const size_t p = s.find("!="); return verify_not_equal(s.substr(0, p), s.substr(p + 2), as); }
        if (e->has_head("Less", 2) || e->has_head("Greater", 2) || e->has_head("LessEqual", 2) || e->has_head("GreaterEqual", 2)) return verify_inequality(s, as);
    } catch (const CasError& ex) {
        return unknown(std::string("could not parse: ") + ex.what(), statement, dummy);
    }
    return unknown("not an equation or an inequality", statement, dummy);
}

} // namespace strata::math
