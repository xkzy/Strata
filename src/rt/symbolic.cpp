// src/rt/symbolic.cpp - symbolic claims in prose: finding them and verifying them on the CAS
#include "strata/rt/symbolic.hpp"

#include "strata/math/cas/engine.hpp"

#include <algorithm>
#include <cctype>
#include <set>
#include <sstream>

namespace strata::rt {

namespace {

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

struct Word {
    std::string raw;     // as written
    std::string bare;    // lower case, trailing sentence punctuation removed
};

std::string strip_punct(const std::string& s) {
    size_t e = s.size();
    while (e > 0 && (s[e - 1] == '.' || s[e - 1] == ',' || s[e - 1] == ';' || s[e - 1] == ':' || s[e - 1] == '!' || s[e - 1] == '?')) --e;
    return s.substr(0, e);
}

std::vector<Word> split_words(const std::string& s) {
    std::vector<Word> out;
    std::istringstream in(s);
    std::string w;
    while (in >> w) out.push_back({w, lower(strip_punct(w))});
    return out;
}

// Does the word read as part of a mathematical expression (as opposed to prose)?
bool mathy(const Word& w) {
    const std::string& b = w.bare;
    if (b.empty()) return false;
    for (unsigned char c : b) if (std::isdigit(c) || c == '+' || c == '-' || c == '*' || c == '/' || c == '^' || c == '(' || c == ')' || c == '=') return true;
    if (b.size() == 1 && std::isalpha(static_cast<unsigned char>(b[0]))) return b != "a" && b != "i";
    static const std::set<std::string> fns = {"sin", "cos", "tan", "log", "ln", "exp", "sqrt", "pi", "dx", "dy", "dt"};
    return fns.count(b) > 0;
}

// Words from `i` on that read as an expression. `stop_eq`: an '=' word ends the span.
std::string span_after(const std::vector<Word>& w, size_t i, size_t* next, bool stop_eq) {
    std::string out;
    size_t j = i;
    while (j < w.size() && j - i < 40 && mathy(w[j]) && !(stop_eq && w[j].bare == "=")) {
        if (!out.empty()) out += ' ';
        out += (j + 1 == w.size() || !mathy(w[j + 1]) || (stop_eq && w[j + 1].bare == "=")) ? strip_punct(w[j].raw) : w[j].raw;
        ++j;
    }
    if (next) *next = j;
    return out;
}

std::string span_before(const std::vector<Word>& w, size_t end) {
    size_t b = end;
    while (b > 0 && end - b < 40 && mathy(w[b - 1])) --b;
    std::string out;
    for (size_t k = b; k < end; ++k) { if (!out.empty()) out += ' '; out += w[k].raw; }
    return out;
}

bool has_variable(const std::string& expr) {
    for (size_t i = 0; i < expr.size(); ++i) {
        if (!std::isalpha(static_cast<unsigned char>(expr[i]))) continue;
        size_t j = i;
        while (j < expr.size() && std::isalpha(static_cast<unsigned char>(expr[j]))) ++j;
        const std::string name = lower(expr.substr(i, j - i));
        static const std::set<std::string> fns = {"sin", "cos", "tan", "log", "ln", "exp", "sqrt", "pi", "e", "dx", "dy", "dt"};
        if (!fns.count(name)) return true;
        i = j;
    }
    return false;
}

bool parses(const std::string& e) {
    if (e.empty() || e.size() > 200) return false;
    try { strata::math::cas::parse(e); return true; } catch (...) { return false; }
}

// "x = 5" / "x = 2 and x = 3" -> "x = 5" / "x = 2; x = 3"
bool read_assignments(const std::vector<Word>& w, size_t i, std::string& out, std::string& var) {
    std::vector<std::string> parts;
    size_t j = i;
    while (j < w.size()) {
        size_t next = j;
        std::string a = span_after(w, j, &next, false);
        if (a.empty() || a.find('=') == std::string::npos) break;
        // "x = 5": a variable, '=', a number-like expression
        const size_t eq = a.find('=');
        std::string lhs = a.substr(0, eq), rhs = a.substr(eq + 1);
        while (!rhs.empty() && rhs.front() == ' ') rhs.erase(rhs.begin());
        lhs.erase(std::remove(lhs.begin(), lhs.end(), ' '), lhs.end());
        if (lhs.size() != 1 || !std::isalpha(static_cast<unsigned char>(lhs[0])) || !parses(rhs) || has_variable(rhs)) return false;
        if (var.empty()) var = lhs; else if (var != lhs) return false;
        parts.push_back(lhs + " = " + rhs);
        j = next;
        if (j < w.size() && (w[j].bare == "and" || w[j].bare == "or")) { ++j; continue; }
        break;
    }
    if (parts.empty()) return false;
    for (size_t k = 0; k < parts.size(); ++k) out += (k ? "; " : "") + parts[k];
    return true;
}

} // namespace

std::vector<SymbolicSpec> find_symbolic_claims(const std::string& sentence) {
    std::vector<SymbolicSpec> out;
    if (sentence.size() > 1500) return out;
    const std::vector<Word> w = split_words(sentence);
    auto at = [&](size_t i, const char* word) { return i < w.size() && w[i].bare == word; };
    auto add = [&](SymbolicSpec s) {
        for (const auto& o : out) if (o.kind == s.kind && o.f == s.f && o.g == s.g) return;
        out.push_back(std::move(s));
    };

    for (size_t i = 0; i < w.size(); ++i) {
        // ---- "the determinant / rank / trace of [[1, 2], [3, 4]] is D" ----
        if ((at(i, "determinant") || at(i, "rank") || at(i, "trace")) && at(i + 1, "of")) {
            const std::string head = at(i, "determinant") ? "Det" : at(i, "rank") ? "Rank" : "Tr";
            size_t j = 0;
            std::string m = span_after(w, i + 2, &j, false);
            if (j < w.size() && (at(j, "is") || at(j, "equals") || at(j, "="))) {
                size_t k = 0;
                std::string d = span_after(w, j + 1, &k, false);
                if (m.rfind("[[", 0) == 0 && !d.empty() && parses(m) && parses(d))
                    add({"matfn", m, d, head, "the " + w[i].bare + " of " + m + " is " + d});
            }
            continue;
        }
        // ---- "[[1, 2], [3, 4]] is invertible / singular" ----
        if ((at(i, "invertible") || at(i, "singular") || at(i, "non-singular") || at(i, "nonsingular")) && i >= 2 && at(i - 1, "is")) {
            const std::string m = span_before(w, i - 1);
            if (m.rfind("[[", 0) == 0 && parses(m)) {
                const std::string want = at(i, "singular") ? "singular" : "invertible";
                add({"invertible", m, want, "", m + " is " + w[i].bare});
            }
            continue;
        }
        // ---- "the limit of F as x approaches A is L" ----
        if (at(i, "limit") && at(i + 1, "of")) {
            size_t j = 0;
            std::string f = span_after(w, i + 2, &j, false);
            auto inf = [](std::string t) {
                for (const char* a : {"infinity", "\xE2\x88\x9E"}) { size_t p = t.find(a); if (p != std::string::npos) t.replace(p, std::string(a).size(), "Infinity"); }
                return t;
            };
            if (at(j, "as") && j + 2 < w.size() && w[j + 1].bare.size() == 1 && std::isalpha(static_cast<unsigned char>(w[j + 1].bare[0]))) {
                const std::string var = w[j + 1].bare;
                j += 2;
                if (at(j, "approaches") || at(j, "tends") || at(j, "goes") || at(j, "->") || at(j, "\xE2\x86\x92")) ++j; else continue;
                if (at(j, "to")) ++j;
                std::string a;
                if (j < w.size() && (w[j].bare == "infinity" || w[j].bare == "-infinity" || w[j].bare == "\xE2\x88\x9E" || w[j].bare == "-\xE2\x88\x9E")) { a = inf(w[j].bare); ++j; }
                else a = span_after(w, j, &j, false);
                if (!(at(j, "is") || at(j, "equals") || at(j, "="))) continue;
                ++j;
                std::string l;
                if (j < w.size() && (w[j].bare == "infinity" || w[j].bare == "-infinity")) l = inf(w[j].bare);
                else { size_t k = 0; l = span_after(w, j, &k, false); }
                while (!l.empty() && l.back() == ' ') l.pop_back();
                if (!f.empty() && !a.empty() && !l.empty() && parses(f) && parses(a) && parses(l))
                    add({"limit", f, l, var + "\n" + a, "the limit of " + f + " as " + var + " -> " + a + " is " + l});
            }
            continue;
        }
        // ---- derivative / antiderivative: "the derivative of F [with respect to v] is G" ----
        size_t fstart = 0;
        std::string kind;
        if (at(i, "derivative") && at(i + 1, "of")) { kind = "derivative"; fstart = i + 2; }
        else if ((at(i, "antiderivative") || at(i, "integral")) && at(i + 1, "of")) { kind = "integral"; fstart = i + 2; }
        else if (w[i].bare == "d/dx") { kind = "derivative"; fstart = i + 1; }
        if (!kind.empty()) {
            size_t j = 0;
            std::string f = span_after(w, fstart, &j, true);
            std::string var;
            if (at(j, "with") && at(j + 1, "respect") && at(j + 2, "to") && j + 3 < w.size()) { var = w[j + 3].bare; j += 4; }
            if (w[i].bare == "d/dx") var = "x";
            if (j < w.size() && (at(j, "is") || at(j, "equals") || at(j, "=") || at(j, "are"))) {
                ++j;
                size_t k = 0;
                std::string g = span_after(w, j, &k, false);
                // an integral is only determined up to a constant, and "dx" closes the integrand
                if (kind == "integral") {
                    std::string gl = g;
                    for (const char* c : {"+ c", "+c", "+ C", "+C"}) { size_t p = gl.rfind(c); if (p != std::string::npos && p + std::string(c).size() == gl.size()) gl = gl.substr(0, p); }
                    g = gl;
                    for (const char* d : {" dx", " dy", " dt"}) { size_t p = f.rfind(d); if (p != std::string::npos && p + 3 == f.size()) { var = var.empty() ? std::string(1, d[2]) : var; f = f.substr(0, p); } }
                }
                while (!g.empty() && g.back() == ' ') g.pop_back();
                if (!f.empty() && !g.empty() && has_variable(f) && parses(f) && parses(g)) {
                    if (var.size() > 1) var.clear();
                    add({kind, f, g, var, (kind == "derivative" ? "d/d" + (var.empty() ? std::string("x") : var) + " of " : "integral of ") + f + " is " + g});
                }
            }
            continue;
        }
        // ---- "F expands to G" / "simplifies to" / "factors as|into|to" / "equals" / "is equal to" ----
        size_t kwlen = 0;
        std::string k2;
        if (at(i, "expands") && at(i + 1, "to")) { k2 = "expand"; kwlen = 2; }
        else if (at(i, "simplifies") && at(i + 1, "to")) { k2 = "simplify"; kwlen = 2; }
        else if (at(i, "factors") && (at(i + 1, "as") || at(i + 1, "into") || at(i + 1, "to"))) { k2 = "factor"; kwlen = 2; }
        else if (at(i, "equals")) { k2 = "equals"; kwlen = 1; }
        else if (at(i, "is") && at(i + 1, "equal") && at(i + 2, "to")) { k2 = "equals"; kwlen = 3; }
        if (!k2.empty() && i > 0) {
            std::string f = span_before(w, i);
            size_t next = 0;
            std::string g = span_after(w, i + kwlen, &next, false);
            if (!f.empty() && !g.empty() && f.find('=') == std::string::npos && g.find('=') == std::string::npos && (has_variable(f) || has_variable(g)) &&
                parses(f) && parses(g)) {
                add({k2, f, g, "", f + (k2 == "equals" ? " equals " : k2 == "expand" ? " expands to " : k2 == "simplify" ? " simplifies to " : " factors as ") + g});
            }
            continue;
        }
        // ---- "the solution of EQ is x = a [and x = b]" ----
        if ((at(i, "solution") || at(i, "solutions")) && (at(i + 1, "of") || at(i + 1, "to"))) {
            size_t j = 0;
            std::string eq = span_after(w, i + 2, &j, false);
            const size_t e = eq.find('=');
            if (e == std::string::npos || eq.find('=', e + 1) != std::string::npos) continue;
            if (!(at(j, "is") || at(j, "are"))) continue;
            std::string answers, var;
            if (!read_assignments(w, j + 1, answers, var)) continue;
            if (!parses(eq.substr(0, e)) || !parses(eq.substr(e + 1)) || !has_variable(eq)) continue;
            add({"solve", eq, answers, var, "the solution of " + eq + " is " + answers});
        }
    }
    return out;
}

std::string encode_symbolic(const SymbolicSpec& s) { return s.kind + "\n" + s.f + "\n" + s.g + "\n" + s.var; }

bool decode_symbolic(const std::string& text, SymbolicSpec& out) {
    std::vector<std::string> parts;
    size_t b = 0;
    for (int k = 0; k < 3; ++k) {
        const size_t e = text.find('\n', b);
        if (e == std::string::npos) return false;
        parts.push_back(text.substr(b, e - b));
        b = e + 1;
    }
    parts.push_back(text.substr(b));
    out.kind = parts[0]; out.f = parts[1]; out.g = parts[2]; out.var = parts[3];
    return !out.kind.empty();
}

// ---------------------------------------------------------------------------------------------------------------
// verification
// ---------------------------------------------------------------------------------------------------------------
namespace {
using namespace strata::math::cas;

std::string first_variable(const Expr& e) {
    std::vector<std::string> syms;
    free_symbols(e, syms);
    for (const auto& s : syms) if (s == "x") return s;
    return syms.empty() ? "x" : syms.front();
}

enum class Eq { kEqual, kDiffer, kUnknown };

// Is F - G identically zero? kDiffer comes with a numeric witness; kUnknown when neither could be shown.
Eq equivalent(Engine& en, const Expr& f, const Expr& g, std::string& witness) {
    Expr d = en.eval(apply2("Plus", f, apply2("Times", minus_one(), g)));
    auto zero = [](const Expr& e) { return e->is_number() && e->q.is_zero(); };
    if (zero(d)) return Eq::kEqual;
    Expr x = en.expand(d);
    if (zero(x)) return Eq::kEqual;
    try { if (zero(en.together(d))) return Eq::kEqual; } catch (const CasError&) {}
    try { if (zero(en.simplify(d))) return Eq::kEqual; } catch (const CasError&) {}
    if (en.find_nonzero_witness(d, witness)) return Eq::kDiffer;
    return Eq::kUnknown;
}

// "x = 2; x = 3" -> the set of values (as printed exact numbers)
std::set<std::string> claimed_values(Engine& en, const std::string& g) {
    std::set<std::string> vals;
    std::istringstream in(g);
    std::string part;
    while (std::getline(in, part, ';')) {
        const size_t eq = part.find('=');
        if (eq == std::string::npos) continue;
        vals.insert(to_string(en.eval(parse(part.substr(eq + 1)))));
    }
    return vals;
}
} // namespace

VerifyOutcome SymbolicVerifier::verify(const Claim& claim, const EvidenceSet&, Clock::time_point deadline) {
    VerifyOutcome out;
    out.verifier = name();
    out.authoritative = true;
    SymbolicSpec s;
    if (!decode_symbolic(claim.formal.expression, s)) { out.explanation = "not a symbolic statement"; return out; }
    try {
        Budget budget;
        const double remaining = std::chrono::duration<double, std::milli>(deadline - Clock::now()).count();
        budget.timeout_ms = std::max(5.0, std::min(budget.timeout_ms, remaining));
        Engine en(budget);
        const std::string stmt = "\"" + claim.subject + "\"";
        std::string witness;

        if (s.kind == "solve") {
            const size_t e = s.f.find('=');
            Expr eq = apply2("Equal", parse(s.f.substr(0, e)), parse(s.f.substr(e + 1)));
            Expr sol = en.solve(eq, {s.var});
            std::set<std::string> truth;
            for (const auto& set : sol->args) {
                if (!set->has_head("List") || set->args.size() != 1 || !set->args[0]->has_head("Rule", 2)) { out.explanation = "solution set not in a simple form"; return out; }
                truth.insert(to_string(set->args[0]->args[1]));
            }
            const std::set<std::string> claimed = claimed_values(en, s.g);
            if (truth == claimed) { out.state = VerificationState::kVerified; out.explanation = stmt + " (solved exactly)"; return out; }
            std::string t;
            for (const auto& v : truth) t += (t.empty() ? "" : ", ") + s.var + " = " + v;
            out.state = VerificationState::kContradicted;
            out.corrected_value = t.empty() ? "no solution" : t;
            out.explanation = "the solutions of " + s.f + " are " + out.corrected_value + " (solved exactly), not " + s.g;
            return out;
        }

        if (s.kind == "matfn") {
            Expr truth = en.eval(app(s.var, {parse(s.f)}));
            if (truth->has_head(s.var.c_str())) { out.explanation = "could not evaluate it"; return out; }
            std::string w2;
            if (equivalent(en, truth, parse(s.g), w2) == Eq::kEqual) { out.state = VerificationState::kVerified; out.explanation = stmt + " (computed exactly)"; return out; }
            out.state = VerificationState::kContradicted;
            out.corrected_value = to_string(truth);
            out.explanation = stmt + " is false: the value is " + out.corrected_value + " (computed exactly)";
            return out;
        }

        if (s.kind == "invertible") {
            Expr d = en.eval(app("Det", {parse(s.f)}));
            if (!d->is_number()) { out.explanation = "the determinant is not a number: invertibility depends on the entries"; return out; }
            const bool invertible = !d->q.is_zero();
            const bool claim_invertible = s.g == "invertible";
            out.state = invertible == claim_invertible ? VerificationState::kVerified : VerificationState::kContradicted;
            out.explanation = stmt + (out.state == VerificationState::kVerified ? " (determinant " : " is false: the determinant is ") + to_string(d) + ")";
            if (out.state == VerificationState::kContradicted) out.corrected_value = invertible ? "invertible" : "singular";
            return out;
        }

        if (s.kind == "limit") {
            const size_t nl = s.var.find('\n');
            const std::string var = s.var.substr(0, nl);
            Expr lim = en.limit(parse(s.f), var, parse(s.var.substr(nl + 1)));
            const std::string have = to_string(lim);
            if (lim->has_head("Limit")) { out.explanation = "could not evaluate the limit"; return out; }
            if (have == to_string(en.eval(parse(s.g)))) { out.state = VerificationState::kVerified; out.explanation = stmt + " (limit computed exactly)"; return out; }
            std::string w2;
            if (equivalent(en, lim, parse(s.g), w2) == Eq::kEqual) { out.state = VerificationState::kVerified; out.explanation = stmt + " (limit computed exactly)"; return out; }
            out.state = VerificationState::kContradicted;
            out.corrected_value = have;
            out.explanation = stmt + " is false: the limit is " + have + " (computed exactly)";
            return out;
        }

        Expr f = parse(s.f), g = parse(s.g);
        std::string var = s.var.empty() ? first_variable(f) : s.var;
        Expr truth, compare_a, compare_b;
        if (s.kind == "derivative") { truth = en.diff(f, var); compare_a = truth; compare_b = g; }
        else if (s.kind == "integral") { compare_a = en.diff(g, var); compare_b = f; }   // an antiderivative: its derivative is the integrand
        else { compare_a = f; compare_b = g; }
        switch (equivalent(en, compare_a, compare_b, witness)) {
            case Eq::kEqual:
                out.state = VerificationState::kVerified;
                out.explanation = stmt + " (checked symbolically)";
                return out;
            case Eq::kUnknown:
                out.state = VerificationState::kUnknown;
                out.explanation = "could not decide whether " + stmt + " holds";
                return out;
            case Eq::kDiffer: break;
        }
        out.state = VerificationState::kContradicted;
        if (s.kind == "derivative") out.corrected_value = to_string(truth);
        else if (s.kind == "expand") out.corrected_value = to_string(en.expand(f));
        else if (s.kind == "simplify") out.corrected_value = to_string(en.simplify(f));
        else if (s.kind == "factor") out.corrected_value = to_string(en.factor(f));
        else if (s.kind == "integral") { try { out.corrected_value = to_string(en.integrate(f, var)) + " + C"; } catch (const CasError&) {} }
        out.explanation = stmt + " is false: the two sides differ (" + witness + ")" + (out.corrected_value.empty() ? "" : "; the correct result is " + out.corrected_value);
        return out;
    } catch (const CasLimitError& e) {
        out.state = VerificationState::kUnknown; out.failed = true; out.explanation = std::string("resource limit: ") + e.what();
    } catch (const CasError& e) {
        out.state = VerificationState::kUnknown; out.explanation = e.what();
    }
    return out;
}

} // namespace strata::rt
