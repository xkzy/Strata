// tests/math/test_math_verify.cpp - verification with assumptions, tri-state results, provenance, units
#undef NDEBUG
#include "strata/math/verify.hpp"

#include <cstdio>

using namespace strata::math;

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  FAIL: %s (line %d)\n", msg, __LINE__); ++failures; } } while (0)
static bool has(const std::vector<std::string>& v, const std::string& s) { for (auto& x : v) if (x == s) return true; return false; }

int main() {
    MathVerifier mv;

    std::printf("[1] exact equality\n");
    CHECK(mv.verify_equal("1/3 + 1/3 + 1/3", "1").state == VerifyState::kVerified, "1/3 three times is 1 (exact, not 0.999...)");
    CHECK(mv.verify_equal("sqrt(8)", "2*sqrt(2)").state == VerifyState::kVerified, "radicals stay exact");
    CHECK(mv.verify_equal("2^100", "1267650600228229401496703205376").state == VerifyState::kVerified, "big integers");
    CHECK(mv.verify_equal("2 + 2", "5").state == VerifyState::kContradicted, "2+2 is not 5");
    CHECK(mv.verify_equal("sin(x)^2 + cos(x)^2", "1").state == VerifyState::kVerified, "trig identity");
    CHECK(mv.verify_equal("(x+1)^2", "x^2 + 2*x + 1").state == VerifyState::kVerified, "expansion");
    CHECK(mv.verify_equal("x + 1", "x + 2").state == VerifyState::kContradicted, "differ by a non-zero constant: never equal");
    CHECK(mv.verify_not_equal("x + 1", "x + 2").state == VerifyState::kVerified, "not-equal for the same pair");
    CHECK(mv.verify_not_equal("1/3 + 1/3", "2/3").state == VerifyState::kContradicted, "not-equal of equal values");

    std::printf("[2] UNKNOWN is not FALSE\n");
    auto q = mv.verify_equal("x^2", "x");
    CHECK(q.state == VerifyState::kUnknown && !q.witness.empty(), "x^2 = x holds only for some x: UNKNOWN, with the point that shows it");
    CHECK(mv.verify_identity("x^2", "x").state == VerifyState::kContradicted, "as an identity it is false");
    CHECK(mv.verify_not_equal("x^2", "x").state == VerifyState::kUnknown, "and not-equal is not established either");

    std::printf("[3] assumptions are explicit and never implied\n");
    auto s0 = mv.verify_identity("sqrt(x^2)", "x");
    CHECK(s0.state == VerifyState::kContradicted, "sqrt(x^2) = x is false in general");
    auto s1 = mv.verify_identity("sqrt(x^2)", "x", "x >= 0");
    CHECK(s1.state == VerifyState::kVerified && has(s1.assumptions_used, "x >= 0"), "with x >= 0 it is verified, and the result says which assumption it used");
    CHECK(mv.verify_identity("sqrt(x^2)", "Abs(x)", "x real").state == VerifyState::kVerified, "for real x it is |x|");
    CHECK(mv.verify_identity("sqrt(x^2)", "-x", "x <= 0").state == VerifyState::kVerified, "for x <= 0 it is -x");
    CHECK(mv.verify_identity("sqrt(x^2)", "x", "x < 0").state == VerifyState::kContradicted, "wrong sign under the assumption");
    CHECK(mv.verify_identity("sqrt(x^2)", "Abs(x)").state != VerifyState::kVerified, "without realness it is not verified (complex x)");
    CHECK(mv.verify_equal("x^2", "x", "x > 5").state != VerifyState::kVerified, "an assumption never makes a false statement true");
    CHECK(mv.verify_identity("x^2", "x", "x = 1").state != VerifyState::kContradicted, "x = 1: no admissible counterexample, so not contradicted");
    auto bad = mv.verify_equal("x", "y", "this is nonsense !!");
    CHECK(bad.state == VerifyState::kUnknown && !bad.diagnostic.empty(), "unreadable assumptions: UNKNOWN with a diagnostic");

    std::printf("[4] rational functions: defined where?\n");
    auto r = mv.verify_identity("(x^2 - 1)/(x - 1)", "x + 1");
    CHECK(r.state == VerifyState::kVerified && has(r.conditions, "x - 1 != 0"), "equal as functions, with the condition under which both sides exist");
    CHECK(mv.verify_equal("1/x + 1/y", "(x + y)/(x*y)").state == VerifyState::kVerified, "common denominator");

    std::printf("[5] inequalities\n");
    CHECK(mv.verify_inequality("2 < 3").state == VerifyState::kVerified, "exact numbers");
    CHECK(mv.verify_inequality("sqrt(2) < 3/2").state == VerifyState::kVerified, "irrational vs rational");
    CHECK(mv.verify_inequality("3 <= 2").state == VerifyState::kContradicted, "false numeric inequality");
    CHECK(mv.verify_inequality("x^2 + 1 > 0").state == VerifyState::kVerified, "always positive");
    auto z = mv.verify_inequality("x^2 > 0");
    CHECK(z.state == VerifyState::kContradicted && !z.witness.empty(), "x^2 > 0 fails at x = 0");
    CHECK(mv.verify_inequality("x^2 > 0", "x != 0").state == VerifyState::kVerified, "with x != 0 it holds");
    CHECK(mv.verify_inequality("x > 2", "x > 3").state == VerifyState::kVerified, "a stronger assumption implies it");
    CHECK(mv.verify_inequality("x > 3", "x > 2").state == VerifyState::kContradicted, "a weaker assumption does not");
    CHECK(mv.verify_inequality("x^2 - 5*x + 6 <= 0", "x >= 2, x <= 3").state == VerifyState::kVerified, "on [2, 3]");
    CHECK(mv.verify_inequality("x^2 >= x", "x integer").state == VerifyState::kVerified, "for integers x^2 >= x");
    CHECK(mv.verify_inequality("x^2 >= x").state == VerifyState::kContradicted, "but not for all reals (x = 1/2)");
    CHECK(mv.verify_inequality("x + y > 0", "x > 0, y > 0").state != VerifyState::kContradicted, "two variables: no counterexample");
    CHECK(mv.verify_inequality("x + y > 0").state == VerifyState::kContradicted, "two variables: counterexample");

    std::printf("[6] statements, equations, bad input\n");
    CHECK(mv.verify("2 + 2 = 4").state == VerifyState::kVerified, "a single '=' is an equation");
    CHECK(mv.verify("2 + 2 == 5").state == VerifyState::kContradicted, "wrong equation");
    CHECK(mv.verify("3 != 4").state == VerifyState::kVerified, "not equal");
    CHECK(mv.verify("x > 5", "x > 6").state == VerifyState::kVerified, "inequality through verify()");
    CHECK(mv.verify("2 +* 3 = 5").state == VerifyState::kUnknown, "unparseable input is UNKNOWN, never false");
    CHECK(mv.verify("hello world").state == VerifyState::kUnknown, "not mathematics: UNKNOWN");
    CHECK(mv.verify_equal("__import__('os').system('ls')", "0").state == VerifyState::kUnknown, "code is not mathematics and is never run");

    std::printf("[7] provenance\n");
    auto p = mv.verify_identity("sqrt(x^2)", "x", "x >= 0");
    CHECK(!p.provenance.expression_hash.empty() && !p.provenance.normalized_expression.empty(), "hash and normalized expression");
    CHECK(p.provenance.backend == "StrataCAS" && !p.provenance.backend_version.empty() && !p.provenance.ir_version.empty(), "backend, version, IR version");
    CHECK(p.provenance.assumptions.find("x") != std::string::npos && !p.provenance.algorithm.empty() && p.provenance.timestamp_ms > 0, "assumptions, algorithm, timestamp");
    auto p1 = mv.verify_identity("sqrt(x^2)", "x", "x >= 0"), p2 = mv.verify_identity("sqrt(x^2)", "x", "x >= 0"), p3 = mv.verify_identity("sqrt(x^2)", "x", "x > 0");
    CHECK(p1.provenance.expression_hash == p2.provenance.expression_hash, "the same input hashes the same");
    CHECK(p1.provenance.expression_hash != p3.provenance.expression_hash, "different assumptions hash differently");

    std::printf("[8] units and dimensions\n");
    CHECK(UnitEngine::verify_equation("V*A = W").state == VerifyState::kVerified, "voltage * current = power");
    CHECK(UnitEngine::verify_equation("m/s = N").state == VerifyState::kContradicted, "velocity is not force");
    CHECK(UnitEngine::verify_equation("kg*m/s^2 = N").state == VerifyState::kVerified, "newton");
    CHECK(UnitEngine::verify_equation("N*m = J").state == VerifyState::kVerified, "joule");
    CHECK(UnitEngine::verify_equation("J/s = W").state == VerifyState::kVerified, "watt");
    CHECK(UnitEngine::verify_equation("Hz = 1/s").state == VerifyState::kVerified, "hertz");
    CHECK(UnitEngine::verify_equation("N/m^2 = Pa").state == VerifyState::kVerified, "pascal");
    CHECK(UnitEngine::verify_equation("V/A = Ohm").state == VerifyState::kVerified, "ohm");
    CHECK(UnitEngine::verify_equation("A*s = C").state == VerifyState::kVerified, "coulomb");
    CHECK(UnitEngine::verify_equation("C/V = F").state == VerifyState::kVerified, "farad");
    CHECK(UnitEngine::verify_equation("km = 1000 m").state == VerifyState::kVerified, "prefix scale");
    CHECK(UnitEngine::verify_equation("km = m").state == VerifyState::kContradicted, "same dimension, wrong magnitude");
    CHECK(UnitEngine::verify_equation("bytes = 8 bits").state == VerifyState::kVerified, "information units");
    CHECK(UnitEngine::verify_equation("m + s = m").state == VerifyState::kUnknown, "a sum of units is not a unit expression: UNKNOWN");
    auto u = UnitEngine::verify_equation("furlong/fortnight = m/s");
    CHECK(u.state == VerifyState::kUnknown && u.diagnostic.find("unknown unit") != std::string::npos, "an unknown unit is UNKNOWN, never guessed");
    CHECK(UnitEngine::verify_convertible("km/h", "m/s").state == VerifyState::kVerified, "km/h converts to m/s");
    CHECK(UnitEngine::verify_convertible("km/h", "N").state == VerifyState::kContradicted, "km/h does not convert to N");
    CHECK(UnitEngine::parse("kg*m^2/s^3").dim == UnitEngine::parse("W").dim, "kg m^2 s^-3 is watt");
    CHECK(!UnitEngine::parse(std::string(2000, 'm')).ok, "oversized unit expression refused");

    std::printf(failures ? "FAILED (%d)\n" : "ALL PASSED\n", failures);
    return failures ? 1 : 0;
}
