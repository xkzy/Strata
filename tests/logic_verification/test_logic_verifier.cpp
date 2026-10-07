// tests/logic_verification/test_logic_verifier.cpp - Logic Verification Runtime Test Suite
// The build defines NDEBUG (Release), which turns every assert() into nothing. Keep asserts active so this
// suite checks what it claims to.
#undef NDEBUG
#include "strata/logic_verification/claim_extractor.hpp"
#include "strata/logic_verification/constraint_engine.hpp"
#include "strata/logic_verification/logic_verification_runtime.hpp"
#include "strata/logic_verification/rule_engine.hpp"
#include "strata/logic_verification/schema_verifier.hpp"
#include "strata/logic_verification/unit_verifier.hpp"
#include "strata/logic_verification/verification_cache.hpp"
#include "strata/logic_verification/verification_planner.hpp"
#include "strata/logic_verification/verification_types.hpp"
#include "strata/math/math_runtime.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

void test_exact_and_floating_arithmetic() {
    std::cout << "[Test 1/12] Testing Exact & Floating Arithmetic Verification..." << std::endl;
    strata::logic::LogicVerificationRuntime verifier;

    // Case 1: Exact integer arithmetic (PASS)
    strata::logic::VerificationClaim c1;
    c1.claim_id = "c1";
    c1.type = strata::logic::ClaimType::kArithmetic;
    c1.expression = "2384 * 7291";
    c1.claimed_value = "17381744";
    auto r1 = verifier.verify(c1);
    assert(r1.status == strata::logic::VerificationStatus::kPass);
    assert(!r1.evidence.empty());

    // Case 2: Exact rational arithmetic (PASS)
    strata::logic::VerificationClaim c2;
    c2.claim_id = "c2";
    c2.type = strata::logic::ClaimType::kArithmetic;
    c2.expression = "1/3 + 1/6";
    c2.claimed_value = "1/2";
    auto r2 = verifier.verify(c2);
    assert(r2.status == strata::logic::VerificationStatus::kPass);

    // Case 3: Hallucinated arithmetic (FAIL)
    strata::logic::VerificationClaim c3;
    c3.claim_id = "c3";
    c3.type = strata::logic::ClaimType::kArithmetic;
    c3.expression = "2384 * 7291";
    c3.claimed_value = "17382094";
    auto r3 = verifier.verify(c3);
    assert(r3.status == strata::logic::VerificationStatus::kFail);
    assert(!r3.failure_reason.empty());

    // Case 4: Floating point with tolerance (PASS)
    strata::logic::VerificationClaim c4;
    c4.claim_id = "c4";
    c4.type = strata::logic::ClaimType::kArithmetic;
    c4.expression = "17.5 / 3.2";
    c4.claimed_value = "5.46875";
    auto r4 = verifier.verify(c4);
    assert(r4.status == strata::logic::VerificationStatus::kPass);

    std::cout << "  Passed. Exact integers, rationals, tolerances, and hallucination detections verified." << std::endl;
}

void test_symbolic_identities_and_equations() {
    std::cout << "[Test 2/12] Testing Symbolic Identities & Equation Verification..." << std::endl;
    strata::logic::LogicVerificationRuntime verifier;

    // Case 1: Valid algebraic expansion (PASS)
    strata::logic::VerificationClaim c1;
    c1.claim_id = "sym1";
    c1.type = strata::logic::ClaimType::kSymbolic;
    c1.expression = "(x + 1)^2";
    c1.claimed_value = "x^2 + 2*x + 1";
    auto r1 = verifier.verify(c1);
    assert(r1.status == strata::logic::VerificationStatus::kPass);

    // Case 2: Invalid algebraic expansion (FAIL)
    strata::logic::VerificationClaim c2;
    c2.claim_id = "sym2";
    c2.type = strata::logic::ClaimType::kSymbolic;
    c2.expression = "(x + 1)^2";
    c2.claimed_value = "x^2 + x + 1";
    auto r2 = verifier.verify(c2);
    assert(r2.status == strata::logic::VerificationStatus::kFail);
    assert(r2.evidence.find("difference") != std::string::npos);

    // Case 3: Equation solution (PASS)
    strata::logic::VerificationClaim c3;
    c3.claim_id = "eq1";
    c3.type = strata::logic::ClaimType::kEquation;
    c3.expression = "2*x + 5 == 15";
    c3.claimed_value = "x == 5";
    auto r3 = verifier.verify(c3);
    assert(r3.status == strata::logic::VerificationStatus::kPass);

    // Case 4: Invalid equation solution (FAIL)
    strata::logic::VerificationClaim c4;
    c4.claim_id = "eq2";
    c4.type = strata::logic::ClaimType::kEquation;
    c4.expression = "2*x + 5 == 15";
    c4.claimed_value = "x == 8";
    auto r4 = verifier.verify(c4);
    assert(r4.status == strata::logic::VerificationStatus::kFail);

    std::cout << "  Passed. Symbolic identities, invalid expansions, and algebraic equation checks verified." << std::endl;
}

void test_propositional_logic_and_rules() {
    std::cout << "[Test 3/12] Testing Propositional Rules, Modus Ponens & Fallacy Detection..." << std::endl;
    strata::logic::RuleEngine rules;

    // Modus Ponens: A, A -> B |= B (PASS)
    auto r1 = rules.verify_deduction({"A", "A -> B"}, "B");
    assert(r1.status == strata::logic::VerificationStatus::kPass);

    // Hypothetical Syllogism: A -> B, B -> C |= A -> C (PASS)
    auto r2 = rules.verify_deduction({"A -> B", "B -> C"}, "A -> C");
    assert(r2.status == strata::logic::VerificationStatus::kPass);

    // Disjunctive Syllogism: A | B, !A |= B (PASS)
    auto r3 = rules.verify_deduction({"A | B", "!A"}, "B");
    assert(r3.status == strata::logic::VerificationStatus::kPass);

    // Affirming the Consequent Fallacy: A -> B, B |= A (FAIL with countermodel)
    auto r4 = rules.verify_deduction({"A -> B", "B"}, "A");
    assert(r4.status == strata::logic::VerificationStatus::kFail);
    assert(r4.evidence.find("Countermodel") != std::string::npos);

    // Contradictory Premises: A, !A |= B (FAIL)
    auto r5 = rules.verify_deduction({"A", "!A"}, "B");
    assert(r5.status == strata::logic::VerificationStatus::kFail);

    std::cout << "  Passed. Modus ponens, syllogisms, fallacy countermodels, and contradiction guards verified." << std::endl;
}

void test_constraint_and_interval_engine() {
    std::cout << "[Test 4/12] Testing Constraint & Interval Bound Engine..." << std::endl;
    strata::logic::ConstraintEngine constrs;

    // Case 1: Satisfying variable assignment (PASS)
    auto r1 = constrs.verify_constraints({"x > 0", "x < 10"}, "x = 5");
    assert(r1.status == strata::logic::VerificationStatus::kPass);

    // Case 2: Violating variable assignment (FAIL)
    auto r2 = constrs.verify_constraints({"x > 0", "x < 10"}, "x = 15");
    assert(r2.status == strata::logic::VerificationStatus::kFail);
    assert(r2.failure_reason.find("x < 10") != std::string::npos);

    // Case 3: Inconsistent constraint bounds (FAIL)
    auto r3 = constrs.verify_constraints({"x > 10", "x < 5"}, "x = 7");
    assert(r3.status == strata::logic::VerificationStatus::kFail);

    // Case 4: Entailed sub-bound (PASS)
    auto r4 = constrs.verify_constraints({"x > 10"}, "x > 5");
    assert(r4.status == strata::logic::VerificationStatus::kPass);

    std::cout << "  Passed. Intervals, assignments, bound violations, and entailed bounds verified." << std::endl;
}

void test_unit_and_dimensional_analysis() {
    std::cout << "[Test 5/12] Testing Unit & Dimensional Analysis..." << std::endl;
    strata::logic::UnitVerifier units;

    // Case 1: Velocity = distance / time (PASS)
    auto r1 = units.verify_dimensions("10 m / 2 s", "5 m/s");
    assert(r1.status == strata::logic::VerificationStatus::kPass);
    assert(r1.expected_value == "[L T^-1]");

    // Case 2: Dimensional Mismatch (FAIL)
    auto r2 = units.verify_dimensions("10 m / 2 s", "5 kg");
    assert(r2.status == strata::logic::VerificationStatus::kFail);
    assert(r2.failure_reason.find("Dimensional mismatch") != std::string::npos);

    // Case 3: Force = mass * acceleration (PASS)
    auto r3 = units.verify_dimensions("10 kg * 2 m/s^2", "20 N");
    assert(r3.status == strata::logic::VerificationStatus::kPass);

    // Case 4: Work / Energy = Force * distance (PASS)
    auto r4 = units.verify_dimensions("10 N * 5 m", "50 J");
    assert(r4.status == strata::logic::VerificationStatus::kPass);

    std::cout << "  Passed. SI dimensions (Length, Mass, Time), derived units, and mismatch detections verified." << std::endl;
}

void test_schema_and_json_verification() {
    std::cout << "[Test 6/12] Testing Structured JSON & Schema Verifier..." << std::endl;
    strata::logic::SchemaVerifier schema;

    // Case 1: Valid JSON syntax (PASS)
    std::string valid_json = R"({"id": 101, "name": "Strata", "active": true})";
    auto r1 = schema.verify_schema(valid_json, "");
    assert(r1.status == strata::logic::VerificationStatus::kPass);

    // Case 2: Malformed JSON syntax (FAIL)
    std::string bad_json = R"({"id": 101, "name": "Strata", "active": })";
    auto r2 = schema.verify_schema(bad_json, "");
    assert(r2.status == strata::logic::VerificationStatus::kFail);

    // Case 3: Required fields present (PASS)
    std::string schema_def = R"({"required": ["id", "name"]})";
    auto r3 = schema.verify_schema(valid_json, schema_def);
    assert(r3.status == strata::logic::VerificationStatus::kPass);

    // Case 4: Missing required field (FAIL)
    std::string missing_field_schema = R"({"required": ["id", "name", "token"]})";
    auto r4 = schema.verify_schema(valid_json, missing_field_schema);
    assert(r4.status == strata::logic::VerificationStatus::kFail);
    assert(r4.failure_reason.find("token") != std::string::npos);

    std::cout << "  Passed. JSON syntax checks and schema requirements verified." << std::endl;
}

void test_tristate_unknown_preservation() {
    std::cout << "[Test 7/12] Testing Strict Tri-State Semantics (UNKNOWN NEVER converted to PASS)..." << std::endl;
    strata::logic::LogicVerificationRuntime verifier;

    // Unformalized natural language prediction
    strata::logic::VerificationClaim c;
    c.claim_id = "nl_claim_1";
    c.type = strata::logic::ClaimType::kUnknown;
    c.expression = "The quarterly revenue will increase by 25%.";
    c.claimed_value = "true";

    auto res = verifier.verify(c);
    assert(res.status == strata::logic::VerificationStatus::kUnknown);
    assert(res.status != strata::logic::VerificationStatus::kPass);
    assert(!res.failure_reason.empty());

    std::cout << "  Passed. Tri-state invariant preserved: UNKNOWN is never falsely reported as PASS." << std::endl;
}

void test_composite_partial_verification() {
    std::cout << "[Test 8/12] Testing Composite & PARTIAL Verification..." << std::endl;
    strata::logic::LogicVerificationRuntime verifier;

    strata::logic::VerificationClaim composite;
    composite.claim_id = "composite_1";
    composite.type = strata::logic::ClaimType::kComposite;

    // Sub-claim 1: Valid arithmetic (PASS)
    strata::logic::VerificationClaim sub1;
    sub1.claim_id = "sub1";
    sub1.type = strata::logic::ClaimType::kArithmetic;
    sub1.expression = "100 * 5";
    sub1.claimed_value = "500";
    composite.sub_claims.push_back(sub1);

    // Sub-claim 2: Valid unit (PASS)
    strata::logic::VerificationClaim sub2;
    sub2.claim_id = "sub2";
    sub2.type = strata::logic::ClaimType::kUnitDimension;
    sub2.unit_expression = "10 m / 2 s";
    sub2.claimed_unit = "5 m/s";
    composite.sub_claims.push_back(sub2);

    // Sub-claim 3: Unverifiable prediction (UNKNOWN)
    strata::logic::VerificationClaim sub3;
    sub3.claim_id = "sub3";
    sub3.type = strata::logic::ClaimType::kUnknown;
    sub3.expression = "Company stock will rise tomorrow.";
    sub3.claimed_value = "true";
    composite.sub_claims.push_back(sub3);

    auto res = verifier.verify(composite);
    assert(res.status == strata::logic::VerificationStatus::kPartial);
    assert(res.verified_components.size() == 2);
    assert(res.unknown_components.size() == 1);

    std::cout << "  Passed. Multi-component structured claims correctly evaluated to PARTIAL status." << std::endl;
}

void test_verification_cache_and_isolation() {
    std::cout << "[Test 9/12] Testing Verification Cache & Multi-Tenant Isolation..." << std::endl;
    strata::logic::LogicVerificationRuntime verifier;

    strata::logic::VerificationClaim claim_tenant_a;
    claim_tenant_a.claim_id = "c_a";
    claim_tenant_a.type = strata::logic::ClaimType::kArithmetic;
    claim_tenant_a.expression = "999 * 888";
    claim_tenant_a.claimed_value = "887112";
    claim_tenant_a.tenant_id = "tenant_a";

    // First call -> Cache Miss
    auto r1 = verifier.verify(claim_tenant_a);
    assert(r1.status == strata::logic::VerificationStatus::kPass);
    assert(!r1.cache_hit);

    // Second call -> Cache Hit
    auto r2 = verifier.verify(claim_tenant_a);
    assert(r2.status == strata::logic::VerificationStatus::kPass);
    assert(r2.cache_hit);

    // Tenant B call with different tenant -> Cache Miss
    strata::logic::VerificationClaim claim_tenant_b = claim_tenant_a;
    claim_tenant_b.tenant_id = "tenant_b";
    auto r3 = verifier.verify(claim_tenant_b);
    assert(r3.status == strata::logic::VerificationStatus::kPass);
    assert(!r3.cache_hit);

    auto m = verifier.get_metrics();
    assert(m.cache_hits >= 1);

    std::cout << "  Passed. Cache hits and multi-tenant key isolation verified." << std::endl;
}

void test_claim_extractor_from_text() {
    std::cout << "[Test 10/12] Testing Claim Extractor on LLM Generation Text..." << std::endl;
    strata::logic::ClaimExtractor extractor;

    std::string text =
        "Here is the calculation: 2384 * 7291 = 17381744.\n"
        "For the physical velocity: 10 m / 2 s = 5 m/s.\n"
        "Also equation: 2*x + 5 = 15 => x = 5.\n"
        "Logical reasoning: Premises: [A -> B, B -> C], Conclusion: A -> C.\n"
        "Bounds: constraints: [x > 0, x < 10], claim: x = 5.\n"
        "```json\n{\"status\": \"ok\", \"code\": 200}\n```\n";

    auto claims = extractor.extract_claims(text);
    assert(!claims.empty());

    bool has_arith = false, has_unit = false, has_eq = false, has_prop = false, has_constr = false, has_json = false;
    for (const auto& c : claims) {
        if (c.type == strata::logic::ClaimType::kArithmetic) has_arith = true;
        if (c.type == strata::logic::ClaimType::kUnitDimension) has_unit = true;
        if (c.type == strata::logic::ClaimType::kEquation) has_eq = true;
        if (c.type == strata::logic::ClaimType::kProposition) has_prop = true;
        if (c.type == strata::logic::ClaimType::kConstraint) has_constr = true;
        if (c.type == strata::logic::ClaimType::kSchemaType) has_json = true;
    }

    assert(has_arith && has_unit && has_eq && has_prop && has_constr && has_json);
    std::cout << "  Passed. Arithmetic, units, equations, propositions, constraints, and JSON claims extracted." << std::endl;
}

void test_generation_interception_and_augmentation() {
    std::cout << "[Test 11/12] Testing Generation Interception & Bounded Observations..." << std::endl;
    strata::logic::LogicVerificationRuntime verifier;

    std::string prompt = "The computed total is 2384 * 7291 = 17381744.";
    std::vector<strata::logic::VerificationResult> results;
    std::string augmented = verifier.intercept_and_verify(prompt, results);

    assert(!results.empty());
    assert(results[0].status == strata::logic::VerificationStatus::kPass);
    assert(augmented.find("[Verification: PASS") != std::string::npos);

    std::cout << "  Passed. Generation intercepted and augmented with compact external verification observation." << std::endl;
}

void test_diagnostics_and_metrics() {
    std::cout << "[Test 12/12] Testing Observability Diagnostics & Metrics..." << std::endl;
    strata::logic::LogicVerificationRuntime verifier;

    strata::logic::VerificationClaim c;
    c.claim_id = "diag_1";
    c.type = strata::logic::ClaimType::kArithmetic;
    c.expression = "50 + 50";
    c.claimed_value = "100";
    verifier.verify(c);

    std::string diag = verifier.print_diagnostics();
    assert(diag.find("Strata Deterministic Logic Verifier Diagnostics") != std::string::npos);
    assert(diag.find("Total Verifications:") != std::string::npos);

    std::cout << "  Passed. Diagnostics report rendered with metrics and backend breakdown." << std::endl;
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "    STRATA DETERMINISTIC LOGIC VERIFICATION RUNTIME TEST SUITE        \n";
    std::cout << "======================================================================\n";

    test_exact_and_floating_arithmetic();
    test_symbolic_identities_and_equations();
    test_propositional_logic_and_rules();
    test_constraint_and_interval_engine();
    test_unit_and_dimensional_analysis();
    test_schema_and_json_verification();
    test_tristate_unknown_preservation();
    test_composite_partial_verification();
    test_verification_cache_and_isolation();
    test_claim_extractor_from_text();
    test_generation_interception_and_augmentation();
    test_diagnostics_and_metrics();

    std::cout << "======================================================================\n";
    std::cout << "  ALL 12 DETERMINISTIC LOGIC VERIFICATION TEST SUITES PASSED (12/12)   \n";
    std::cout << "======================================================================\n";
    return 0;
}
