// SPDX-License-Identifier: GPL-3.0-or-later
// tests/math/test_theorems.cpp - Unit tests for mathematical theorem database and verifiers
#include "strata/math/theorems.hpp"
#include <cassert>
#include <iostream>
#include <vector>

using namespace strata::math;

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAIL: " << (msg) << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            g_failed++; \
        } else { \
            g_passed++; \
        } \
    } while(0)

void test_registry() {
    auto& engine = TheoremEngine::instance();
    auto all = engine.list_all();
    CHECK(all.size() >= 20, "Should have at least 20 theorems registered");

    auto nt = engine.list_by_category(TheoremCategory::kNumberTheory);
    CHECK(!nt.empty(), "Should have number theory theorems");

    auto alg = engine.list_by_category(TheoremCategory::kAlgebra);
    CHECK(!alg.empty(), "Should have algebra theorems");

    auto la = engine.list_by_category(TheoremCategory::kLinearAlgebra);
    CHECK(!la.empty(), "Should have linear algebra theorems");

    auto calc = engine.list_by_category(TheoremCategory::kCalculus);
    CHECK(!calc.empty(), "Should have calculus theorems");

    auto ineq = engine.list_by_category(TheoremCategory::kInequality);
    CHECK(!ineq.empty(), "Should have inequality theorems");

    auto trig = engine.list_by_category(TheoremCategory::kTrigonometry);
    CHECK(!trig.empty(), "Should have trigonometry theorems");

    auto prob = engine.list_by_category(TheoremCategory::kProbability);
    CHECK(!prob.empty(), "Should have probability theorems");
}

void test_number_theory_theorems() {
    auto& engine = TheoremEngine::instance();

    // 1. Fermat's Little Theorem: a=7, p=101 -> Verified
    auto res1 = engine.verify_theorem("fermat_little_theorem", {"7", "101"});
    CHECK(res1.state == VerifyState::kVerified, "FLT for a=7, p=101 should be VERIFIED");

    // Fermat with composite p=100 -> Contradicted
    auto res1_comp = engine.verify_theorem("fermat_little_theorem", {"7", "100"});
    CHECK(res1_comp.state == VerifyState::kContradicted, "FLT for composite p should be CONTRADICTED");

    // Fermat with non-coprime a=7, p=14 -> Contradicted
    auto res1_coprime = engine.verify_theorem("fermat_little_theorem", {"7", "14"});
    CHECK(res1_coprime.state == VerifyState::kContradicted, "FLT for non-coprime should be CONTRADICTED");

    // 2. Euler's Totient Theorem: a=3, n=10 (phi(10)=4, 3^4=81=1 mod 10)
    auto res2 = engine.verify_theorem("euler_totient_theorem", {"3", "10"});
    CHECK(res2.state == VerifyState::kVerified, "Euler Totient for a=3, n=10 should be VERIFIED");

    // 3. Wilson's Theorem: p=13 -> prime -> Verified
    auto res3 = engine.verify_theorem("wilson_theorem", {"13"});
    CHECK(res3.state == VerifyState::kVerified, "Wilson for p=13 should be VERIFIED");

    // Wilson for composite p=12 -> Verified composite condition
    auto res3_comp = engine.verify_theorem("wilson_theorem", {"12"});
    CHECK(res3_comp.state == VerifyState::kVerified, "Wilson for p=12 should report verified composite");

    // 4. Chinese Remainder Theorem: x = 2 mod 3, x = 3 mod 5, x = 2 mod 7 -> x = 23 mod 105
    auto res4 = engine.verify_theorem("chinese_remainder_theorem", {"2", "3", "3", "5", "2", "7"});
    CHECK(res4.state == VerifyState::kVerified, "CRT should solve system to x=23 mod 105");

    // 5. Bézout's Identity: a=240, b=46 -> gcd=2
    auto res5 = engine.verify_theorem("bezout_identity", {"240", "46"});
    CHECK(res5.state == VerifyState::kVerified, "Bézout for (240, 46) should be VERIFIED");

    // 6. Fundamental Theorem of Arithmetic: n=360 = 2^3 * 3^2 * 5^1
    auto res6 = engine.verify_theorem("fundamental_theorem_of_arithmetic", {"360"});
    CHECK(res6.state == VerifyState::kVerified, "FTA factorization for 360 should be VERIFIED");

    // 7. Lagrange's Four-Square: n=2023 -> sum of 4 squares
    auto res7 = engine.verify_theorem("lagrange_four_square_theorem", {"2023"});
    CHECK(res7.state == VerifyState::kVerified, "Lagrange 4 squares for 2023 should be VERIFIED");
}

void test_algebra_linear_theorems() {
    auto& engine = TheoremEngine::instance();

    // Binomial Theorem: (2 + 3)^4 = 5^4 = 625
    auto res1 = engine.verify_theorem("binomial_theorem", {"2", "3", "4"});
    CHECK(res1.state == VerifyState::kVerified, "Binomial expansion for (2+3)^4 should be VERIFIED");

    // Fundamental Theorem of Algebra
    auto res2 = engine.verify_theorem("fundamental_theorem_of_algebra", {});
    CHECK(res2.state == VerifyState::kVerified, "FTA should be VERIFIED");

    // Cayley-Hamilton
    auto res3 = engine.verify_theorem("cayley_hamilton_theorem", {});
    CHECK(res3.state == VerifyState::kVerified, "Cayley-Hamilton should be VERIFIED");

    // Rank-Nullity: rank=2, nullity=1, cols=3
    auto res4 = engine.verify_theorem("rank_nullity_theorem", {"2", "1", "3"});
    CHECK(res4.state == VerifyState::kVerified, "Rank-Nullity (2,1,3) should be VERIFIED");

    // Lagrange Group Theorem: |G|=24, |H|=6 -> Verified
    auto res_grp = engine.verify_theorem("lagrange_group_theorem", {"24", "6"});
    CHECK(res_grp.state == VerifyState::kVerified, "Lagrange Group Theorem (24, 6) should be VERIFIED");

    auto res_grp_inv = engine.verify_theorem("lagrange_group_theorem", {"24", "7"});
    CHECK(res_grp_inv.state == VerifyState::kContradicted, "Lagrange Group Theorem (24, 7) should be CONTRADICTED");
}

void test_calculus_inequality_theorems() {
    auto& engine = TheoremEngine::instance();

    // Fundamental Theorem of Calculus
    auto res1 = engine.verify_theorem("fundamental_theorem_of_calculus", {});
    CHECK(res1.state == VerifyState::kVerified, "FTC should be VERIFIED");

    // Mean Value Theorem
    auto res2 = engine.verify_theorem("mean_value_theorem", {});
    CHECK(res2.state == VerifyState::kVerified, "MVT should be VERIFIED");

    // Euler's Formula
    auto res3 = engine.verify_theorem("euler_formula", {});
    CHECK(res3.state == VerifyState::kVerified, "Euler formula should be VERIFIED");

    // AM-GM: [2, 4, 8] -> AM=14/3 ~ 4.666, GM=4 -> AM >= GM -> Verified
    auto res4 = engine.verify_theorem("am_gm_inequality", {"2", "4", "8"});
    CHECK(res4.state == VerifyState::kVerified, "AM-GM for [2, 4, 8] should be VERIFIED");

    // AM-GM with negative -> Contradicted (precondition violated)
    auto res4_neg = engine.verify_theorem("am_gm_inequality", {"-2", "4", "8"});
    CHECK(res4_neg.state == VerifyState::kContradicted, "AM-GM with negative element should be CONTRADICTED");

    // Cauchy-Schwarz
    auto res5 = engine.verify_theorem("cauchy_schwarz_inequality", {});
    CHECK(res5.state == VerifyState::kVerified, "Cauchy-Schwarz should be VERIFIED");
}

void test_probability_geometry_theorems() {
    auto& engine = TheoremEngine::instance();

    // Bayes Theorem: P(A)=0.01, P(B|A)=0.9, P(B|~A)=0.05
    // P(B) = 0.9 * 0.01 + 0.05 * 0.99 = 0.009 + 0.0495 = 0.0585
    // P(A|B) = 0.009 / 0.0585 ~ 0.153846
    auto res1 = engine.verify_theorem("bayes_theorem", {"0.01", "0.9", "0.05"});
    CHECK(res1.state == VerifyState::kVerified, "Bayes theorem should be VERIFIED");
    CHECK(res1.witness.find("0.1538") != std::string::npos, "Bayes calculation should yield ~0.1538");

    // Euler Polyhedral: Cube -> V=8, E=12, F=6 -> 8 - 12 + 6 = 2
    auto res2 = engine.verify_theorem("euler_polyhedral_formula", {"8", "12", "6"});
    CHECK(res2.state == VerifyState::kVerified, "Euler polyhedral for cube (8, 12, 6) should be VERIFIED");

    // Euler Polyhedral invalid: V=8, E=10, F=6 -> 8 - 10 + 6 = 4 != 2 -> Contradicted
    auto res2_inv = engine.verify_theorem("euler_polyhedral_formula", {"8", "10", "6"});
    CHECK(res2_inv.state == VerifyState::kContradicted, "Euler polyhedral invalid (8, 10, 6) should be CONTRADICTED");

    // Shannon-Hartley: B=1000, SNR=7 -> C = 1000 * log2(8) = 3000
    auto res_sh = engine.verify_theorem("shannon_hartley_theorem", {"1000", "7"});
    CHECK(res_sh.state == VerifyState::kVerified, "Shannon-Hartley should be VERIFIED");
    CHECK(res_sh.witness.find("3000") != std::string::npos, "Shannon capacity should be 3000 bits/s");

    // Quadratic Reciprocity: p=3, q=5
    auto res_qr = engine.verify_theorem("quadratic_reciprocity", {"3", "5"});
    CHECK(res_qr.state == VerifyState::kVerified, "Quadratic Reciprocity for (3, 5) should be VERIFIED");

    // Sherman-Morrison
    auto res_sm = engine.verify_theorem("sherman_morrison_formula", {});
    CHECK(res_sm.state == VerifyState::kVerified, "Sherman-Morrison should be VERIFIED");

    // Basel Problem
    auto res_bp = engine.verify_theorem("basel_problem", {});
    CHECK(res_bp.state == VerifyState::kVerified, "Basel Problem identity should be VERIFIED");

    // RSA: p=61, q=53 -> N=3233, e=17, d=2753, m=65
    auto res_rsa = engine.verify_theorem("rsa_correctness_theorem", {"65", "17", "2753", "3233"});
    CHECK(res_rsa.state == VerifyState::kVerified, "RSA correctness should be VERIFIED");

    // Natural language claim matching
    auto res3 = engine.verify_claim("By Fermat's Little Theorem, the congruence holds.");
    CHECK(res3.state == VerifyState::kVerified, "Claim matching Fermat's Little Theorem should succeed");

    auto res4 = engine.verify_claim("Applying Bayes' Rule to update the probability distribution.");
    CHECK(res4.state == VerifyState::kVerified, "Claim matching Bayes' Rule should succeed");
}

int main() {
    std::cout << "=== Running Mathematical Theorems Test Suite ===\n";
    test_registry();
    test_number_theory_theorems();
    test_algebra_linear_theorems();
    test_calculus_inequality_theorems();
    test_probability_geometry_theorems();

    std::cout << "\nSummary: " << g_passed << " passed, " << g_failed << " failed.\n";
    return g_failed == 0 ? 0 : 1;
}
