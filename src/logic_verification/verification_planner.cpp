// src/logic_verification/verification_planner.cpp - Verification Planner Implementation
#include "strata/math/cas/engine.hpp"
#include "strata/logic_verification/verification_planner.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <sstream>

namespace strata::logic {

VerificationPlanner::VerificationPlanner(std::shared_ptr<math::MathRuntime> math_runtime)
    : math_runtime_(std::move(math_runtime)),
      rule_engine_(std::make_unique<RuleEngine>()),
      constraint_engine_(std::make_unique<ConstraintEngine>()),
      unit_verifier_(std::make_unique<UnitVerifier>()),
      schema_verifier_(std::make_unique<SchemaVerifier>()) {
    if (!math_runtime_) {
        math_runtime_ = std::make_shared<math::MathRuntime>();
    }
}

VerificationPlanner::~VerificationPlanner() = default;

VerificationResult VerificationPlanner::verify_arithmetic(const VerificationClaim& claim) const {
    auto start_time = std::chrono::steady_clock::now();
    VerificationResult res;
    res.claim_id = claim.claim_id;
    res.type = ClaimType::kArithmetic;
    res.backend_used = "fast_numeric";

    // Evaluate expression using MathRuntime
    auto math_res = math_runtime_->evaluate(claim.expression, math::MathMode::kExact,
                                            claim.tenant_id, claim.session_id);

    if (math_res.status != math::MathStatus::kSuccess) {
        res.status = VerificationStatus::kUnknown;
        res.failure_reason = "Mathematical evaluation failed: " + math_res.error_message;
        res.compact_observation = "Verification: UNKNOWN | Math evaluation failed";
        return res;
    }

    res.expected_value = math_res.exact_result.empty() ? math_res.numeric_result : math_res.exact_result;
    res.actual_value = claim.claimed_value;

    // Clean whitespace
    std::string expected = res.expected_value;
    std::string claimed = claim.claimed_value;
    expected.erase(std::remove_if(expected.begin(), expected.end(), ::isspace), expected.end());
    claimed.erase(std::remove_if(claimed.begin(), claimed.end(), ::isspace), claimed.end());

    // 1. Exact string/rational match
    if (expected == claimed) {
        res.status = VerificationStatus::kPass;
        res.evidence = "Exact arithmetic match: " + claim.expression + " == " + claim.claimed_value;
        res.compact_observation = "Verification: PASS | Exact match (" + claim.claimed_value + ")";
        auto end_time = std::chrono::steady_clock::now();
        res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        return res;
    }

    // 2. Numerical tolerance comparison (for floating point / decimals vs rationals)
    try {
        double exp_val = 0.0;
        if (expected.find('/') != std::string::npos) {
            size_t slash = expected.find('/');
            double n = std::stod(expected.substr(0, slash));
            double d = std::stod(expected.substr(slash + 1));
            exp_val = (d != 0.0) ? (n / d) : 0.0;
        } else {
            exp_val = std::stod(expected);
        }

        double claim_val = 0.0;
        if (claimed.find('/') != std::string::npos) {
            size_t slash = claimed.find('/');
            double n = std::stod(claimed.substr(0, slash));
            double d = std::stod(claimed.substr(slash + 1));
            claim_val = (d != 0.0) ? (n / d) : 0.0;
        } else {
            claim_val = std::stod(claimed);
        }

        double diff = std::abs(exp_val - claim_val);
        double max_abs = std::max(std::abs(exp_val), std::abs(claim_val));
        double rel_diff = max_abs > 1e-12 ? (diff / max_abs) : diff;

        if (diff <= claim.tolerance.abs_tol || rel_diff <= claim.tolerance.rel_tol) {
            res.status = VerificationStatus::kPass;
            std::ostringstream ev;
            ev << "Numerical match within tolerance (abs_diff=" << diff
               << ", rel_diff=" << rel_diff << "): " << exp_val << " ~= " << claim_val;
            res.evidence = ev.str();
            res.compact_observation = "Verification: PASS | " + claim.expression + " ~= " + claim.claimed_value;
        } else {
            res.status = VerificationStatus::kFail;
            std::ostringstream ev;
            ev << "Arithmetic contradiction: expected " << expected << " (" << exp_val
               << "), but claimed " << claimed << " (" << claim_val << ") (diff=" << diff << ")";
            res.evidence = ev.str();
            res.failure_reason = "Claimed value contradicts deterministic arithmetic result";
            res.compact_observation = "Verification: FAIL | Expected " + expected + ", claimed " + claimed;
        }
    } catch (...) {
        res.status = VerificationStatus::kFail;
        res.evidence = "String mismatch between expected '" + expected + "' and claimed '" + claimed + "'";
        res.failure_reason = "Claimed value does not match arithmetic result";
        res.compact_observation = "Verification: FAIL | Expected " + expected + ", claimed " + claimed;
    }

    auto end_time = std::chrono::steady_clock::now();
    res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    return res;
}

VerificationResult VerificationPlanner::verify_symbolic(const VerificationClaim& claim) const {
    auto start_time = std::chrono::steady_clock::now();
    VerificationResult res;
    res.claim_id = claim.claim_id;
    res.type = ClaimType::kSymbolic;
    res.backend_used = "mathics";

    // For symbolic identity "A == B", we simplify "(A) - (B)"
    std::string lhs = claim.expression;
    std::string rhs = claim.claimed_value;

    if (rhs.empty() && lhs.find("==") != std::string::npos) {
        size_t eq_pos = lhs.find("==");
        rhs = lhs.substr(eq_pos + 2);
        lhs = lhs.substr(0, eq_pos);
    }

    std::string diff_expr = "(" + lhs + ") - (" + rhs + ")";
    auto math_res = math_runtime_->simplify(diff_expr, claim.tenant_id, claim.session_id);

    if (math_res.status != math::MathStatus::kSuccess) {
        res.status = VerificationStatus::kUnknown;
        res.failure_reason = "Symbolic simplification failed: " + math_res.error_message;
        res.compact_observation = "Verification: UNKNOWN | Symbolic engine could not establish identity";
        return res;
    }

    std::string diff = math_res.exact_result;
    diff.erase(std::remove_if(diff.begin(), diff.end(), ::isspace), diff.end());

    if (diff == "0" || diff == "0.0") {
        res.status = VerificationStatus::kPass;
        res.evidence = "Symbolic equality proven: (" + lhs + ") - (" + rhs + ") simplifies to 0.";
        res.expected_value = lhs;
        res.actual_value = rhs;
        res.compact_observation = "Verification: PASS | Symbolic identity proven (" + lhs + " == " + rhs + ")";
    } else {
        // "Did not simplify to 0" is not a contradiction: the simplifier may just be too weak. Only a concrete
        // numeric counterexample is evidence of FAIL; everything else is UNKNOWN.
        std::string witness;
        bool found = false;
        try {
            math::cas::Engine engine;
            found = engine.find_nonzero_witness(math::cas::parse(diff_expr), witness);
        } catch (const math::cas::CasError&) {
            found = false;
        }
        if (found) {
            res.status = VerificationStatus::kFail;
            res.evidence = "Symbolic contradiction: difference (" + lhs + ") - (" + rhs + ") = " + math_res.exact_result +
                           "; counterexample: " + witness;
            res.failure_reason = "Expressions are not symbolically equivalent";
            res.expected_value = lhs;
            res.actual_value = rhs;
            res.compact_observation = "Verification: FAIL | Symbolic contradiction (" + witness + ")";
        } else {
            res.status = VerificationStatus::kUnknown;
            res.evidence = "Difference (" + lhs + ") - (" + rhs + ") did not simplify to 0 and no counterexample was found: " + math_res.exact_result;
            res.failure_reason = "Could not establish or refute the identity";
            res.compact_observation = "Verification: UNKNOWN | Symbolic engine could not establish identity";
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    return res;
}

namespace {

// Reads claimed assignments such as "x == 5", "x = 5", "x -> 5", "{x -> 1, y -> 2}", "x = 1 and y = 2".
bool parse_assignments(std::string text, std::vector<std::pair<std::string, math::cas::Expr>>& out) {
    auto trim = [](std::string s) {
        size_t a = 0, b = s.size();
        while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
        while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
        return s.substr(a, b - a);
    };
    text = trim(text);
    while (text.size() >= 2 && ((text.front() == '{' && text.back() == '}') || (text.front() == '(' && text.back() == ')'))) text = trim(text.substr(1, text.size() - 2));
    // split on top-level ',', ';', "&&", " and "
    std::vector<std::string> pieces;
    std::string cur;
    int depth = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        if (c == '(' || c == '[' || c == '{') ++depth;
        if (c == ')' || c == ']' || c == '}') --depth;
        bool split = false;
        size_t skip = 0;
        if (depth == 0) {
            if (c == ',' || c == ';') split = true;
            else if (text.compare(i, 2, "&&") == 0) { split = true; skip = 1; }
            else if (text.compare(i, 5, " and ") == 0) { split = true; skip = 4; }
        }
        if (split) { pieces.push_back(cur); cur.clear(); i += skip; }
        else cur.push_back(c);
    }
    pieces.push_back(cur);
    for (auto& piece : pieces) {
        piece = trim(piece);
        if (piece.empty()) return false;
        size_t pos = std::string::npos, len = 0;
        if ((pos = piece.find("->")) != std::string::npos) len = 2;
        else if ((pos = piece.find("==")) != std::string::npos) len = 2;
        else if ((pos = piece.find('=')) != std::string::npos) len = 1;
        else return false;
        std::string var = trim(piece.substr(0, pos));
        if (var.empty() || !std::isalpha(static_cast<unsigned char>(var[0]))) return false;
        for (char ch : var) if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '_') return false;
        out.emplace_back(var, math::cas::parse(piece.substr(pos + len)));
    }
    return !out.empty();
}

} // namespace

VerificationResult VerificationPlanner::verify_equation(const VerificationClaim& claim) const {
    auto start_time = std::chrono::steady_clock::now();
    VerificationResult res;
    res.claim_id = claim.claim_id;
    res.type = ClaimType::kEquation;
    res.backend_used = "strata_cas";
    namespace cas = math::cas;

    auto finish = [&]() {
        auto end_time = std::chrono::steady_clock::now();
        res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        return res;
    };
    auto unknown = [&](const std::string& why) {
        res.status = VerificationStatus::kUnknown;
        res.failure_reason = why;
        res.compact_observation = "Verification: UNKNOWN | " + why;
        return finish();
    };

    try {
        cas::Engine engine;
        cas::Expr eqs = cas::parse(claim.expression);
        std::vector<std::pair<std::string, cas::Expr>> claimed;
        if (!parse_assignments(claim.claimed_value, claimed)) return unknown("Could not read the claimed solution '" + claim.claimed_value + "'");
        for (auto& c : claimed) c.second = engine.eval(c.second);

        std::vector<cas::Expr> equations = eqs->has_head("List") ? eqs->args : std::vector<cas::Expr>{eqs};
        std::string claim_text;
        for (size_t i = 0; i < claimed.size(); ++i) claim_text += (i ? ", " : "") + claimed[i].first + " = " + cas::to_string(claimed[i].second);

        // 1. Substitute the claimed values into every equation: exact zero is a proof, a non-zero value a counterexample.
        bool all_satisfied = true;
        for (const auto& q : equations) {
            cas::Expr lhs = q->has_head("Equal", 2) ? q->args[0] : q;
            cas::Expr rhs = q->has_head("Equal", 2) ? q->args[1] : cas::zero();
            cas::Expr residual = engine.simplify(engine.sub(cas::Engine::substitute(lhs, claimed), cas::Engine::substitute(rhs, claimed)));
            if (residual->is_number() && residual->q.is_zero()) continue;
            double v = 0.0;
            if (engine.numeric_value(residual, v)) {
                if (std::fabs(v) > 1e-9) {
                    res.status = VerificationStatus::kFail;
                    res.evidence = "Substituting " + claim_text + " into (" + cas::to_string(lhs) + ") - (" + cas::to_string(rhs) + ") gives " + cas::to_string(residual) + " != 0";
                    res.failure_reason = "Claimed solution does not satisfy the equation";
                    res.expected_value = "residual 0";
                    res.actual_value = cas::to_string(residual);
                    res.compact_observation = "Verification: FAIL | " + claim_text + " leaves residual " + cas::to_string(residual);
                    return finish();
                }
                continue;  // numerically zero (e.g. irrational), treat as satisfied
            }
            all_satisfied = false;
        }
        if (!all_satisfied) return unknown("Could not decide whether the claimed values satisfy the equation");

        // 2. The claimed values satisfy the equation. Is it the complete solution set?
        std::vector<std::string> vars;
        for (const auto& c : claimed) vars.push_back(c.first);
        std::string completeness;
        bool complete = false;
        try {
            cas::Expr sol = engine.solve(eqs, vars);
            size_t sets = sol->args.size();
            bool matches_one = false;
            for (const auto& set : sol->args) {
                bool all = set->args.size() == claimed.size();
                for (const auto& r : set->args) {
                    bool found = false;
                    for (const auto& c : claimed) {
                        if (r->args[0]->name != c.first) continue;
                        cas::Expr d = engine.simplify(engine.sub(r->args[1], c.second));
                        found = d->is_number() && d->q.is_zero();
                    }
                    if (!found) { all = false; break; }
                }
                if (all) matches_one = true;
            }
            complete = matches_one && sets == 1;
            completeness = "full solution set: " + cas::to_string(sol);
        } catch (const cas::CasUnsupported& ex) {
            completeness = std::string("completeness not established (") + ex.what() + ")";
        }
        if (complete) {
            res.status = VerificationStatus::kPass;
            res.evidence = "Claimed solution " + claim_text + " satisfies the equation and is the complete solution set.";
            res.expected_value = claim_text;
            res.actual_value = claim_text;
            res.compact_observation = "Verification: PASS | Equation solution verified (" + claim_text + ")";
        } else {
            res.status = VerificationStatus::kPartial;
            res.evidence = "Claimed solution " + claim_text + " satisfies the equation, but " + completeness;
            res.failure_reason = "Solution is valid but not shown to be the only one";
            res.expected_value = claim_text;
            res.actual_value = claim_text;
            res.compact_observation = "Verification: PARTIAL | " + claim_text + " is a solution; " + completeness;
        }
        return finish();
    } catch (const cas::CasLimitError& ex) {
        return unknown(std::string("Resource limit: ") + ex.what());
    } catch (const cas::CasError& ex) {
        return unknown(ex.what());
    }
}

VerificationResult VerificationPlanner::verify_claim(const VerificationClaim& claim,
                                                    VerificationCache* cache) const {
    // Step 1: Check Cache (Hierarchy Level 1)
    if (cache) {
        VerificationResult cached_res;
        if (cache->get(claim, cached_res)) {
            return cached_res;
        }
    }

    VerificationResult res;

    // Step 2: Route by Claim Type
    switch (claim.type) {
        case ClaimType::kArithmetic:
            res = verify_arithmetic(claim);
            break;
        case ClaimType::kSymbolic:
            res = verify_symbolic(claim);
            break;
        case ClaimType::kEquation:
            res = verify_equation(claim);
            break;
        case ClaimType::kProposition:
            res = rule_engine_->verify_deduction(claim.premises, claim.claimed_value, claim.claim_id);
            break;
        case ClaimType::kConstraint:
            res = constraint_engine_->verify_constraints(claim.constraints, claim.claimed_value, claim.claim_id);
            break;
        case ClaimType::kUnitDimension:
            res = unit_verifier_->verify_dimensions(claim.unit_expression, claim.claimed_unit, claim.claim_id);
            break;
        case ClaimType::kSchemaType:
            res = schema_verifier_->verify_schema(claim.schema_json, claim.expression, claim.claim_id);
            break;
        case ClaimType::kComposite: {
            // Handle composite multi-claims
            res.claim_id = claim.claim_id;
            res.type = ClaimType::kComposite;
            res.backend_used = "composite_planner";

            size_t pass_count = 0, fail_count = 0, unk_count = 0;
            for (const auto& sub : claim.sub_claims) {
                auto sub_res = verify_claim(sub, cache);
                if (sub_res.status == VerificationStatus::kPass) {
                    pass_count++;
                    res.verified_components.push_back(sub.claim_id + ": " + sub.expression);
                } else if (sub_res.status == VerificationStatus::kFail) {
                    fail_count++;
                    res.failed_components.push_back(sub.claim_id + ": " + sub_res.failure_reason);
                } else {
                    unk_count++;
                    res.unknown_components.push_back(sub.claim_id + ": " + sub.expression);
                }
            }

            if (fail_count > 0 && pass_count == 0) {
                res.status = VerificationStatus::kFail;
                res.failure_reason = "All sub-claims failed verification";
            } else if (fail_count > 0 || unk_count > 0) {
                if (pass_count > 0) {
                    res.status = VerificationStatus::kPartial;
                    res.evidence = "Partial verification: " + std::to_string(pass_count) + " passed, " +
                                   std::to_string(fail_count) + " failed, " + std::to_string(unk_count) + " unknown.";
                } else {
                    res.status = VerificationStatus::kUnknown;
                    res.failure_reason = "No sub-claims could be deterministically verified";
                }
            } else {
                res.status = VerificationStatus::kPass;
                res.evidence = "All " + std::to_string(pass_count) + " composite sub-claims passed verification.";
            }
            res.compact_observation = "Verification: " + std::string(verification_status_to_string(res.status)) +
                                      " (" + std::to_string(pass_count) + "/" + std::to_string(claim.sub_claims.size()) + " verified)";
            break;
        }
        case ClaimType::kCodeSyntax:
        case ClaimType::kSQL:
        case ClaimType::kConsistency:
        case ClaimType::kUnknown:
        default:
            // Conservative fallback -> UNKNOWN (NEVER PASS)
            res.claim_id = claim.claim_id;
            res.type = claim.type;
            res.status = VerificationStatus::kUnknown;
            res.backend_used = "conservative_fallback";
            res.failure_reason = "Unsupported claim type or natural-language reasoning cannot be verified without formal model";
            res.evidence = "Conservative tri-state verifier refuses to guess truth value of unformalized natural language claim.";
            res.compact_observation = "Verification: UNKNOWN | Unformalized claim";
            break;
    }

    // Step 3: Put into Cache if deterministic PASS or FAIL
    if (cache && (res.status == VerificationStatus::kPass || res.status == VerificationStatus::kFail)) {
        cache->put(claim, res);
    }

    return res;
}

} // namespace strata::logic
