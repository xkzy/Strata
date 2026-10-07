// src/logic_verification/constraint_engine.cpp - Constraint Engine Implementation
#include "strata/logic_verification/constraint_engine.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <regex>
#include <sstream>

namespace strata::logic {

bool IntervalBound::contains(double v, double tol) const {
    if (has_min) {
        if (min_inclusive) {
            if (v < min_val - tol) return false;
        } else {
            if (v <= min_val + tol) return false;
        }
    }
    if (has_max) {
        if (max_inclusive) {
            if (v > max_val + tol) return false;
        } else {
            if (v >= max_val - tol) return false;
        }
    }
    for (double excl : excluded_values) {
        if (std::abs(v - excl) <= tol) return false;
    }
    return true;
}

bool IntervalBound::is_empty() const {
    if (has_min && has_max) {
        if (min_val > max_val) return true;
        if (min_val == max_val && (!min_inclusive || !max_inclusive)) return true;
    }
    return false;
}

bool IntervalBound::intersect(const IntervalBound& other) {
    if (other.has_min) {
        if (!has_min || other.min_val > min_val) {
            min_val = other.min_val;
            min_inclusive = other.min_inclusive;
            has_min = true;
        } else if (other.min_val == min_val) {
            min_inclusive = min_inclusive && other.min_inclusive;
        }
    }
    if (other.has_max) {
        if (!has_max || other.max_val < max_val) {
            max_val = other.max_val;
            max_inclusive = other.max_inclusive;
            has_max = true;
        } else if (other.max_val == max_val) {
            max_inclusive = max_inclusive && other.max_inclusive;
        }
    }
    for (double excl : other.excluded_values) {
        excluded_values.push_back(excl);
    }
    return !is_empty();
}

// ---------------------------------------------------------------------------
// ConstraintEngine Implementation
// ---------------------------------------------------------------------------

ConstraintEngine::ConstraintEngine() = default;
ConstraintEngine::~ConstraintEngine() = default;

bool ConstraintEngine::parse_constraint(const std::string& s, std::string& out_var,
                                        IntervalBound& out_bound, std::string& out_raw_op,
                                        std::string& err) const {
    // Regex for: var (< | <= | > | >= | == | = | !=) number
    static const std::regex c_regex(
        R"(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(<=|>=|<|>|==|=|!=)\s*([\-\+]?\d+(?:\.\d+)?(?:[eE][\-\+]?\d+)?)\s*)",
        std::regex::optimize);

    std::smatch match;
    if (!std::regex_match(s, match, c_regex)) {
        err = "Unsupported or malformed constraint syntax: " + s;
        return false;
    }

    out_var = match[1].str();
    out_raw_op = match[2].str();
    double val = std::stod(match[3].str());

    out_bound = IntervalBound();
    if (out_raw_op == ">") {
        out_bound.min_val = val;
        out_bound.min_inclusive = false;
        out_bound.has_min = true;
    } else if (out_raw_op == ">=") {
        out_bound.min_val = val;
        out_bound.min_inclusive = true;
        out_bound.has_min = true;
    } else if (out_raw_op == "<") {
        out_bound.max_val = val;
        out_bound.max_inclusive = false;
        out_bound.has_max = true;
    } else if (out_raw_op == "<=") {
        out_bound.max_val = val;
        out_bound.max_inclusive = true;
        out_bound.has_max = true;
    } else if (out_raw_op == "==" || out_raw_op == "=") {
        out_bound.min_val = val;
        out_bound.max_val = val;
        out_bound.min_inclusive = true;
        out_bound.max_inclusive = true;
        out_bound.has_min = true;
        out_bound.has_max = true;
    } else if (out_raw_op == "!=") {
        out_bound.excluded_values.push_back(val);
    }

    return true;
}

bool ConstraintEngine::parse_assignment(const std::string& s, std::string& out_var,
                                        double& out_val, std::string& err) const {
    static const std::regex a_regex(
        R"(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:=|==|:=)\s*([\-\+]?\d+(?:\.\d+)?(?:[eE][\-\+]?\d+)?)\s*)",
        std::regex::optimize);

    std::smatch match;
    if (!std::regex_match(s, match, a_regex)) {
        err = "Not a variable assignment: " + s;
        return false;
    }

    out_var = match[1].str();
    out_val = std::stod(match[2].str());
    return true;
}

bool ConstraintEngine::check_consistency(const std::vector<std::string>& constraints,
                                         std::string& out_conflict_explanation) const {
    std::unordered_map<std::string, IntervalBound> system;
    std::string err;

    for (const auto& c : constraints) {
        std::string var, op;
        IntervalBound bound;
        if (!parse_constraint(c, var, bound, op, err)) {
            continue;
        }

        if (system.find(var) == system.end()) {
            system[var] = bound;
        } else {
            if (!system[var].intersect(bound)) {
                out_conflict_explanation = "Constraint '" + c + "' causes empty interval for variable '" + var + "'";
                return false;
            }
        }
    }
    return true;
}

VerificationResult ConstraintEngine::verify_constraints(const std::vector<std::string>& constraints,
                                                        const std::string& claim,
                                                        const std::string& claim_id) const {
    auto start_time = std::chrono::steady_clock::now();
    VerificationResult res;
    res.claim_id = claim_id;
    res.type = ClaimType::kConstraint;
    res.backend_used = "constraint_engine";

    // 1. Check Constraint System Consistency
    std::unordered_map<std::string, IntervalBound> system;
    std::unordered_map<std::string, std::vector<std::string>> var_constraints;
    std::string err;

    for (const auto& c : constraints) {
        std::string var, op;
        IntervalBound bound;
        if (!parse_constraint(c, var, bound, op, err)) {
            res.status = VerificationStatus::kUnknown;
            res.failure_reason = "Unparseable constraint syntax: " + err;
            res.compact_observation = "Verification: UNKNOWN | Reason: unparseable constraint";
            return res;
        }

        var_constraints[var].push_back(c);
        if (system.find(var) == system.end()) {
            system[var] = bound;
        } else {
            if (!system[var].intersect(bound)) {
                res.status = VerificationStatus::kFail;
                res.failure_reason = "Inconsistent constraint system: '" + c + "' conflicts with existing bounds on '" + var + "'";
                res.evidence = "Constraint set is mathematically unsatisfiable";
                res.compact_observation = "Verification: FAIL | Inconsistent constraint bounds";
                auto end_time = std::chrono::steady_clock::now();
                res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                return res;
            }
        }
    }

    // 2. Parse Claim as Variable Assignment (e.g. "x = 15")
    std::string claim_var;
    double claim_val = 0.0;
    if (parse_assignment(claim, claim_var, claim_val, err)) {
        auto it = system.find(claim_var);
        if (it != system.end()) {
            if (!it->second.contains(claim_val)) {
                // Find specifically which constraint was violated
                std::string violated_constraint;
                for (const auto& c_str : var_constraints[claim_var]) {
                    std::string v, op;
                    IntervalBound b;
                    if (parse_constraint(c_str, v, b, op, err)) {
                        if (!b.contains(claim_val)) {
                            violated_constraint = c_str;
                            break;
                        }
                    }
                }

                res.status = VerificationStatus::kFail;
                res.failure_reason = "Claim '" + claim + "' violates constraint '" + violated_constraint + "'";
                std::ostringstream ev;
                ev << "Value " << claim_val << " for " << claim_var << " contradicts constraint " << violated_constraint;
                res.evidence = ev.str();
                res.compact_observation = "Verification: FAIL | Violates " + violated_constraint;
                auto end_time = std::chrono::steady_clock::now();
                res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                return res;
            } else {
                res.status = VerificationStatus::kPass;
                std::ostringstream ev;
                ev << "Value " << claim_val << " for " << claim_var << " satisfies all bounds in constraint system.";
                res.evidence = ev.str();
                res.expected_value = "in " + claim_var + " valid interval";
                res.actual_value = std::to_string(claim_val);
                res.compact_observation = "Verification: PASS | Satisfies constraints";
                auto end_time = std::chrono::steady_clock::now();
                res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                return res;
            }
        } else {
            res.status = VerificationStatus::kUnknown;
            res.failure_reason = "Variable '" + claim_var + "' is unconstrained in premise system";
            res.compact_observation = "Verification: UNKNOWN | Unconstrained variable";
            return res;
        }
    }

    // 3. Parse Claim as Sub-Constraint (e.g. "x > 5")
    std::string sub_var, sub_op;
    IntervalBound sub_bound;
    if (parse_constraint(claim, sub_var, sub_bound, sub_op, err)) {
        auto it = system.find(sub_var);
        if (it != system.end()) {
            // Check if sub_bound is a logical consequence of system[sub_var]
            bool entailed = true;
            if (sub_bound.has_min && (!it->second.has_min || it->second.min_val < sub_bound.min_val)) {
                entailed = false;
            }
            if (sub_bound.has_max && (!it->second.has_max || it->second.max_val > sub_bound.max_val)) {
                entailed = false;
            }

            if (entailed) {
                res.status = VerificationStatus::kPass;
                res.evidence = "Claim '" + claim + "' is strictly entailed by constraint interval.";
                res.compact_observation = "Verification: PASS | Entailed by constraints";
            } else {
                // Check if directly contradictory
                IntervalBound copy = it->second;
                if (!copy.intersect(sub_bound)) {
                    res.status = VerificationStatus::kFail;
                    res.evidence = "Claim '" + claim + "' contradicts constraint system interval.";
                    res.compact_observation = "Verification: FAIL | Contradicts constraint system";
                } else {
                    res.status = VerificationStatus::kUnknown;
                    res.evidence = "Claim is consistent with constraints but not strictly entailed.";
                    res.compact_observation = "Verification: UNKNOWN | Consistent but not proven";
                }
            }
            auto end_time = std::chrono::steady_clock::now();
            res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            return res;
        }
    }

    res.status = VerificationStatus::kUnknown;
    res.failure_reason = "Unsupported claim syntax for constraint verification: " + claim;
    res.compact_observation = "Verification: UNKNOWN | Unsupported syntax";
    return res;
}

} // namespace strata::logic
