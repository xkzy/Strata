// include/strata/logic_verification/constraint_engine.hpp - Constraint & Interval Engine
//
// Verifies linear constraints, interval boundaries, variable domain assignments,
// and detects conflicting bounds without neural guesswork.
#pragma once

#include "strata/logic_verification/verification_types.hpp"

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::logic {

struct IntervalBound {
    double min_val = -std::numeric_limits<double>::infinity();
    double max_val = std::numeric_limits<double>::infinity();
    bool min_inclusive = false;
    bool max_inclusive = false;
    bool has_min = false;
    bool has_max = false;
    std::vector<double> excluded_values;

    bool contains(double v, double tol = 1e-9) const;
    bool is_empty() const;
    bool intersect(const IntervalBound& other);
};

class ConstraintEngine {
public:
    ConstraintEngine();
    ~ConstraintEngine();

    // Verify a claim (e.g. "x = 15" or "x > 5") against a set of constraints (e.g. ["x > 0", "x < 10"])
    VerificationResult verify_constraints(const std::vector<std::string>& constraints,
                                          const std::string& claim,
                                          const std::string& claim_id = "constr_1") const;

    // Check if a system of constraints is internally consistent
    bool check_consistency(const std::vector<std::string>& constraints,
                           std::string& out_conflict_explanation) const;

private:
    bool parse_constraint(const std::string& s, std::string& out_var, IntervalBound& out_bound,
                          std::string& out_raw_op, std::string& err) const;
    bool parse_assignment(const std::string& s, std::string& out_var, double& out_val, std::string& err) const;
};

} // namespace strata::logic
