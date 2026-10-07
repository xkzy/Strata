// src/logic_verification/unit_verifier.cpp - Unit & Dimensional Verification Engine Implementation
#include "strata/logic_verification/unit_verifier.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <regex>
#include <sstream>

namespace strata::logic {

std::string DimensionVector::to_dimension_string() const {
    std::ostringstream ss;
    bool first = true;
    auto append_dim = [&](const char* symbol, int exp) {
        if (exp != 0) {
            if (!first) ss << " ";
            ss << symbol;
            if (exp != 1) ss << "^" << exp;
            first = false;
        }
    };
    append_dim("L", length);
    append_dim("M", mass);
    append_dim("T", time);
    append_dim("I", current);
    append_dim("Theta", temperature);
    append_dim("N", amount);
    append_dim("J", luminous);

    if (first) return "[Dimensionless]";
    return "[" + ss.str() + "]";
}

// ---------------------------------------------------------------------------
// UnitVerifier Implementation
// ---------------------------------------------------------------------------

UnitVerifier::UnitVerifier() {
    init_unit_table();
}

UnitVerifier::~UnitVerifier() = default;

void UnitVerifier::init_unit_table() {
    // Length
    DimensionVector len{1, 0, 0, 0, 0, 0, 0};
    base_units_["m"] = len;
    base_units_["meter"] = len;
    base_units_["meters"] = len;
    base_units_["km"] = len;
    base_units_["kilometer"] = len;
    base_units_["cm"] = len;
    base_units_["centimeter"] = len;
    base_units_["mm"] = len;
    base_units_["millimeter"] = len;
    base_units_["mi"] = len;
    base_units_["mile"] = len;
    base_units_["ft"] = len;
    base_units_["foot"] = len;
    base_units_["feet"] = len;

    // Mass
    DimensionVector mass{0, 1, 0, 0, 0, 0, 0};
    base_units_["kg"] = mass;
    base_units_["kilogram"] = mass;
    base_units_["kilograms"] = mass;
    base_units_["g"] = mass;
    base_units_["gram"] = mass;
    base_units_["grams"] = mass;
    base_units_["mg"] = mass;
    base_units_["milligram"] = mass;
    base_units_["lb"] = mass;
    base_units_["pound"] = mass;

    // Time
    DimensionVector time{0, 0, 1, 0, 0, 0, 0};
    base_units_["s"] = time;
    base_units_["sec"] = time;
    base_units_["second"] = time;
    base_units_["seconds"] = time;
    base_units_["min"] = time;
    base_units_["minute"] = time;
    base_units_["minutes"] = time;
    base_units_["h"] = time;
    base_units_["hr"] = time;
    base_units_["hour"] = time;
    base_units_["hours"] = time;

    // Current
    DimensionVector curr{0, 0, 0, 1, 0, 0, 0};
    base_units_["A"] = curr;
    base_units_["amp"] = curr;
    base_units_["ampere"] = curr;
    base_units_["amperes"] = curr;

    // Temperature
    DimensionVector temp{0, 0, 0, 0, 1, 0, 0};
    base_units_["K"] = temp;
    base_units_["kelvin"] = temp;
    base_units_["degC"] = temp;
    base_units_["degF"] = temp;

    // Amount
    DimensionVector amt{0, 0, 0, 0, 0, 1, 0};
    base_units_["mol"] = amt;
    base_units_["mole"] = amt;
    base_units_["moles"] = amt;

    // Luminous
    DimensionVector lum{0, 0, 0, 0, 0, 0, 1};
    base_units_["cd"] = lum;
    base_units_["candela"] = lum;

    // Derived SI Units
    base_units_["N"] = len.multiply(mass).divide(time.power(2));            // kg*m/s^2
    base_units_["newton"] = base_units_["N"];
    base_units_["newtons"] = base_units_["N"];

    base_units_["J"] = base_units_["N"].multiply(len);                       // N*m = kg*m^2/s^2
    base_units_["joule"] = base_units_["J"];
    base_units_["joules"] = base_units_["J"];

    base_units_["W"] = base_units_["J"].divide(time);                        // J/s
    base_units_["watt"] = base_units_["W"];
    base_units_["watts"] = base_units_["W"];

    base_units_["Pa"] = base_units_["N"].divide(len.power(2));               // N/m^2
    base_units_["pascal"] = base_units_["Pa"];

    base_units_["Hz"] = DimensionVector().divide(time);                      // 1/s
    base_units_["hertz"] = base_units_["Hz"];

    base_units_["C"] = curr.multiply(time);                                  // A*s
    base_units_["coulomb"] = base_units_["C"];

    base_units_["V"] = base_units_["W"].divide(curr);                        // W/A
    base_units_["volt"] = base_units_["V"];
    base_units_["volts"] = base_units_["V"];
}

bool UnitVerifier::parse_unit(const std::string& unit_str, DimensionVector& out_dim, std::string& err) const {
    if (unit_str.empty()) {
        err = "Empty unit string";
        return false;
    }

    // Tokenize compound unit by '/' and '*'
    // e.g. "m/s", "kg*m/s^2", "10 m / 2 s", "5 m/s", "5 kg"
    // Extract numerator and denominator
    std::string s = unit_str;

    // Strip leading numeric constants if any (e.g. "10 m" -> "m")
    static const std::regex num_strip(R"(^\s*[\-\+]?\d+(?:\.\d+)?(?:[eE][\-\+]?\d+)?\s*)");
    s = std::regex_replace(s, num_strip, "");

    // Split on '/'
    size_t slash_pos = s.find('/');
    std::string num_part = (slash_pos != std::string::npos) ? s.substr(0, slash_pos) : s;
    std::string den_part = (slash_pos != std::string::npos) ? s.substr(slash_pos + 1) : "";

    auto parse_subpart = [this, &err](const std::string& part, bool is_denominator, DimensionVector& acc) -> bool {
        if (part.empty()) return true;

        // Split by '*' or space
        std::stringstream ss(part);
        std::string token;
        while (ss >> token) {
            if (token == "*") continue;

            // Strip leading number if present
            static const std::regex leading_num(R"(^[\-\+]?\d+(?:\.\d+)?(?:[eE][\-\+]?\d+)?\s*)");
            token = std::regex_replace(token, leading_num, "");
            if (token.empty()) continue;

            // Check for exponent ^2, ^-1 etc.
            int power = 1;
            size_t caret = token.find('^');
            std::string u_name = token;
            if (caret != std::string::npos) {
                u_name = token.substr(0, caret);
                try {
                    power = std::stoi(token.substr(caret + 1));
                } catch (...) {
                    power = 1;
                }
            }

            // Clean unit name
            u_name.erase(0, u_name.find_first_not_of(" \t\n\r()"));
            u_name.erase(u_name.find_last_not_of(" \t\n\r()") + 1);

            auto it = base_units_.find(u_name);
            if (it == base_units_.end()) {
                // Try case-insensitive lookup
                std::string lower = u_name;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                it = base_units_.find(lower);
            }

            if (it == base_units_.end()) {
                err = "Unknown physical unit: " + u_name;
                return false;
            }

            DimensionVector token_dim = it->second.power(power);
            if (is_denominator) {
                acc = acc.divide(token_dim);
            } else {
                acc = acc.multiply(token_dim);
            }
        }
        return true;
    };

    DimensionVector res;
    if (!parse_subpart(num_part, false, res)) return false;
    if (!parse_subpart(den_part, true, res)) return false;

    out_dim = res;
    return true;
}

VerificationResult UnitVerifier::verify_dimensions(const std::string& expression_or_derived,
                                                   const std::string& claimed_unit,
                                                   const std::string& claim_id) const {
    auto start_time = std::chrono::steady_clock::now();
    VerificationResult res;
    res.claim_id = claim_id;
    res.type = ClaimType::kUnitDimension;
    res.backend_used = "unit_verifier";

    std::string err;
    DimensionVector derived_dim;
    if (!parse_unit(expression_or_derived, derived_dim, err)) {
        res.status = VerificationStatus::kUnknown;
        res.failure_reason = "Unparseable derived unit expression: " + err;
        res.compact_observation = "Verification: UNKNOWN | Unparseable unit expression";
        return res;
    }

    DimensionVector claimed_dim;
    if (!parse_unit(claimed_unit, claimed_dim, err)) {
        res.status = VerificationStatus::kUnknown;
        res.failure_reason = "Unparseable claimed unit: " + err;
        res.compact_observation = "Verification: UNKNOWN | Unparseable claimed unit";
        return res;
    }

    if (derived_dim == claimed_dim) {
        res.status = VerificationStatus::kPass;
        res.evidence = "Dimensional equality proven: " + derived_dim.to_dimension_string() +
                       " == " + claimed_dim.to_dimension_string();
        res.expected_value = derived_dim.to_dimension_string();
        res.actual_value = claimed_dim.to_dimension_string();
        res.compact_observation = "Verification: PASS | Dimensionally consistent (" + derived_dim.to_dimension_string() + ")";
    } else {
        res.status = VerificationStatus::kFail;
        res.failure_reason = "Dimensional mismatch: derived dimensions " + derived_dim.to_dimension_string() +
                             " do not match claimed dimensions " + claimed_dim.to_dimension_string();
        res.evidence = "Physical formula violates dimensional homogeneity (" +
                       expression_or_derived + " vs " + claimed_unit + ")";
        res.expected_value = derived_dim.to_dimension_string();
        res.actual_value = claimed_dim.to_dimension_string();
        res.compact_observation = "Verification: FAIL | Dimensional mismatch (" + derived_dim.to_dimension_string() + " != " + claimed_dim.to_dimension_string() + ")";
    }

    auto end_time = std::chrono::steady_clock::now();
    res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    return res;
}

} // namespace strata::logic
