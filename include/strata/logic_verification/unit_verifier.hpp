// include/strata/logic_verification/unit_verifier.hpp - Unit & Dimensional Verification Engine
//
// Performs strict SI dimensional analysis [Length, Mass, Time, Current, Temp, Amount, Luminous]
// to verify physical formula consistency and flag dimensional mismatches deterministically.
#pragma once

#include "strata/logic_verification/verification_types.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace strata::logic {

struct DimensionVector {
    int length = 0;      // m
    int mass = 0;        // kg
    int time = 0;        // s
    int current = 0;     // A
    int temperature = 0; // K
    int amount = 0;      // mol
    int luminous = 0;    // cd

    bool operator==(const DimensionVector& o) const {
        return length == o.length && mass == o.mass && time == o.time &&
               current == o.current && temperature == o.temperature &&
               amount == o.amount && luminous == o.luminous;
    }
    bool operator!=(const DimensionVector& o) const { return !(*this == o); }

    DimensionVector multiply(const DimensionVector& o) const {
        return {length + o.length, mass + o.mass, time + o.time,
                current + o.current, temperature + o.temperature,
                amount + o.amount, luminous + o.luminous};
    }

    DimensionVector divide(const DimensionVector& o) const {
        return {length - o.length, mass - o.mass, time - o.time,
                current - o.current, temperature - o.temperature,
                amount - o.amount, luminous - o.luminous};
    }

    DimensionVector power(int p) const {
        return {length * p, mass * p, time * p,
                current * p, temperature * p,
                amount * p, luminous * p};
    }

    std::string to_dimension_string() const;
};

class UnitVerifier {
public:
    UnitVerifier();
    ~UnitVerifier();

    // Verify dimensional consistency between expression and claimed unit
    VerificationResult verify_dimensions(const std::string& expression_or_derived,
                                         const std::string& claimed_unit,
                                         const std::string& claim_id = "unit_1") const;

    // Parse compound unit string into DimensionVector
    bool parse_unit(const std::string& unit_str, DimensionVector& out_dim, std::string& err) const;

private:
    std::unordered_map<std::string, DimensionVector> base_units_;
    void init_unit_table();
};

} // namespace strata::logic
