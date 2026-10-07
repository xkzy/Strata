// SPDX-License-Identifier: GPL-3.0-or-later
// include/strata/math/theorems.hpp - Exhaustive mathematical theorem database and deterministic verifier
#pragma once

#include "strata/math/verify.hpp"
#include "strata/math/math_types.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <map>

namespace strata::math {

enum class TheoremCategory {
    kNumberTheory,
    kAlgebra,
    kGroupTheory,
    kLinearAlgebra,
    kCalculus,
    kComplexAnalysis,
    kSpecialFunctions,
    kInequality,
    kTrigonometry,
    kProbability,
    kDiscreteMath,
    kGraphTheory,
    kGeometry,
    kTopology,
    kInformationTheory,
    kCryptography
};

inline const char* to_string(TheoremCategory cat) {
    switch (cat) {
        case TheoremCategory::kNumberTheory: return "NumberTheory";
        case TheoremCategory::kAlgebra: return "Algebra";
        case TheoremCategory::kGroupTheory: return "GroupTheory";
        case TheoremCategory::kLinearAlgebra: return "LinearAlgebra";
        case TheoremCategory::kCalculus: return "Calculus";
        case TheoremCategory::kComplexAnalysis: return "ComplexAnalysis";
        case TheoremCategory::kSpecialFunctions: return "SpecialFunctions";
        case TheoremCategory::kInequality: return "Inequality";
        case TheoremCategory::kTrigonometry: return "Trigonometry";
        case TheoremCategory::kProbability: return "Probability";
        case TheoremCategory::kDiscreteMath: return "DiscreteMath";
        case TheoremCategory::kGraphTheory: return "GraphTheory";
        case TheoremCategory::kGeometry: return "Geometry";
        case TheoremCategory::kTopology: return "Topology";
        case TheoremCategory::kInformationTheory: return "InformationTheory";
        case TheoremCategory::kCryptography: return "Cryptography";
        default: return "Unknown";
    }
}

struct TheoremDef {
    std::string id;
    std::string name;
    std::vector<std::string> aliases;
    TheoremCategory category;
    std::string statement;
    std::string formula;
    std::vector<std::string> preconditions;
    std::string literature_ref;
    
    // Verifier function: given arguments and assumptions, returns deterministic verification verdict.
    std::function<VerifyResult(const std::vector<std::string>& args, const std::string& assumptions)> verifier;
};

class TheoremEngine {
public:
    TheoremEngine();
    ~TheoremEngine();

    // Query theorems
    const TheoremDef* find_theorem(const std::string& name_or_alias) const;
    std::vector<const TheoremDef*> list_all() const;
    std::vector<const TheoremDef*> list_by_category(TheoremCategory cat) const;
    std::vector<const TheoremDef*> search(const std::string& query) const;

    // Direct theorem verification
    VerifyResult verify_theorem(const std::string& name_or_alias,
                               const std::vector<std::string>& args,
                               const std::string& assumptions = "") const;

    // Pattern matching from text claim (e.g. "Fermat's little theorem for a=3, p=7")
    VerifyResult verify_claim(const std::string& claim, const std::string& assumptions = "") const;

    static TheoremEngine& instance();

private:
    void register_all_theorems();
    void register_number_theory();
    void register_algebra_and_groups();
    void register_linear_algebra();
    void register_calculus_and_analysis();
    void register_complex_analysis();
    void register_special_functions();
    void register_inequalities();
    void register_trigonometry();
    void register_probability_and_stats();
    void register_discrete_and_graphs();
    void register_geometry_and_topology();
    void register_information_and_crypto();

    std::vector<TheoremDef> theorems_;
    std::map<std::string, size_t> lookup_map_; // lowercase name/alias -> index
};

} // namespace strata::math
