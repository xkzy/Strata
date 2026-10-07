// include/strata/logic_verification/rule_engine.hpp - Propositional & First-Order Rule Engine
//
// Deterministic proof and truth checker: AND, OR, NOT, IMPLIES, EQUIV, modus ponens,
// syllogisms, contradictions, and truth-table model validation without neural approximations.
#pragma once

#include "strata/logic_verification/verification_types.hpp"

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::logic {

enum class PropositionNodeType {
    kVariable = 0,
    kNot,
    kAnd,
    kOr,
    kImplies,
    kEquivalence,
    kConstant
};

struct PropositionNode {
    PropositionNodeType type = PropositionNodeType::kVariable;
    std::string variable_name;
    bool constant_value = false;
    std::shared_ptr<PropositionNode> left;
    std::shared_ptr<PropositionNode> right;

    bool evaluate(const std::unordered_map<std::string, bool>& valuation) const;
    void collect_variables(std::set<std::string>& vars) const;
    std::string to_string() const;
};

class RuleEngine {
public:
    RuleEngine();
    ~RuleEngine();

    // Verify logical deduction: Premises |= Conclusion
    VerificationResult verify_deduction(const std::vector<std::string>& premises,
                                        const std::string& conclusion,
                                        const std::string& claim_id = "prop_1") const;

    // Verify single proposition is a tautology (valid under all truth assignments)
    VerificationResult verify_tautology(const std::string& proposition,
                                         const std::string& claim_id = "taut_1") const;

    // Check for contradiction in a set of propositions
    bool check_contradiction(const std::vector<std::string>& propositions,
                             std::string& out_conflicting_pair) const;

    // Parse text formula to AST node
    std::shared_ptr<PropositionNode> parse_formula(const std::string& formula, std::string& err) const;

private:
    std::shared_ptr<PropositionNode> parse_implication(const std::string& s, size_t& pos, std::string& err) const;
    std::shared_ptr<PropositionNode> parse_or(const std::string& s, size_t& pos, std::string& err) const;
    std::shared_ptr<PropositionNode> parse_and(const std::string& s, size_t& pos, std::string& err) const;
    std::shared_ptr<PropositionNode> parse_unary(const std::string& s, size_t& pos, std::string& err) const;
    std::shared_ptr<PropositionNode> parse_primary(const std::string& s, size_t& pos, std::string& err) const;

    void skip_whitespace(const std::string& s, size_t& pos) const;
};

} // namespace strata::logic
