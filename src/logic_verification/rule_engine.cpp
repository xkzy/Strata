// src/logic_verification/rule_engine.cpp - Rule Engine Implementation
#include "strata/logic_verification/rule_engine.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>

namespace strata::logic {

bool PropositionNode::evaluate(const std::unordered_map<std::string, bool>& valuation) const {
    switch (type) {
        case PropositionNodeType::kConstant:
            return constant_value;
        case PropositionNodeType::kVariable: {
            auto it = valuation.find(variable_name);
            if (it != valuation.end()) return it->second;
            return false;
        }
        case PropositionNodeType::kNot:
            return left ? !left->evaluate(valuation) : true;
        case PropositionNodeType::kAnd:
            return (left && right) ? (left->evaluate(valuation) && right->evaluate(valuation)) : false;
        case PropositionNodeType::kOr:
            return (left && right) ? (left->evaluate(valuation) || right->evaluate(valuation)) : false;
        case PropositionNodeType::kImplies:
            // A -> B is equivalent to !A || B
            return (left && right) ? (!left->evaluate(valuation) || right->evaluate(valuation)) : false;
        case PropositionNodeType::kEquivalence:
            return (left && right) ? (left->evaluate(valuation) == right->evaluate(valuation)) : false;
    }
    return false;
}

void PropositionNode::collect_variables(std::set<std::string>& vars) const {
    if (type == PropositionNodeType::kVariable && !variable_name.empty()) {
        vars.insert(variable_name);
    }
    if (left) left->collect_variables(vars);
    if (right) right->collect_variables(vars);
}

std::string PropositionNode::to_string() const {
    switch (type) {
        case PropositionNodeType::kConstant:
            return constant_value ? "true" : "false";
        case PropositionNodeType::kVariable:
            return variable_name;
        case PropositionNodeType::kNot:
            return "!" + (left ? left->to_string() : "");
        case PropositionNodeType::kAnd:
            return "(" + (left ? left->to_string() : "") + " & " + (right ? right->to_string() : "") + ")";
        case PropositionNodeType::kOr:
            return "(" + (left ? left->to_string() : "") + " | " + (right ? right->to_string() : "") + ")";
        case PropositionNodeType::kImplies:
            return "(" + (left ? left->to_string() : "") + " -> " + (right ? right->to_string() : "") + ")";
        case PropositionNodeType::kEquivalence:
            return "(" + (left ? left->to_string() : "") + " <-> " + (right ? right->to_string() : "") + ")";
    }
    return "";
}

// ---------------------------------------------------------------------------
// RuleEngine Recursive Descent Parser
// ---------------------------------------------------------------------------

RuleEngine::RuleEngine() = default;
RuleEngine::~RuleEngine() = default;

void RuleEngine::skip_whitespace(const std::string& s, size_t& pos) const {
    while (pos < s.length() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r')) {
        pos++;
    }
}

std::shared_ptr<PropositionNode> RuleEngine::parse_primary(const std::string& s, size_t& pos, std::string& err) const {
    skip_whitespace(s, pos);
    if (pos >= s.length()) {
        err = "Unexpected end of expression";
        return nullptr;
    }

    if (s[pos] == '(') {
        pos++;
        auto node = parse_implication(s, pos, err);
        if (!node) return nullptr;
        skip_whitespace(s, pos);
        if (pos >= s.length() || s[pos] != ')') {
            err = "Expected closing parenthesis ')'";
            return nullptr;
        }
        pos++;
        return node;
    }

    // Check for boolean literal
    if (s.compare(pos, 4, "true") == 0 || s.compare(pos, 4, "TRUE") == 0) {
        pos += 4;
        auto node = std::make_shared<PropositionNode>();
        node->type = PropositionNodeType::kConstant;
        node->constant_value = true;
        return node;
    }
    if (s.compare(pos, 5, "false") == 0 || s.compare(pos, 5, "FALSE") == 0) {
        pos += 5;
        auto node = std::make_shared<PropositionNode>();
        node->type = PropositionNodeType::kConstant;
        node->constant_value = false;
        return node;
    }

    // Variable identifier
    if (std::isalpha(static_cast<unsigned char>(s[pos])) || s[pos] == '_') {
        size_t start = pos;
        while (pos < s.length() && (std::isalnum(static_cast<unsigned char>(s[pos])) || s[pos] == '_')) {
            pos++;
        }
        std::string var = s.substr(start, pos - start);
        auto node = std::make_shared<PropositionNode>();
        node->type = PropositionNodeType::kVariable;
        node->variable_name = var;
        return node;
    }

    err = std::string("Unexpected character: ") + s[pos];
    return nullptr;
}

std::shared_ptr<PropositionNode> RuleEngine::parse_unary(const std::string& s, size_t& pos, std::string& err) const {
    skip_whitespace(s, pos);
    if (pos >= s.length()) return nullptr;

    if (s[pos] == '!' || s[pos] == '~') {
        pos++;
        auto sub = parse_unary(s, pos, err);
        if (!sub) return nullptr;
        auto node = std::make_shared<PropositionNode>();
        node->type = PropositionNodeType::kNot;
        node->left = sub;
        return node;
    }

    if (s.compare(pos, 3, "not") == 0 && (pos + 3 == s.length() || !std::isalnum(static_cast<unsigned char>(s[pos + 3])))) {
        pos += 3;
        auto sub = parse_unary(s, pos, err);
        if (!sub) return nullptr;
        auto node = std::make_shared<PropositionNode>();
        node->type = PropositionNodeType::kNot;
        node->left = sub;
        return node;
    }

    return parse_primary(s, pos, err);
}

std::shared_ptr<PropositionNode> RuleEngine::parse_and(const std::string& s, size_t& pos, std::string& err) const {
    auto left = parse_unary(s, pos, err);
    if (!left) return nullptr;

    while (true) {
        skip_whitespace(s, pos);
        if (pos >= s.length()) break;

        bool is_and = false;
        if (s.compare(pos, 2, "&&") == 0) {
            pos += 2;
            is_and = true;
        } else if (s[pos] == '&' || s[pos] == '*') {
            pos += 1;
            is_and = true;
        } else if (s.compare(pos, 3, "and") == 0 && (pos + 3 == s.length() || !std::isalnum(static_cast<unsigned char>(s[pos + 3])))) {
            pos += 3;
            is_and = true;
        }

        if (is_and) {
            auto right = parse_unary(s, pos, err);
            if (!right) return nullptr;
            auto parent = std::make_shared<PropositionNode>();
            parent->type = PropositionNodeType::kAnd;
            parent->left = left;
            parent->right = right;
            left = parent;
        } else {
            break;
        }
    }
    return left;
}

std::shared_ptr<PropositionNode> RuleEngine::parse_or(const std::string& s, size_t& pos, std::string& err) const {
    auto left = parse_and(s, pos, err);
    if (!left) return nullptr;

    while (true) {
        skip_whitespace(s, pos);
        if (pos >= s.length()) break;

        bool is_or = false;
        if (s.compare(pos, 2, "||") == 0) {
            pos += 2;
            is_or = true;
        } else if (s[pos] == '|' || s[pos] == '+') {
            pos += 1;
            is_or = true;
        } else if (s.compare(pos, 2, "or") == 0 && (pos + 2 == s.length() || !std::isalnum(static_cast<unsigned char>(s[pos + 2])))) {
            pos += 2;
            is_or = true;
        }

        if (is_or) {
            auto right = parse_and(s, pos, err);
            if (!right) return nullptr;
            auto parent = std::make_shared<PropositionNode>();
            parent->type = PropositionNodeType::kOr;
            parent->left = left;
            parent->right = right;
            left = parent;
        } else {
            break;
        }
    }
    return left;
}

std::shared_ptr<PropositionNode> RuleEngine::parse_implication(const std::string& s, size_t& pos, std::string& err) const {
    auto left = parse_or(s, pos, err);
    if (!left) return nullptr;

    skip_whitespace(s, pos);
    if (pos >= s.length()) return left;

    if (s.compare(pos, 2, "->") == 0 || s.compare(pos, 2, "=>") == 0) {
        pos += 2;
        auto right = parse_implication(s, pos, err); // Right-associative
        if (!right) return nullptr;
        auto parent = std::make_shared<PropositionNode>();
        parent->type = PropositionNodeType::kImplies;
        parent->left = left;
        parent->right = right;
        return parent;
    }

    if (s.compare(pos, 3, "<->") == 0 || s.compare(pos, 3, "<=>") == 0) {
        pos += 3;
        auto right = parse_implication(s, pos, err);
        if (!right) return nullptr;
        auto parent = std::make_shared<PropositionNode>();
        parent->type = PropositionNodeType::kEquivalence;
        parent->left = left;
        parent->right = right;
        return parent;
    }

    if (s.compare(pos, 7, "implies") == 0 && (pos + 7 == s.length() || !std::isalnum(static_cast<unsigned char>(s[pos + 7])))) {
        pos += 7;
        auto right = parse_implication(s, pos, err);
        if (!right) return nullptr;
        auto parent = std::make_shared<PropositionNode>();
        parent->type = PropositionNodeType::kImplies;
        parent->left = left;
        parent->right = right;
        return parent;
    }

    return left;
}

std::shared_ptr<PropositionNode> RuleEngine::parse_formula(const std::string& formula, std::string& err) const {
    size_t pos = 0;
    auto ast = parse_implication(formula, pos, err);
    if (!ast) return nullptr;
    skip_whitespace(formula, pos);
    if (pos < formula.length()) {
        err = std::string("Trailing unexpected characters at index ") + std::to_string(pos) + ": " + formula.substr(pos);
        return nullptr;
    }
    return ast;
}

// ---------------------------------------------------------------------------
// Verification Logic (Truth Table & Model Checker)
// ---------------------------------------------------------------------------

VerificationResult RuleEngine::verify_deduction(const std::vector<std::string>& premises,
                                                const std::string& conclusion,
                                                const std::string& claim_id) const {
    auto start_time = std::chrono::steady_clock::now();
    VerificationResult res;
    res.claim_id = claim_id;
    res.type = ClaimType::kProposition;
    res.backend_used = "rule_engine";

    // 1. Parse Conclusion
    std::string err;
    auto concl_ast = parse_formula(conclusion, err);
    if (!concl_ast) {
        res.status = VerificationStatus::kUnknown;
        res.failure_reason = "Unparseable conclusion proposition: " + err;
        res.evidence = "Syntax does not match supported formal logic grammar";
        res.compact_observation = "Verification: UNKNOWN | Reason: unparseable proposition";
        return res;
    }

    // 2. Parse Premises
    std::vector<std::shared_ptr<PropositionNode>> premise_asts;
    for (const auto& p : premises) {
        auto p_ast = parse_formula(p, err);
        if (!p_ast) {
            res.status = VerificationStatus::kUnknown;
            res.failure_reason = "Unparseable premise: " + p + " (" + err + ")";
            res.evidence = "Unsupported formal logical grammar in premise";
            res.compact_observation = "Verification: UNKNOWN | Reason: unparseable premise";
            return res;
        }
        premise_asts.push_back(p_ast);
    }

    // 3. Collect all distinct propositional variables
    std::set<std::string> var_set;
    concl_ast->collect_variables(var_set);
    for (const auto& p_ast : premise_asts) {
        p_ast->collect_variables(var_set);
    }

    std::vector<std::string> vars(var_set.begin(), var_set.end());
    if (vars.size() > 16) {
        res.status = VerificationStatus::kUnknown;
        res.failure_reason = "Propositional formula contains > 16 variables; exceeds deterministic table bound";
        res.compact_observation = "Verification: UNKNOWN | Reason: variable count exceeds exact solver bound";
        return res;
    }

    // 4. Exhaustive Truth Table Search (2^N model evaluations)
    uint64_t total_models = 1ULL << vars.size();
    bool premise_satisfiable = false;

    for (uint64_t i = 0; i < total_models; ++i) {
        std::unordered_map<std::string, bool> valuation;
        for (size_t v = 0; v < vars.size(); ++v) {
            valuation[vars[v]] = ((i >> v) & 1ULL) != 0;
        }

        // Check if all premises hold under this valuation
        bool premises_hold = true;
        for (const auto& p_ast : premise_asts) {
            if (!p_ast->evaluate(valuation)) {
                premises_hold = false;
                break;
            }
        }

        if (premises_hold) {
            premise_satisfiable = true;
            // If premises hold, conclusion MUST hold
            if (!concl_ast->evaluate(valuation)) {
                // Found a countermodel!
                res.status = VerificationStatus::kFail;
                std::ostringstream ev;
                ev << "Countermodel found: ";
                for (size_t v = 0; v < vars.size(); ++v) {
                    ev << vars[v] << "=" << (valuation[vars[v]] ? "true" : "false") << " ";
                }
                ev << "makes all premises TRUE but conclusion FALSE.";
                res.evidence = ev.str();
                res.failure_reason = "Logical deduction is invalid; countermodel exists";
                res.compact_observation = "Verification: FAIL | Deduction invalid (countermodel found)";
                auto end_time = std::chrono::steady_clock::now();
                res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                return res;
            }
        }
    }

    if (!premise_satisfiable && !premise_asts.empty()) {
        res.status = VerificationStatus::kFail;
        res.failure_reason = "Premises are inherently contradictory (no model satisfies all premises)";
        res.evidence = "Contradiction detected in premise set; principle of explosion prevented";
        res.compact_observation = "Verification: FAIL | Contradictory premises";
        auto end_time = std::chrono::steady_clock::now();
        res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        return res;
    }

    // Valid in all models!
    res.status = VerificationStatus::kPass;
    res.evidence = "Logical deduction formally proven across all " + std::to_string(total_models) + " truth valuations.";
    res.expected_value = conclusion;
    res.actual_value = conclusion;
    res.compact_observation = "Verification: PASS | Deduction formally proven";
    auto end_time = std::chrono::steady_clock::now();
    res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    return res;
}

VerificationResult RuleEngine::verify_tautology(const std::string& proposition,
                                                const std::string& claim_id) const {
    return verify_deduction({}, proposition, claim_id);
}

bool RuleEngine::check_contradiction(const std::vector<std::string>& propositions,
                                     std::string& out_conflicting_pair) const {
    std::string err;
    std::vector<std::shared_ptr<PropositionNode>> asts;
    std::set<std::string> var_set;

    for (const auto& p : propositions) {
        auto ast = parse_formula(p, err);
        if (!ast) return false;
        ast->collect_variables(var_set);
        asts.push_back(ast);
    }

    if (var_set.size() > 16) return false;

    std::vector<std::string> vars(var_set.begin(), var_set.end());
    uint64_t total_models = 1ULL << vars.size();

    for (uint64_t i = 0; i < total_models; ++i) {
        std::unordered_map<std::string, bool> valuation;
        for (size_t v = 0; v < vars.size(); ++v) {
            valuation[vars[v]] = ((i >> v) & 1ULL) != 0;
        }

        bool all_hold = true;
        for (const auto& ast : asts) {
            if (!ast->evaluate(valuation)) {
                all_hold = false;
                break;
            }
        }
        if (all_hold) {
            return false; // Satisfiable, not a contradiction
        }
    }

    out_conflicting_pair = "Contradiction across propositional set";
    return true;
}

} // namespace strata::logic
