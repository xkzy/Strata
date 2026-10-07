// src/logic_verification/claim_extractor.cpp - Claim Extraction Engine Implementation
#include "strata/logic_verification/claim_extractor.hpp"

#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>

namespace strata::logic {

ClaimExtractor::ClaimExtractor() = default;
ClaimExtractor::~ClaimExtractor() = default;

bool ClaimExtractor::has_verifiable_content(const std::string& text) const {
    if (text.empty()) return false;
    // Look for math symbols, equation signs, logic keywords, code fences, JSON braces
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
    return (lower.find('=') != std::string::npos ||
            lower.find("==") != std::string::npos ||
            lower.find("->") != std::string::npos ||
            lower.find("=>") != std::string::npos ||
            lower.find("```") != std::string::npos ||
            lower.find('{') != std::string::npos ||
            lower.find("select") != std::string::npos ||
            lower.find("premise") != std::string::npos ||
            lower.find("constraint") != std::string::npos ||
            lower.find("given") != std::string::npos ||
            lower.find("equation") != std::string::npos);
}

void ClaimExtractor::extract_arithmetic_claims(const std::string& text,
                                               std::vector<VerificationClaim>& claims,
                                               const std::string& tenant_id,
                                               const std::string& session_id) const {
    // Matches patterns like: "2384 * 7291 = 17381744" or "1/3 + 1/6 = 0.5" or "12.5 + 4.2 == 16.7"
    static const std::regex arith_regex(
        R"(([\d\.\s\+\-\*\/\^\(\)]+?)\s*(?:=|==|is equal to)\s*([\-\+]?\d+(?:\.\d+)?(?:[eE][\-\+]?\d+)?|\d+\/\d+))",
        std::regex::optimize);

    auto words_begin = std::sregex_iterator(text.begin(), text.end(), arith_regex);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        std::string expr = match[1].str();
        std::string claimed = match[2].str();

        // Clean expression and check that it contains operators
        expr.erase(0, expr.find_first_not_of(" \t\n\r"));
        expr.erase(expr.find_last_not_of(" \t\n\r") + 1);

        bool has_op = (expr.find('+') != std::string::npos ||
                       expr.find('-') != std::string::npos ||
                       expr.find('*') != std::string::npos ||
                       expr.find('/') != std::string::npos ||
                       expr.find('^') != std::string::npos);

        if (has_op && expr.length() >= 3) {
            VerificationClaim claim;
            claim.claim_id = "claim_arith_" + std::to_string(claims.size() + 1);
            claim.type = ClaimType::kArithmetic;
            claim.raw_statement = match[0].str();
            claim.expression = expr;
            claim.claimed_value = claimed;
            claim.tenant_id = tenant_id;
            claim.session_id = session_id;
            claims.push_back(claim);
        }
    }
}

void ClaimExtractor::extract_equation_claims(const std::string& text,
                                             std::vector<VerificationClaim>& claims,
                                             const std::string& tenant_id,
                                             const std::string& session_id) const {
    // Matches patterns like: "solve 2*x + 5 == 15, x = 5" or "2*x + 5 = 15 => x = 5"
    static const std::regex eq_regex(
        R"((?:solve\s+|equation:\s*)?([a-zA-Z0-9\s\+\-\*\/\^\(\)]+?)\s*(?:==|=)\s*([a-zA-Z0-9\s\+\-\*\/\^\(\)]+?)\s*(?:=>|->|therefore|,)\s*([a-zA-Z])\s*(?:=|==)\s*([\-\+]?\d+(?:\.\d+)?|\d+\/\d+))",
        std::regex::optimize | std::regex::icase);

    auto words_begin = std::sregex_iterator(text.begin(), text.end(), eq_regex);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        std::string lhs = match[1].str();
        std::string rhs = match[2].str();
        std::string var = match[3].str();
        std::string val = match[4].str();

        lhs.erase(0, lhs.find_first_not_of(" \t\n\r"));
        lhs.erase(lhs.find_last_not_of(" \t\n\r") + 1);
        rhs.erase(0, rhs.find_first_not_of(" \t\n\r"));
        rhs.erase(rhs.find_last_not_of(" \t\n\r") + 1);

        VerificationClaim claim;
        claim.claim_id = "claim_eq_" + std::to_string(claims.size() + 1);
        claim.type = ClaimType::kEquation;
        claim.raw_statement = match[0].str();
        claim.expression = lhs + " == " + rhs;
        claim.claimed_value = var + " == " + val;
        claim.tenant_id = tenant_id;
        claim.session_id = session_id;
        claims.push_back(claim);
    }
}

void ClaimExtractor::extract_proposition_claims(const std::string& text,
                                                std::vector<VerificationClaim>& claims,
                                                const std::string& tenant_id,
                                                const std::string& session_id) const {
    // Matches patterns like: "Premises: [A -> B, B -> C], Conclusion: A -> C"
    // Or "Given A and A => B, therefore B"
    static const std::regex prop_regex(
        R"((?:Premises?:\s*\[?([^\]\n]+)\]?|Given\s+([^,\n]+))\s*,?\s*(?:Conclusion:|therefore|so)\s*([a-zA-Z0-9\s\-\>\=\<\!\&\|\~]+))",
        std::regex::optimize | std::regex::icase);

    auto words_begin = std::sregex_iterator(text.begin(), text.end(), prop_regex);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        std::string premises_str = match[1].matched ? match[1].str() : match[2].str();
        std::string conclusion = match[3].str();

        VerificationClaim claim;
        claim.claim_id = "claim_prop_" + std::to_string(claims.size() + 1);
        claim.type = ClaimType::kProposition;
        claim.raw_statement = match[0].str();

        // Split premises by comma / 'and'
        std::stringstream ss(premises_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            item.erase(0, item.find_first_not_of(" \t\n\r"));
            item.erase(item.find_last_not_of(" \t\n\r") + 1);
            if (!item.empty() && item != "and") {
                claim.premises.push_back(item);
            }
        }

        conclusion.erase(0, conclusion.find_first_not_of(" \t\n\r"));
        conclusion.erase(conclusion.find_last_not_of(" \t\n\r") + 1);
        claim.claimed_value = conclusion;
        claim.expression = conclusion;
        claim.tenant_id = tenant_id;
        claim.session_id = session_id;
        claims.push_back(claim);
    }
}

void ClaimExtractor::extract_constraint_claims(const std::string& text,
                                               std::vector<VerificationClaim>& claims,
                                               const std::string& tenant_id,
                                               const std::string& session_id) const {
    // Matches patterns like: "Constraints: [x > 0, x < 10], Claim: x = 15"
    static const std::regex constr_regex(
        R"(constraints?:\s*\[?([^\]\n]+)\]?\s*,?\s*(?:claim|value|assignment):\s*([a-zA-Z0-9\s\=\<\>\!]+))",
        std::regex::optimize | std::regex::icase);

    auto words_begin = std::sregex_iterator(text.begin(), text.end(), constr_regex);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        std::string constrs_str = match[1].str();
        std::string claim_str = match[2].str();

        VerificationClaim claim;
        claim.claim_id = "claim_constr_" + std::to_string(claims.size() + 1);
        claim.type = ClaimType::kConstraint;
        claim.raw_statement = match[0].str();

        std::stringstream ss(constrs_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            item.erase(0, item.find_first_not_of(" \t\n\r"));
            item.erase(item.find_last_not_of(" \t\n\r") + 1);
            if (!item.empty()) {
                claim.constraints.push_back(item);
            }
        }

        claim_str.erase(0, claim_str.find_first_not_of(" \t\n\r"));
        claim_str.erase(claim_str.find_last_not_of(" \t\n\r") + 1);
        claim.claimed_value = claim_str;
        claim.expression = claim_str;
        claim.tenant_id = tenant_id;
        claim.session_id = session_id;
        claims.push_back(claim);
    }
}

void ClaimExtractor::extract_unit_claims(const std::string& text,
                                         std::vector<VerificationClaim>& claims,
                                         const std::string& tenant_id,
                                         const std::string& session_id) const {
    // Matches patterns like: "10 m / 2 s = 5 m/s" or "distance = 10 m, time = 2 s, velocity = 5 kg"
    static const std::regex unit_regex(
        R"(([\d\.\s\+\-\*\/\(\)\w]+?\b(?:m|s|kg|km|cm|mm|g|N|J|W|Pa|Hz|degC|degF|A|K|mol|cd)\b[\d\.\s\+\-\*\/\(\)\w]*)\s*(?:=|==|results in)\s*([\d\.\s\+\-\*\/\(\)\w]+?\b(?:m|s|kg|km|cm|mm|g|N|J|W|Pa|Hz|degC|degF|A|K|mol|cd)\b[\w\s\/\*\^]*))",
        std::regex::optimize);

    auto words_begin = std::sregex_iterator(text.begin(), text.end(), unit_regex);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        std::string left = match[1].str();
        std::string right = match[2].str();

        left.erase(0, left.find_first_not_of(" \t\n\r"));
        left.erase(left.find_last_not_of(" \t\n\r") + 1);
        right.erase(0, right.find_first_not_of(" \t\n\r"));
        right.erase(right.find_last_not_of(" \t\n\r") + 1);

        VerificationClaim claim;
        claim.claim_id = "claim_unit_" + std::to_string(claims.size() + 1);
        claim.type = ClaimType::kUnitDimension;
        claim.raw_statement = match[0].str();
        claim.unit_expression = left;
        claim.claimed_unit = right;
        claim.expression = left;
        claim.claimed_value = right;
        claim.tenant_id = tenant_id;
        claim.session_id = session_id;
        claims.push_back(claim);
    }
}

void ClaimExtractor::extract_json_and_code_claims(const std::string& text,
                                                  std::vector<VerificationClaim>& claims,
                                                  const std::string& tenant_id,
                                                  const std::string& session_id) const {
    // Matches ```json ... ``` codeblocks or ```python ... ```
    static const std::regex code_regex(
        R"(```(json|python|cpp|c\+\+|javascript|sql)?\s*\n([\s\S]*?)\n```)",
        std::regex::optimize);

    auto words_begin = std::sregex_iterator(text.begin(), text.end(), code_regex);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        std::string lang = match[1].str();
        std::string content = match[2].str();

        std::transform(lang.begin(), lang.end(), lang.begin(), ::tolower);

        VerificationClaim claim;
        claim.claim_id = "claim_code_" + std::to_string(claims.size() + 1);
        claim.raw_statement = match[0].str();
        claim.tenant_id = tenant_id;
        claim.session_id = session_id;

        if (lang == "json" || (lang.empty() && content.find('{') != std::string::npos)) {
            claim.type = ClaimType::kSchemaType;
            claim.schema_json = content;
            claim.expression = content;
        } else if (lang == "sql") {
            claim.type = ClaimType::kSQL;
            claim.sql_query = content;
            claim.expression = content;
        } else {
            claim.type = ClaimType::kCodeSyntax;
            claim.code_snippet = content;
            claim.language = lang.empty() ? "generic" : lang;
            claim.expression = content;
        }
        claims.push_back(claim);
    }
}

void ClaimExtractor::extract_sql_claims(const std::string& text,
                                        std::vector<VerificationClaim>& claims,
                                        const std::string& tenant_id,
                                        const std::string& session_id) const {
    // Matches inline SQL like: "SELECT id, name FROM users WHERE age > 18"
    static const std::regex sql_inline_regex(
        R"(\b(SELECT\s+[\s\S]+?\s+FROM\s+[a-zA-Z0-9_]+(?:\s+WHERE\s+[\s\S]+?)?);)",
        std::regex::optimize | std::regex::icase);

    auto words_begin = std::sregex_iterator(text.begin(), text.end(), sql_inline_regex);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        std::string sql = match[1].str();

        VerificationClaim claim;
        claim.claim_id = "claim_sql_" + std::to_string(claims.size() + 1);
        claim.type = ClaimType::kSQL;
        claim.raw_statement = match[0].str();
        claim.sql_query = sql;
        claim.expression = sql;
        claim.tenant_id = tenant_id;
        claim.session_id = session_id;
        claims.push_back(claim);
    }
}

std::vector<VerificationClaim> ClaimExtractor::extract_claims(const std::string& text,
                                                              const std::string& tenant_id,
                                                              const std::string& session_id) const {
    std::vector<VerificationClaim> claims;
    if (!has_verifiable_content(text)) {
        return claims;
    }

    extract_json_and_code_claims(text, claims, tenant_id, session_id);
    extract_arithmetic_claims(text, claims, tenant_id, session_id);
    extract_equation_claims(text, claims, tenant_id, session_id);
    extract_proposition_claims(text, claims, tenant_id, session_id);
    extract_constraint_claims(text, claims, tenant_id, session_id);
    extract_unit_claims(text, claims, tenant_id, session_id);
    extract_sql_claims(text, claims, tenant_id, session_id);

    return claims;
}

bool ClaimExtractor::parse_json_claim(const std::string& json_str, VerificationClaim& out_claim, std::string& err) const {
    // Basic parser for structured claim object
    // { "type": "arithmetic", "expression": "2384 * 7291", "claimed_value": "17381744" }
    // { "type": "constraint", "constraints": ["x > 0", "x < 10"], "claimed_value": "x = 15" }
    // { "type": "proposition", "premises": ["A -> B", "B -> C"], "claimed_value": "A -> C" }
    if (json_str.empty()) {
        err = "Empty JSON claim string";
        return false;
    }

    auto extract_field = [](const std::string& json, const std::string& key) -> std::string {
        std::string search = "\"" + key + "\":";
        size_t pos = json.find(search);
        if (pos == std::string::npos) {
            search = "\"" + key + "\" :";
            pos = json.find(search);
        }
        if (pos == std::string::npos) return "";

        pos += search.length();
        while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
        if (pos >= json.length()) return "";

        if (json[pos] == '"') {
            size_t end_pos = json.find('"', pos + 1);
            if (end_pos != std::string::npos) {
                return json.substr(pos + 1, end_pos - pos - 1);
            }
        }
        return "";
    };

    std::string type_str = extract_field(json_str, "type");
    std::string expr = extract_field(json_str, "expression");
    std::string claimed = extract_field(json_str, "claimed_value");
    if (claimed.empty()) claimed = extract_field(json_str, "claim");

    out_claim.claim_id = extract_field(json_str, "claim_id");
    if (out_claim.claim_id.empty()) out_claim.claim_id = "claim_json_1";

    if (type_str == "arithmetic") out_claim.type = ClaimType::kArithmetic;
    else if (type_str == "symbolic") out_claim.type = ClaimType::kSymbolic;
    else if (type_str == "equation") out_claim.type = ClaimType::kEquation;
    else if (type_str == "proposition") out_claim.type = ClaimType::kProposition;
    else if (type_str == "constraint") out_claim.type = ClaimType::kConstraint;
    else if (type_str == "consistency") out_claim.type = ClaimType::kConsistency;
    else if (type_str == "unit" || type_str == "unit_dimension") out_claim.type = ClaimType::kUnitDimension;
    else if (type_str == "schema" || type_str == "schema_type") out_claim.type = ClaimType::kSchemaType;
    else if (type_str == "code" || type_str == "code_syntax") out_claim.type = ClaimType::kCodeSyntax;
    else if (type_str == "sql") out_claim.type = ClaimType::kSQL;
    else out_claim.type = ClaimType::kUnknown;

    out_claim.expression = expr;
    out_claim.claimed_value = claimed;
    out_claim.raw_statement = json_str;
    out_claim.tenant_id = extract_field(json_str, "tenant_id");
    if (out_claim.tenant_id.empty()) out_claim.tenant_id = "default";
    out_claim.session_id = extract_field(json_str, "session_id");
    if (out_claim.session_id.empty()) out_claim.session_id = "default";

    return true;
}

} // namespace strata::logic
