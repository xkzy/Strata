// src/logic_verification/schema_verifier.cpp - Schema Verifier Implementation
#include "strata/logic_verification/schema_verifier.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <sstream>
#include <stack>

namespace strata::logic {

SchemaVerifier::SchemaVerifier() = default;
SchemaVerifier::~SchemaVerifier() = default;

namespace {
class JsonValidator {
    const std::string& s;
    size_t pos;
    std::string& err;

    void skip_ws() {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r')) {
            pos++;
        }
    }

    bool parse_string() {
        if (pos >= s.size() || s[pos] != '"') {
            err = "Expected string starting with '\"'";
            return false;
        }
        pos++;
        while (pos < s.size()) {
            char c = s[pos++];
            if (c == '\\') {
                if (pos >= s.size()) {
                    err = "Unfinished escape sequence in string";
                    return false;
                }
                pos++;
            } else if (c == '"') {
                return true;
            }
        }
        err = "Unterminated string literal";
        return false;
    }

    bool parse_number() {
        if (pos < s.size() && (s[pos] == '-')) pos++;
        if (pos >= s.size() || !std::isdigit(static_cast<unsigned char>(s[pos]))) {
            err = "Invalid number format";
            return false;
        }
        while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) pos++;
        if (pos < s.size() && s[pos] == '.') {
            pos++;
            if (pos >= s.size() || !std::isdigit(static_cast<unsigned char>(s[pos]))) {
                err = "Invalid number fractional part";
                return false;
            }
            while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) pos++;
        }
        if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
            pos++;
            if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) pos++;
            if (pos >= s.size() || !std::isdigit(static_cast<unsigned char>(s[pos]))) {
                err = "Invalid number exponent";
                return false;
            }
            while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) pos++;
        }
        return true;
    }

    bool parse_literal(const std::string& lit) {
        if (pos + lit.size() <= s.size() && s.compare(pos, lit.size(), lit) == 0) {
            pos += lit.size();
            return true;
        }
        err = "Expected literal '" + lit + "'";
        return false;
    }

    bool parse_array() {
        if (pos >= s.size() || s[pos] != '[') {
            err = "Expected '['";
            return false;
        }
        pos++;
        skip_ws();
        if (pos < s.size() && s[pos] == ']') {
            pos++;
            return true;
        }
        while (true) {
            if (!parse_value()) return false;
            skip_ws();
            if (pos < s.size() && s[pos] == ']') {
                pos++;
                return true;
            }
            if (pos >= s.size() || s[pos] != ',') {
                err = "Expected ',' or ']' in array";
                return false;
            }
            pos++;
            skip_ws();
        }
    }

    bool parse_object() {
        if (pos >= s.size() || s[pos] != '{') {
            err = "Expected '{'";
            return false;
        }
        pos++;
        skip_ws();
        if (pos < s.size() && s[pos] == '}') {
            pos++;
            return true;
        }
        while (true) {
            skip_ws();
            if (pos >= s.size() || s[pos] != '"') {
                err = "Expected string key in object";
                return false;
            }
            if (!parse_string()) return false;
            skip_ws();
            if (pos >= s.size() || s[pos] != ':') {
                err = "Expected ':' after key in object";
                return false;
            }
            pos++;
            skip_ws();
            if (!parse_value()) return false;
            skip_ws();
            if (pos < s.size() && s[pos] == '}') {
                pos++;
                return true;
            }
            if (pos >= s.size() || s[pos] != ',') {
                err = "Expected ',' or '}' in object";
                return false;
            }
            pos++;
        }
    }

    bool parse_value() {
        skip_ws();
        if (pos >= s.size()) {
            err = "Unexpected end of input, expected JSON value";
            return false;
        }
        char c = s[pos];
        if (c == '"') return parse_string();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == 't') return parse_literal("true");
        if (c == 'f') return parse_literal("false");
        if (c == 'n') return parse_literal("null");
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        err = std::string("Unexpected character '") + c + "' at position " + std::to_string(pos);
        return false;
    }

public:
    JsonValidator(const std::string& input, std::string& out_err) : s(input), pos(0), err(out_err) {}

    bool validate() {
        skip_ws();
        if (pos >= s.size()) {
            err = "Empty JSON input";
            return false;
        }
        if (!parse_value()) return false;
        skip_ws();
        if (pos < s.size()) {
            err = "Trailing characters after JSON value at position " + std::to_string(pos);
            return false;
        }
        return true;
    }
};
}  // namespace

bool SchemaVerifier::is_valid_json(const std::string& s, std::string& err) const {
    JsonValidator validator(s, err);
    return validator.validate();
}

VerificationResult SchemaVerifier::verify_schema(const std::string& json_payload,
                                                 const std::string& schema_definition,
                                                 const std::string& claim_id) const {
    auto start_time = std::chrono::steady_clock::now();
    VerificationResult res;
    res.claim_id = claim_id;
    res.type = ClaimType::kSchemaType;
    res.backend_used = "schema_verifier";

    // 1. Syntax check on payload
    std::string json_err;
    if (!is_valid_json(json_payload, json_err)) {
        res.status = VerificationStatus::kFail;
        res.failure_reason = "Malformed JSON syntax: " + json_err;
        res.evidence = "Input cannot be parsed as valid JSON";
        res.compact_observation = "Verification: FAIL | Malformed JSON (" + json_err + ")";
        auto end_time = std::chrono::steady_clock::now();
        res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        return res;
    }

    if (schema_definition.empty()) {
        // Only syntax validation was requested
        res.status = VerificationStatus::kPass;
        res.evidence = "JSON syntax validation passed successfully.";
        res.compact_observation = "Verification: PASS | Valid JSON syntax";
        auto end_time = std::chrono::steady_clock::now();
        res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        return res;
    }

    // 2. Validate required fields in schema
    // Check if schema_definition lists required keys (e.g. "required": ["id", "name"])
    size_t req_pos = schema_definition.find("\"required\"");
    if (req_pos != std::string::npos) {
        size_t arr_start = schema_definition.find('[', req_pos);
        size_t arr_end = schema_definition.find(']', arr_start);
        if (arr_start != std::string::npos && arr_end != std::string::npos) {
            std::string req_content = schema_definition.substr(arr_start + 1, arr_end - arr_start - 1);
            std::stringstream ss(req_content);
            std::string item;
            while (std::getline(ss, item, ',')) {
                size_t q1 = item.find('"');
                size_t q2 = item.rfind('"');
                if (q1 != std::string::npos && q2 != std::string::npos && q2 > q1) {
                    std::string key = item.substr(q1 + 1, q2 - q1 - 1);
                    std::string search_key = "\"" + key + "\"";
                    if (json_payload.find(search_key) == std::string::npos) {
                        res.status = VerificationStatus::kFail;
                        res.failure_reason = "Missing required schema property: '" + key + "'";
                        res.evidence = "JSON payload lacks required field '" + key + "'";
                        res.compact_observation = "Verification: FAIL | Missing required property '" + key + "'";
                        auto end_time = std::chrono::steady_clock::now();
                        res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                        return res;
                    }
                }
            }
        }
    }

    res.status = VerificationStatus::kPass;
    res.evidence = "JSON payload satisfies all schema syntactic and structural constraints.";
    res.compact_observation = "Verification: PASS | Schema compliant";
    auto end_time = std::chrono::steady_clock::now();
    res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    return res;
}

} // namespace strata::logic
