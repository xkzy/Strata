// src/rt/json.cpp
#include "strata/rt/json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::rt {

namespace {
const Json& null_json() { static const Json j; return j; }

void escape_to(std::string& out, const std::string& s) {
    out += '"';
    for (size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) { char buf[8]; std::snprintf(buf, sizeof buf, "\\u%04x", c); out += buf; }
                else out += static_cast<char>(c);   // UTF-8 passes through (the producer is expected to send valid UTF-8)
        }
    }
    out += '"';
}

struct Parser {
    const std::string& s;
    size_t i = 0, max_depth;
    std::string err;
    Parser(const std::string& t, size_t d) : s(t), max_depth(d) {}

    bool fail(const char* m) { if (err.empty()) err = std::string(m) + " at byte " + std::to_string(i); return false; }
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i; }

    static void put_utf8(std::string& o, uint32_t cp) {
        if (cp < 0x80) o += static_cast<char>(cp);
        else if (cp < 0x800) { o += static_cast<char>(0xC0 | (cp >> 6)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { o += static_cast<char>(0xE0 | (cp >> 12)); o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
        else { o += static_cast<char>(0xF0 | (cp >> 18)); o += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
    }
    bool hex4(uint32_t& v) {
        if (i + 4 > s.size()) return fail("truncated \\u escape");
        v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s[i++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
            else return fail("bad \\u escape");
        }
        return true;
    }
    bool str(std::string& out) {
        if (i >= s.size() || s[i] != '"') return fail("expected string");
        ++i;
        while (i < s.size()) {
            const unsigned char c = static_cast<unsigned char>(s[i++]);
            if (c == '"') return true;
            if (c < 0x20) return fail("control character in string");
            if (c != '\\') { out += static_cast<char>(c); continue; }
            if (i >= s.size()) break;
            const char e = s[i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp < 0xDC00) {   // surrogate pair
                        if (i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                            i += 2;
                            uint32_t lo;
                            if (!hex4(lo)) return false;
                            if (lo >= 0xDC00 && lo < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            else cp = 0xFFFD;
                        } else cp = 0xFFFD;
                    } else if (cp >= 0xDC00 && cp < 0xE000) cp = 0xFFFD;
                    put_utf8(out, cp);
                    break;
                }
                default: return fail("bad escape");
            }
        }
        return fail("unterminated string");
    }
    bool value(Json& out, size_t depth) {
        if (depth > max_depth) return fail("nesting too deep");
        ws();
        if (i >= s.size()) return fail("unexpected end");
        const char c = s[i];
        if (c == '{') {
            ++i; out = Json::object(); ws();
            if (i < s.size() && s[i] == '}') { ++i; return true; }
            while (true) {
                ws();
                std::string k;
                if (!str(k)) return false;
                ws();
                if (i >= s.size() || s[i] != ':') return fail("expected ':'");
                ++i;
                Json v;
                if (!value(v, depth + 1)) return false;
                out.set(k, std::move(v));
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            ++i; out = Json::array(); ws();
            if (i < s.size() && s[i] == ']') { ++i; return true; }
            while (true) {
                Json v;
                if (!value(v, depth + 1)) return false;
                out.push(std::move(v));
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') { ++i; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"') { std::string v; if (!str(v)) return false; out = Json::string(std::move(v)); return true; }
        if (s.compare(i, 4, "true") == 0) { i += 4; out = Json::boolean(true); return true; }
        if (s.compare(i, 5, "false") == 0) { i += 5; out = Json::boolean(false); return true; }
        if (s.compare(i, 4, "null") == 0) { i += 4; out = Json(); return true; }
        // number
        size_t j = i;
        if (j < s.size() && s[j] == '-') ++j;
        const size_t digits = j;
        while (j < s.size() && s[j] >= '0' && s[j] <= '9') ++j;
        if (j == digits) return fail("unexpected character");
        bool is_int = true;
        if (j < s.size() && s[j] == '.') { is_int = false; ++j; while (j < s.size() && s[j] >= '0' && s[j] <= '9') ++j; }
        if (j < s.size() && (s[j] == 'e' || s[j] == 'E')) {
            is_int = false; ++j;
            if (j < s.size() && (s[j] == '+' || s[j] == '-')) ++j;
            while (j < s.size() && s[j] >= '0' && s[j] <= '9') ++j;
        }
        const std::string tok = s.substr(i, j - i);
        i = j;
        if (is_int && tok.size() < 19) out = Json::integer(std::strtoll(tok.c_str(), nullptr, 10));
        else out = Json::number(std::strtod(tok.c_str(), nullptr));
        return true;
    }
};
} // namespace

const std::string& Json::empty_string() { static const std::string e; return e; }

const Json& Json::operator[](const std::string& key) const {
    if (type_ != Type::kObject) return null_json();
    auto it = o_.find(key);
    return it == o_.end() ? null_json() : it->second;
}
const Json& Json::at(size_t i) const { return type_ == Type::kArray && i < a_.size() ? a_[i] : null_json(); }
Json& Json::set(const std::string& key, Json v) { if (type_ != Type::kObject) { *this = object(); } o_[key] = std::move(v); return *this; }
Json& Json::push(Json v) { if (type_ != Type::kArray) { *this = array(); } a_.push_back(std::move(v)); return *this; }

void Json::dump_to(std::string& out) const {
    switch (type_) {
        case Type::kNull: out += "null"; break;
        case Type::kBool: out += b_ ? "true" : "false"; break;
        case Type::kNumber:
            if (is_int_) out += std::to_string(i_);
            else if (std::isfinite(d_)) { char buf[40]; std::snprintf(buf, sizeof buf, "%.17g", d_); out += buf; }
            else out += "null";
            break;
        case Type::kString: escape_to(out, s_); break;
        case Type::kArray: {
            out += '[';
            bool first = true;
            for (const auto& v : a_) { if (!first) out += ','; first = false; v.dump_to(out); }
            out += ']';
            break;
        }
        case Type::kObject: {
            out += '{';
            bool first = true;
            for (const auto& kv : o_) { if (!first) out += ','; first = false; escape_to(out, kv.first); out += ':'; kv.second.dump_to(out); }
            out += '}';
            break;
        }
    }
}
std::string Json::dump() const { std::string s; dump_to(s); return s; }

bool Json::parse(const std::string& text, Json& out, std::string* error, size_t max_depth, size_t max_bytes) {
    if (text.size() > max_bytes) { if (error) *error = "message too large"; return false; }
    Parser p(text, max_depth);
    Json v;
    if (!p.value(v, 0)) { if (error) *error = p.err; return false; }
    p.ws();
    if (p.i != text.size()) { if (error) *error = "trailing characters"; return false; }
    out = std::move(v);
    return true;
}

} // namespace strata::rt
