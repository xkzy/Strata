// include/strata/rt/json.hpp - a small strict JSON value for the runtime's line protocol
//
// Not a general library: just enough for `strata_rt_server`'s messages. Parsing is bounded (nesting depth, size) because
// the peer is another process whose input is ultimately a network client's.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strata::rt {

class Json {
public:
    enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

    Json() = default;
    static Json null() { return Json(); }
    static Json boolean(bool b) { Json j; j.type_ = Type::kBool; j.b_ = b; return j; }
    static Json number(double d) { Json j; j.type_ = Type::kNumber; j.d_ = d; j.is_int_ = false; return j; }
    static Json integer(int64_t v) { Json j; j.type_ = Type::kNumber; j.i_ = v; j.d_ = static_cast<double>(v); j.is_int_ = true; return j; }
    static Json string(std::string s) { Json j; j.type_ = Type::kString; j.s_ = std::move(s); return j; }
    static Json array() { Json j; j.type_ = Type::kArray; return j; }
    static Json object() { Json j; j.type_ = Type::kObject; return j; }

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::kNull; }
    bool is_string() const { return type_ == Type::kString; }
    bool is_array() const { return type_ == Type::kArray; }
    bool is_object() const { return type_ == Type::kObject; }

    // lookups return a shared null value when absent / of another type, so chains never crash
    const Json& operator[](const std::string& key) const;
    const Json& at(size_t i) const;
    size_t size() const { return type_ == Type::kArray ? a_.size() : type_ == Type::kObject ? o_.size() : 0; }
    bool has(const std::string& key) const { return type_ == Type::kObject && o_.count(key) > 0; }

    const std::string& str(const std::string& dflt = empty_string()) const { return type_ == Type::kString ? s_ : dflt; }
    int64_t i64(int64_t dflt = 0) const { return type_ != Type::kNumber ? dflt : is_int_ ? i_ : static_cast<int64_t>(d_); }
    double num(double dflt = 0.0) const { return type_ == Type::kNumber ? d_ : dflt; }
    bool boolean_or(bool dflt = false) const { return type_ == Type::kBool ? b_ : dflt; }

    Json& set(const std::string& key, Json v);   // object
    Json& push(Json v);                          // array
    const std::vector<Json>& items() const { return a_; }
    const std::map<std::string, Json>& members() const { return o_; }

    std::string dump() const;
    // false (and `error` set) on malformed input or when a limit is exceeded
    static bool parse(const std::string& text, Json& out, std::string* error = nullptr, size_t max_depth = 64, size_t max_bytes = 256u << 20);

    static const std::string& empty_string();

private:
    Type type_ = Type::kNull;
    bool b_ = false;
    bool is_int_ = false;
    int64_t i_ = 0;
    double d_ = 0.0;
    std::string s_;
    std::vector<Json> a_;
    std::map<std::string, Json> o_;
    void dump_to(std::string& out) const;
};

} // namespace strata::rt
