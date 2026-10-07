// src/rt/types.cpp - hashing for the transparent runtime
#include "strata/rt/types.hpp"

#include <cstdio>

namespace strata::rt {

namespace {
uint64_t fnv(const std::string& s, uint64_t h) {
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}
uint64_t mix(uint64_t x) {   // splitmix64 finaliser: decorrelates the two lanes
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}
} // namespace

std::string hash128(const std::string& s) {
    const uint64_t a = mix(fnv(s, 14695981039346656037ULL));
    const uint64_t b = mix(fnv(s, 0x9E3779B97F4A7C15ULL) ^ (static_cast<uint64_t>(s.size()) * 0xD6E8FEB86659FD93ULL));
    char buf[40];
    std::snprintf(buf, sizeof buf, "%016llx%016llx", static_cast<unsigned long long>(a), static_cast<unsigned long long>(b));
    return buf;
}

std::string hash_parts(const std::vector<std::string>& parts) {
    std::string joined;
    for (const auto& p : parts) {
        joined += std::to_string(p.size());
        joined += ':';
        joined += p;
        joined += ';';
    }
    return hash128(joined);
}

} // namespace strata::rt
