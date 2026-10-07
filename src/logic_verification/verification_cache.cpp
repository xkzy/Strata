// src/logic_verification/verification_cache.cpp - Verification Cache Implementation
#include "strata/logic_verification/verification_cache.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace strata::logic {

VerificationCache::VerificationCache(size_t max_entries)
    : max_entries_(max_entries) {}

VerificationCache::~VerificationCache() = default;

std::string VerificationCache::generate_key(const VerificationClaim& claim) const {
    // Canonical hash key from tenant, claim type, expression, claimed value, and premises/constraints
    std::ostringstream ss;
    ss << claim.tenant_id << "|"
       << static_cast<int>(claim.type) << "|"
       << claim.expression << "|"
       << claim.claimed_value << "|";

    for (const auto& p : claim.premises) ss << p << ",";
    ss << "|";
    for (const auto& c : claim.constraints) ss << c << ",";
    ss << "|" << claim.tolerance.abs_tol << "|" << claim.tolerance.rel_tol;

    // Fast 64-bit FNV-1a hash formatted as hex string
    std::string str = ss.str();
    uint64_t hash = 14695981039346656037ULL;
    for (char ch : str) {
        hash ^= static_cast<uint64_t>(static_cast<unsigned char>(ch));
        hash *= 1099511628211ULL;
    }

    std::ostringstream hex_ss;
    hex_ss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return hex_ss.str();
}

bool VerificationCache::get(const VerificationClaim& claim, VerificationResult& out_result) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = generate_key(claim);

    auto it = table_.find(key);
    if (it == table_.end()) {
        return false;
    }

    // Move to front (MRU)
    lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
    out_result = it->second->result;
    out_result.cache_hit = true;
    return true;
}

void VerificationCache::put(const VerificationClaim& claim, const VerificationResult& result) {
    // Only cache PASS or FAIL results (UNKNOWN may be due to temporary resource limits)
    if (result.status == VerificationStatus::kUnknown) return;

    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = generate_key(claim);

    auto it = table_.find(key);
    if (it != table_.end()) {
        it->second->result = result;
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
        return;
    }

    if (table_.size() >= max_entries_) {
        // Evict LRU
        auto last = lru_list_.end();
        --last;
        table_.erase(last->key);
        lru_list_.pop_back();
    }

    lru_list_.push_front({key, result});
    table_[key] = lru_list_.begin();
}

void VerificationCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    table_.clear();
    lru_list_.clear();
}

size_t VerificationCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return table_.size();
}

} // namespace strata::logic
