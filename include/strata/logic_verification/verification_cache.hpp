// include/strata/logic_verification/verification_cache.hpp - Verification Result Cache
//
// Multi-tenant, content-addressed LRU cache for deterministic verification results,
// preventing duplicate verification work across recurring queries and agent sessions.
#pragma once

#include "strata/logic_verification/verification_types.hpp"

#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

namespace strata::logic {

class VerificationCache {
public:
    explicit VerificationCache(size_t max_entries = 10000);
    ~VerificationCache();

    // Generate canonical deterministic cache key
    std::string generate_key(const VerificationClaim& claim) const;

    // Cache operations
    bool get(const VerificationClaim& claim, VerificationResult& out_result);
    void put(const VerificationClaim& claim, const VerificationResult& result);

    void clear();
    size_t size() const;
    size_t max_entries() const { return max_entries_; }

private:
    size_t max_entries_;
    mutable std::mutex mutex_;

    struct CacheEntry {
        std::string key;
        VerificationResult result;
    };

    std::list<CacheEntry> lru_list_;
    std::unordered_map<std::string, std::list<CacheEntry>::iterator> table_;
};

} // namespace strata::logic
