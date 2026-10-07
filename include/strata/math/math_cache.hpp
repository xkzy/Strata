// include/strata/math/math_cache.hpp - Mathematical Result Cache & Content Store
//
// Separates mathematical evaluation results from model KV caches, providing
// deterministic result reuse, multi-tenant isolation, and provenance tracking.
#pragma once

#include "strata/math/math_types.hpp"

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::math {

class MathResultCache {
public:
    explicit MathResultCache(size_t max_entries = 10000);

    // Everything a result depends on: the normalized expression (parsed, so spelling cannot collide with meaning), operation,
    // mode, variable, point, order, assumptions, precision, rounding mode, engine version and IR version.
    static std::string make_cache_key(const MathRequest& req);
    static const char* engine_version() { return "cas-0.3.0"; }
    static const char* ir_version() { return "ir-1"; }
    static std::string content_hash(const std::string& text);

    // Retrieve cached result if available and valid
    bool get(const MathRequest& req, MathResult& out_result);

    // Store evaluation result in cache
    void put(const MathRequest& req, const MathResult& result);

    // Invalidate entries for a specific tenant/session or clear entire cache
    void invalidate_session(const std::string& session_id);
    void invalidate_tenant(const std::string& tenant_id);
    void clear();

    size_t size() const;
    size_t max_entries() const { return max_entries_; }

private:
    struct CacheEntry {
        MathResult result;
        std::string tenant_id;
        std::string session_id;
        uint64_t access_count = 0;
        double timestamp_sec = 0.0;
    };

    size_t max_entries_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, CacheEntry> cache_;
    std::vector<std::string> lru_order_;

    void evict_if_needed();
};

} // namespace strata::math
