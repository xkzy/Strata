// src/math/math_cache.cpp - Mathematical Result Cache & Content Store Implementation
#include "strata/math/math_cache.hpp"
#include "strata/math/expression_parser.hpp"

#include <chrono>
#include <sstream>

namespace strata::math {

MathResultCache::MathResultCache(size_t max_entries)
    : max_entries_(max_entries) {}

std::string MathResultCache::make_cache_key(const MathRequest& req) {
    std::ostringstream ss;
    std::string canon = ExpressionParser::canonicalize(req.expression);

    ss << math_operation_to_string(req.operation) << "|"
       << math_mode_to_string(req.mode) << "|"
       << canon << "|"
       << req.variable << "|"
       << req.point << "|"
       << req.order << "|"
       << req.assumptions << "|"
       << req.precision_digits;

    return ss.str();
}

bool MathResultCache::get(const MathRequest& req, MathResult& out_result) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = make_cache_key(req);

    auto it = cache_.find(key);
    if (it == cache_.end()) {
        return false;
    }

    it->second.access_count++;
    out_result = it->second.result;
    out_result.cache_hit = true;
    return true;
}

void MathResultCache::put(const MathRequest& req, const MathResult& result) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = make_cache_key(req);

    evict_if_needed();

    CacheEntry entry;
    entry.result = result;
    entry.tenant_id = req.tenant_id;
    entry.session_id = req.session_id;
    entry.access_count = 1;
    entry.timestamp_sec = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    cache_[key] = entry;
    lru_order_.push_back(key);
}

void MathResultCache::invalidate_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = cache_.begin(); it != cache_.end(); ) {
        if (it->second.session_id == session_id) {
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
}

void MathResultCache::invalidate_tenant(const std::string& tenant_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = cache_.begin(); it != cache_.end(); ) {
        if (it->second.tenant_id == tenant_id) {
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
}

void MathResultCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    lru_order_.clear();
}

size_t MathResultCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

void MathResultCache::evict_if_needed() {
    if (cache_.size() >= max_entries_ && !lru_order_.empty()) {
        std::string oldest_key = lru_order_.front();
        lru_order_.erase(lru_order_.begin());
        cache_.erase(oldest_key);
    }
}

} // namespace strata::math
