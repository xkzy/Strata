// src/tools/tool_store.cpp - Content-Addressed Tool State Store Implementation
#include "strata/tools/tool_store.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <sstream>

namespace strata::tools {

// ------------------- Content Addressed Store -------------------
ContentAddressedStore::ContentAddressedStore() = default;

std::string ContentAddressedStore::compute_hash(const std::string& content) {
    // 64-bit FNV-1a Hash
    uint64_t hash = 14695981039346656037ULL;
    for (char c : content) {
        hash ^= static_cast<uint8_t>(c);
        hash *= 1099511628211ULL;
    }
    std::ostringstream ss;
    ss << std::hex << std::setw(16) << std::setfill('0') << hash;
    return ss.str();
}

std::string ContentAddressedStore::store(const std::string& raw_content) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    std::string h = compute_hash(raw_content);
    if (store_.find(h) == store_.end()) {
        store_[h] = raw_content;
        total_bytes_ += raw_content.size();
    }
    return h;
}

bool ContentAddressedStore::get(const std::string& content_hash, std::string& out_content) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = store_.find(content_hash);
    if (it != store_.end()) {
        out_content = it->second;
        return true;
    }
    return false;
}

bool ContentAddressedStore::contains(const std::string& content_hash) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return store_.find(content_hash) != store_.end();
}

size_t ContentAddressedStore::total_stored_entries() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return store_.size();
}

uint64_t ContentAddressedStore::total_stored_bytes() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return total_bytes_;
}

// ------------------- Tool State Store -------------------
ToolStateStore::ToolStateStore() = default;
ToolStateStore::~ToolStateStore() = default;

int64_t ToolStateStore::save_result(ToolResult result) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    int64_t id = next_result_id_++;
    result.result_id = id;

    // Store raw output into content-addressed store
    if (!result.raw_output.empty()) {
        result.content_hash = content_store_.store(result.raw_output);
        result.raw_bytes = result.raw_output.size();
    }

    total_raw_tokens_ += result.raw_tokens;
    results_[id] = result;
    return id;
}

const ToolResult* ToolStateStore::get_result(int64_t result_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = results_.find(result_id);
    if (it != results_.end()) {
        return &it->second;
    }
    return nullptr;
}

std::string ToolStateStore::retrieve_fragment(const ToolRetrievalQuery& query) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = results_.find(query.result_id);
    if (it == results_.end()) {
        return "[Error: Tool Result " + std::to_string(query.result_id) + " not found]";
    }

    std::string full_content;
    std::string_view content_view;
    if (content_store_.get(it->second.content_hash, full_content)) {
        content_view = full_content;
    } else {
        content_view = it->second.raw_output;
    }

    std::string out;
    out.reserve(std::min<size_t>(content_view.size(), 4096));

    int cur_line = 0;
    int matches_found = 0;
    int lines_included = 0;

    size_t pos = 0;
    const size_t len = content_view.size();
    while (pos < len) {
        size_t next_nl = content_view.find('\n', pos);
        size_t line_end = (next_nl != std::string_view::npos) ? next_nl : len;
        std::string_view line = content_view.substr(pos, line_end - pos);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        cur_line++;

        bool matches = true;
        if (!query.query.empty()) {
            matches = (line.find(query.query) != std::string_view::npos);
        }
        if (matches && !query.filter_keyword.empty()) {
            matches = (line.find(query.filter_keyword) != std::string_view::npos);
        }

        if (matches && cur_line >= query.line_start) {
            out += "L";
            out += std::to_string(cur_line);
            out += ": ";
            out.append(line.data(), line.size());
            out += "\n";
            lines_included++;
            matches_found++;
            if (lines_included >= query.line_count) break;
        }

        if (next_nl == std::string_view::npos) break;
        pos = next_nl + 1;
    }

    if (matches_found == 0) {
        return "[No matching lines found in Tool Result " + std::to_string(query.result_id) + "]";
    }
    return out;
}

ContextLease ToolStateStore::materialize_lease(int64_t result_id, int64_t max_tokens,
                                              double duration_sec) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    ContextLease lease;
    auto it = results_.find(result_id);
    if (it == results_.end()) {
        return lease;
    }

    lease.lease_id = next_lease_id_++;
    lease.result_id = result_id;

    std::string content;
    if (!content_store_.get(it->second.content_hash, content)) {
        content = it->second.raw_output;
    }

    size_t max_chars = static_cast<size_t>(max_tokens * 4);
    if (content.size() > max_chars) {
        lease.materialized_content = content.substr(0, max_chars) + "\n... [Lease truncated at max_tokens]";
        lease.token_count = max_tokens;
    } else {
        lease.materialized_content = content;
        lease.token_count = static_cast<int64_t>(content.size() / 4);
    }

    lease.expires_timestamp = duration_sec;
    lease.is_active = true;

    active_leases_[lease.lease_id] = lease;
    return lease;
}

void ToolStateStore::release_lease(int64_t lease_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    active_leases_.erase(lease_id);
}

void ToolStateStore::log_event(const ToolEvent& event) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    event_log_.push_back(event);
}

std::vector<ToolEvent> ToolStateStore::get_recent_events(size_t limit) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (event_log_.size() <= limit) {
        return event_log_;
    }
    return std::vector<ToolEvent>(event_log_.end() - limit, event_log_.end());
}

size_t ToolStateStore::total_results() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return results_.size();
}

uint64_t ToolStateStore::total_raw_bytes() const {
    return content_store_.total_stored_bytes();
}

uint64_t ToolStateStore::total_raw_tokens() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return total_raw_tokens_;
}

} // namespace strata::tools
