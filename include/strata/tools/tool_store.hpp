// include/strata/tools/tool_store.hpp - Content-Addressed Tool State Store & Context Lease
//
// Maintains persistent tool execution state, content-addressed raw storage,
// sub-result fragment search, and temporary context materialization.
#pragma once

#include "strata/tools/tool_types.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::tools {

struct ToolRetrievalQuery {
    int64_t result_id = 0;
    std::string query;
    int line_start = 0;
    int line_count = 50;
    std::string filter_keyword;
    int64_t max_tokens = 500;
};

struct ContextLease {
    int64_t lease_id = 0;
    int64_t result_id = 0;
    std::string materialized_content;
    int64_t token_count = 0;
    double expires_timestamp = 0.0;
    bool is_active = true;
};

class ContentAddressedStore {
public:
    ContentAddressedStore();

    // Store raw text data; returns unique content hash
    std::string store(const std::string& raw_content);

    // Retrieve raw text by hash
    bool get(const std::string& content_hash, std::string& out_content) const;

    // Check existence
    bool contains(const std::string& content_hash) const;

    size_t total_stored_entries() const;
    uint64_t total_stored_bytes() const;

    static std::string compute_hash(const std::string& content);

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::string> store_;
    uint64_t total_bytes_ = 0;
};

class ToolStateStore {
public:
    ToolStateStore();
    ~ToolStateStore();

    // Register a tool execution result
    int64_t save_result(ToolResult result);

    // Retrieve complete ToolResult by ID
    const ToolResult* get_result(int64_t result_id) const;

    // Retrieve specific fragments/lines of a tool result on-demand
    std::string retrieve_fragment(const ToolRetrievalQuery& query) const;

    // Temporary Context Materialization: Lease content for active model context
    ContextLease materialize_lease(int64_t result_id, int64_t max_tokens = 1000,
                                  double duration_sec = 60.0);

    // Release an active lease
    void release_lease(int64_t lease_id);

    // Event audit trail
    void log_event(const ToolEvent& event);
    std::vector<ToolEvent> get_recent_events(size_t limit = 50) const;

    // Store statistics
    size_t total_results() const;
    uint64_t total_raw_bytes() const;
    uint64_t total_raw_tokens() const;

    ContentAddressedStore& content_store() { return content_store_; }

private:
    mutable std::mutex mutex_;
    int64_t next_result_id_ = 1;
    int64_t next_lease_id_ = 1;

    ContentAddressedStore content_store_;
    std::unordered_map<int64_t, ToolResult> results_;
    std::unordered_map<int64_t, ContextLease> active_leases_;
    std::vector<ToolEvent> event_log_;

    uint64_t total_raw_tokens_ = 0;
};

} // namespace strata::tools
