// include/strata/context/multi_tenant_context.hpp - Multi-Tenant Hierarchical Context & Memory Isolation
#pragma once

#include "strata/context/context_item.hpp"
#include "strata/context/virtual_context.hpp"
#include "strata/context/retrieval_index.hpp"

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace strata::context {

enum class SharingScope {
    PRIVATE,   // Only accessible by exact session
    SESSION,   // Same as private
    AGENT,     // Shared across all sessions of the same AgentID
    USER,      // Shared across all agents of the same UserID
    PROJECT,   // Shared across all users in the same WorkspaceID / ProjectID
    TEAM,      // Shared across all users in the same TenantID
    GLOBAL     // Public / read-only globally
};

struct SecurityScope {
    std::string tenant_id = "default_tenant";
    std::string user_id = "default_user";
    std::string workspace_id = "default_workspace";
    std::string agent_id = "default_agent";
    std::string session_id = "default_session";
    SharingScope sharing_scope = SharingScope::PRIVATE;
    std::unordered_set<std::string> permissions{"read", "write"};

    bool can_access(const SecurityScope& target_owner, SharingScope target_sharing) const {
        if (target_sharing == SharingScope::GLOBAL) return true;
        if (tenant_id != target_owner.tenant_id) return false;

        switch (target_sharing) {
            case SharingScope::TEAM:
                return true;
            case SharingScope::PROJECT:
                return workspace_id == target_owner.workspace_id;
            case SharingScope::USER:
                return user_id == target_owner.user_id;
            case SharingScope::AGENT:
                return user_id == target_owner.user_id && agent_id == target_owner.agent_id;
            case SharingScope::PRIVATE:
            case SharingScope::SESSION:
            default:
                return user_id == target_owner.user_id && session_id == target_owner.session_id;
        }
    }

    std::string to_string() const {
        return tenant_id + "/" + user_id + "/" + workspace_id + "/" + agent_id + "/" + session_id;
    }
};

struct TenantQuota {
    size_t max_concurrent_requests = 16;
    size_t max_virtual_tokens = 5000000;
    size_t max_stored_data_bytes = 100 * 1024 * 1024; // 100 MB
    size_t max_sessions = 64;
    size_t max_generation_tokens = 4096;
};

struct ScopeMetrics {
    size_t active_requests = 0;
    size_t total_requests = 0;
    size_t input_tokens = 0;
    size_t output_tokens = 0;
    size_t tool_bytes_stored = 0;
    size_t virtual_tokens = 0;
    size_t compaction_events = 0;
};

class MultiTenantContextManager {
public:
    explicit MultiTenantContextManager(int64_t default_physical_limit = 8192);

    // Session Management
    std::shared_ptr<VirtualContextManager> get_or_create_session(const SecurityScope& scope);
    bool delete_session(const SecurityScope& scope);
    std::shared_ptr<VirtualContextManager> fork_session(const SecurityScope& parent_scope, const SecurityScope& new_scope);

    // Scoped Multi-Tenant Retrieval (filters unauthorized items BEFORE ranking)
    std::vector<SearchResult> retrieve(
        const std::string& query,
        const SecurityScope& requester,
        size_t top_k = 5
    );

    // Ingest Shared Project / Workspace Knowledge
    void ingest_shared_knowledge(
        const SecurityScope& owner_scope,
        const std::string& title,
        const std::string& content,
        SharingScope sharing = SharingScope::PROJECT
    );

    // Quotas & Accounting
    void set_tenant_quota(const std::string& tenant_id, const TenantQuota& quota);
    TenantQuota get_tenant_quota(const std::string& tenant_id) const;
    ScopeMetrics get_tenant_metrics(const std::string& tenant_id) const;
    ScopeMetrics get_session_metrics(const SecurityScope& scope) const;

    size_t total_active_sessions() const;

private:
    int64_t default_physical_limit_;
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<VirtualContextManager>> sessions_; // session_id -> VirtualContextManager
    std::unordered_map<std::string, SecurityScope> session_scopes_;                   // session_id -> SecurityScope
    std::unordered_map<std::string, TenantQuota> tenant_quotas_;
    std::unordered_map<std::string, ScopeMetrics> tenant_metrics_;

    // Shared Knowledge Store with Scope Metadata
    struct SharedDocument {
        SecurityScope owner;
        SharingScope sharing;
        std::string title;
        std::string content;
        std::shared_ptr<ContextItem> item;
    };
    std::vector<SharedDocument> shared_documents_;
    std::unique_ptr<HierarchicalBM25Index> shared_index_;
};

} // namespace strata::context
