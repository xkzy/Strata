// src/context/multi_tenant_context.cpp - Multi-Tenant Hierarchical Context & Memory Isolation Implementation
#include "strata/context/multi_tenant_context.hpp"

#include <algorithm>
#include <chrono>

namespace strata::context {

MultiTenantContextManager::MultiTenantContextManager(int64_t default_physical_limit)
    : default_physical_limit_(default_physical_limit),
      shared_index_(std::make_unique<HierarchicalBM25Index>()) {}

std::shared_ptr<VirtualContextManager> MultiTenantContextManager::get_or_create_session(const SecurityScope& scope) {
    std::unique_lock lock(mutex_);
    const std::string key = scope.session_id;

    auto it = sessions_.find(key);
    if (it != sessions_.end()) {
        // Enforce that requester owns the session
        const auto& owner = session_scopes_[key];
        if (owner.tenant_id != scope.tenant_id || owner.user_id != scope.user_id) {
            return nullptr; // Access denied: Tenant/User boundary violation
        }
        return it->second;
    }

    // Check tenant quota
    auto q_it = tenant_quotas_.find(scope.tenant_id);
    TenantQuota quota = (q_it != tenant_quotas_.end()) ? q_it->second : TenantQuota{};

    size_t tenant_session_count = 0;
    for (const auto& [sid, sscope] : session_scopes_) {
        if (sscope.tenant_id == scope.tenant_id) {
            tenant_session_count++;
        }
    }
    if (tenant_session_count >= quota.max_sessions) {
        return nullptr; // Quota exceeded
    }

    auto vctx = std::make_shared<VirtualContextManager>(default_physical_limit_);
    sessions_[key] = vctx;
    session_scopes_[key] = scope;
    tenant_metrics_[scope.tenant_id].total_requests++;

    return vctx;
}

bool MultiTenantContextManager::delete_session(const SecurityScope& scope) {
    std::unique_lock lock(mutex_);
    const std::string key = scope.session_id;

    auto it = sessions_.find(key);
    if (it == sessions_.end()) {
        return false;
    }

    const auto& owner = session_scopes_[key];
    if (owner.tenant_id != scope.tenant_id || owner.user_id != scope.user_id) {
        return false; // Access denied
    }

    sessions_.erase(it);
    session_scopes_.erase(key);
    return true;
}

std::shared_ptr<VirtualContextManager> MultiTenantContextManager::fork_session(
    const SecurityScope& parent_scope,
    const SecurityScope& new_scope) {
    std::unique_lock lock(mutex_);
    const std::string parent_key = parent_scope.session_id;

    auto it = sessions_.find(parent_key);
    if (it == sessions_.end()) {
        return nullptr;
    }

    const auto& owner = session_scopes_[parent_key];
    if (!new_scope.can_access(owner, SharingScope::USER)) {
        return nullptr; // Cannot fork across unauthorized tenants/users
    }

    auto child_ctx = std::make_shared<VirtualContextManager>(default_physical_limit_);
    // Cloned child session starts isolated
    sessions_[new_scope.session_id] = child_ctx;
    session_scopes_[new_scope.session_id] = new_scope;
    return child_ctx;
}

std::vector<SearchResult> MultiTenantContextManager::retrieve(
    const std::string& query,
    const SecurityScope& requester,
    size_t top_k) {
    std::shared_lock lock(mutex_);
    std::vector<SearchResult> combined_results;

    // 1. Search private session context if session exists
    auto it = sessions_.find(requester.session_id);
    if (it != sessions_.end()) {
        const auto& owner = session_scopes_[requester.session_id];
        if (owner.tenant_id == requester.tenant_id && owner.user_id == requester.user_id) {
            auto session_results = it->second->retrieve_relevant(query, top_k);
            combined_results.insert(combined_results.end(), session_results.begin(), session_results.end());
        }
    }

    // 2. Search shared / project knowledge (strictly pre-filtering by authorized scope)
    for (const auto& doc : shared_documents_) {
        if (requester.can_access(doc.owner, doc.sharing)) {
            // Document is in authorized candidate set
            double score = 0.0;
            std::string q_lower = query;
            std::string text_lower = doc.title + " " + doc.content;
            if (text_lower.find(q_lower) != std::string::npos) {
                score = 5.0;
            }
            if (score > 0.0) {
                SearchResult res;
                res.item_id = doc.item->id();
                res.relevance_score = score;
                res.matched_content = doc.content;
                res.type = doc.item->type();
                res.token_count = doc.item->token_count();
                res.filename = doc.title;
                combined_results.push_back(res);
            }
        }
    }

    // Sort ranked results
    std::sort(combined_results.begin(), combined_results.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.relevance_score > b.relevance_score;
    });

    if (combined_results.size() > top_k) {
        combined_results.resize(top_k);
    }
    return combined_results;
}

void MultiTenantContextManager::ingest_shared_knowledge(
    const SecurityScope& owner_scope,
    const std::string& title,
    const std::string& content,
    SharingScope sharing) {
    std::unique_lock lock(mutex_);

    auto item = std::make_shared<ContextItem>(
        static_cast<int64_t>(shared_documents_.size() + 1),
        ContextItemType::kFileContent,
        content,
        std::max<int64_t>(1, content.size() / 4),
        0.0
    );
    item->mutable_metadata().filename = title;
    item->mutable_metadata().tags = {owner_scope.workspace_id, owner_scope.tenant_id};

    SharedDocument doc;
    doc.owner = owner_scope;
    doc.sharing = sharing;
    doc.title = title;
    doc.content = content;
    doc.item = item;

    shared_documents_.push_back(doc);
    shared_index_->index_item(*item);
}

void MultiTenantContextManager::set_tenant_quota(const std::string& tenant_id, const TenantQuota& quota) {
    std::unique_lock lock(mutex_);
    tenant_quotas_[tenant_id] = quota;
}

TenantQuota MultiTenantContextManager::get_tenant_quota(const std::string& tenant_id) const {
    std::shared_lock lock(mutex_);
    auto it = tenant_quotas_.find(tenant_id);
    return (it != tenant_quotas_.end()) ? it->second : TenantQuota{};
}

ScopeMetrics MultiTenantContextManager::get_tenant_metrics(const std::string& tenant_id) const {
    std::shared_lock lock(mutex_);
    auto it = tenant_metrics_.find(tenant_id);
    ScopeMetrics metrics = (it != tenant_metrics_.end()) ? it->second : ScopeMetrics{};

    for (const auto& [sid, sscope] : session_scopes_) {
        if (sscope.tenant_id == tenant_id) {
            auto v_it = sessions_.find(sid);
            if (v_it != sessions_.end()) {
                auto stats = v_it->second->get_stats();
                metrics.virtual_tokens += stats.virtual_context_total_tokens;
                metrics.compaction_events += stats.compaction_events;
            }
        }
    }
    return metrics;
}

ScopeMetrics MultiTenantContextManager::get_session_metrics(const SecurityScope& scope) const {
    std::shared_lock lock(mutex_);
    ScopeMetrics metrics;
    auto it = sessions_.find(scope.session_id);
    if (it != sessions_.end()) {
        auto stats = it->second->get_stats();
        metrics.virtual_tokens = stats.virtual_context_total_tokens;
        metrics.compaction_events = stats.compaction_events;
    }
    return metrics;
}

size_t MultiTenantContextManager::total_active_sessions() const {
    std::shared_lock lock(mutex_);
    return sessions_.size();
}

} // namespace strata::context
