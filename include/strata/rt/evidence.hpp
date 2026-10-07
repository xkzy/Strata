// include/strata/rt/evidence.hpp - evidence abstraction: where a claim can be checked against
#pragma once

#include "strata/rt/types.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::rt {

enum class EvidenceKind { kExactState, kRag, kToolResult, kMemory, kSourceCode, kSchema, kConversation };

inline const char* to_string(EvidenceKind k) {
    switch (k) {
        case EvidenceKind::kExactState: return "exact_state";
        case EvidenceKind::kRag: return "rag";
        case EvidenceKind::kToolResult: return "tool_result";
        case EvidenceKind::kMemory: return "memory";
        case EvidenceKind::kSourceCode: return "source_code";
        case EvidenceKind::kSchema: return "schema";
        case EvidenceKind::kConversation: return "conversation";
    }
    return "rag";
}

struct Evidence {
    std::string id;               // stable id of this piece
    EvidenceKind kind = EvidenceKind::kRag;
    std::string source_id;        // what it came from (document, tool result, memory key); invalidation unit
    std::string source_version;   // changes when the source changes (content hash or explicit version)
    std::string text;
    std::string hash;             // hash of text
    bool authoritative = false;   // exact state (a config store) outranks retrieved prose
    double score = 0.0;
};

// A key/value fact read from evidence ("port: 8080", "the timeout is 30 seconds").
struct Fact {
    std::string key;     // normalised: lower case, separators -> spaces
    std::string value;
    std::string evidence_id;
    bool authoritative = false;
};

std::string normalize_key(const std::string& key);
// Parses line-oriented config (key: value, key = value, "key": value) and simple prose ("the port is 8080").
std::vector<Fact> extract_facts(const Evidence& ev);
// True if two subjects refer to the same thing ("server port" ~ "port").
bool subjects_match(const std::string& a, const std::string& b);
// Value comparison: numbers numerically ("1,200" == "1200"), words case-insensitively, units spaced alike.
bool values_equal(const std::string& a, const std::string& b);

struct EvidenceSet {
    std::vector<Evidence> items;
    // order-independent hash of (id, source_version) pairs: changes exactly when any piece or its version changes
    std::string hash() const;
    std::vector<std::string> source_ids() const;
    bool empty() const { return items.empty(); }
    EvidenceState state() const { return items.empty() ? EvidenceState::kNone : EvidenceState::kAvailable; }
};

class IEvidenceProvider {
public:
    virtual ~IEvidenceProvider() = default;
    virtual std::string name() const = 0;
    // Must honour the scope: a provider never returns what the requester may not read.
    virtual std::vector<Evidence> retrieve(const RequestScope& scope, const std::string& query, size_t top_k) = 0;
};

// Authoritative key/value state (configuration, environment, tool-reported facts). Versioned and scoped.
class ExactStateStore : public IEvidenceProvider {
public:
    // Sets key = value for the scope; bumps the entry's version when the value changes.
    void put(const context::SecurityScope& owner, context::SharingScope sharing, const std::string& source_id,
             const std::string& key, const std::string& value);
    bool erase(const context::SecurityScope& owner, const std::string& source_id, const std::string& key);
    std::string name() const override { return "exact_state"; }
    std::vector<Evidence> retrieve(const RequestScope& scope, const std::string& query, size_t top_k) override;

private:
    struct Entry {
        context::SecurityScope owner;
        context::SharingScope sharing = context::SharingScope::PRIVATE;
        std::string source_id, key, value;
        uint64_t version = 1;
    };
    mutable std::mutex mu_;
    std::vector<Entry> entries_;
};

// Adapter over the existing multi-tenant context: scoped, permission-aware document retrieval (RAG).
class DocumentEvidenceProvider : public IEvidenceProvider {
public:
    explicit DocumentEvidenceProvider(std::shared_ptr<context::MultiTenantContextManager> mtc) : mtc_(std::move(mtc)) {}
    std::string name() const override { return "documents"; }
    std::vector<Evidence> retrieve(const RequestScope& scope, const std::string& query, size_t top_k) override;

private:
    std::shared_ptr<context::MultiTenantContextManager> mtc_;
};

// Evidence that arrived with the request itself (tool results, pasted files): never leaves the request.
class RequestEvidenceProvider : public IEvidenceProvider {
public:
    void add(const std::string& request_key, Evidence ev);
    void drop(const std::string& request_key);
    std::string name() const override { return "request"; }
    std::vector<Evidence> retrieve(const RequestScope& scope, const std::string& query, size_t top_k) override;

private:
    mutable std::mutex mu_;
    std::map<std::string, std::vector<Evidence>> by_request_;
};

} // namespace strata::rt
