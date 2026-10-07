// include/strata/rt/virtual_context.hpp - the virtual context window
//
// A VirtualContext is the logical, ordered, persistent context of one session. Everything the agent has ever sent or
// received (messages, files, tool results, artifacts) stays in it, byte for byte. Only a working set of it is ever
// materialized into the model's physical context. Paging, retrieval, summaries and eviction are internal: nothing here
// is a tool, and nothing in a materialized prompt says that something was paged.
//
// Rules this file keeps:
//   * original content is authoritative; summaries are caches and never replace it
//   * nothing is deleted; "eviction" moves a page to a colder tier and the original stays recoverable
//   * every read is authorized against the requester's scope BEFORE it is scored or returned
//   * unavailable context is reported as unavailable, never invented
#pragma once

#include "strata/rt/generator.hpp"
#include "strata/rt/types.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

namespace strata::rt {

enum class ContextTier { kHot, kWarm, kCold, kArchive };
enum class Provenance { kInternalContext, kExternalKnowledge };   // earlier session state vs files/project/documents
enum class ItemKind { kSystem, kUser, kAssistant, kToolResult, kFile, kArtifact, kNote };

inline const char* to_string(ContextTier t) {
    switch (t) { case ContextTier::kHot: return "HOT"; case ContextTier::kWarm: return "WARM"; case ContextTier::kCold: return "COLD"; case ContextTier::kArchive: return "ARCHIVE"; }
    return "COLD";
}
inline const char* to_string(Provenance p) { return p == Provenance::kInternalContext ? "internal" : "external"; }

// ---- page: the unit of paging ----
struct ContextPage {
    uint64_t page_id = 0;
    uint64_t item_id = 0;
    uint64_t sequence_position = 0;       // position of the page in the logical context (monotonic)
    uint64_t token_begin = 0, token_end = 0;   // logical token range
    std::string content_hash;
    double importance = 0.0;              // last computed (deterministic signals, see VirtualContext::importance)
    double recency = 0.0;                 // 0..1, relative to the newest page
    std::vector<uint64_t> dependency;     // pages this page needs in order to make sense
    std::string source;                   // "conversation", a file path, a tool name
    uint32_t access_frequency = 0;
    uint32_t verification_refs = 0;       // times this page backed a verified claim
    std::string summary;                  // cache only
    ContextTier tier = ContextTier::kCold;
    Provenance provenance = Provenance::kInternalContext;
    ItemKind kind = ItemKind::kUser;
    std::string role;                     // role the content is materialized with
    bool pinned = false;
    int priority = 0;
    uint32_t chunk_index = 0, chunk_count = 1;
    size_t tokens = 0;
    context::SharingScope sharing = context::SharingScope::PRIVATE;
    context::SecurityScope owner;
    uint64_t version = 1;
};

// ---- storage ----
// Where cold content can live. Content is addressed by its hash, so identical content is stored once.
class IPageBackend {
public:
    virtual ~IPageBackend() = default;
    virtual bool put(const std::string& hash, const std::string& content) = 0;
    virtual bool get(const std::string& hash, std::string& content) = 0;
    virtual bool has(const std::string& hash) = 0;
};
class FilePageBackend : public IPageBackend {
public:
    explicit FilePageBackend(std::string dir);
    bool put(const std::string& hash, const std::string& content) override;
    bool get(const std::string& hash, std::string& content) override;
    bool has(const std::string& hash) override;
    const std::string& dir() const { return dir_; }
private:
    std::string dir_;
    std::string path(const std::string& hash) const;
};

// ---- configuration ----
struct VirtualContextConfig {
    bool enabled = true;
    int64_t physical_tokens = 8192;          // what the underlying model can hold at once
    int64_t advertised_virtual_tokens = 10000000;   // configured ceiling of the logical space
    uint64_t storage_budget_bytes = 8ull << 30;     // external state this context may use
    uint64_t resident_bytes = 256ull << 20;         // page content kept in memory before cold pages spill to the backend
    size_t chunk_tokens = 400;               // target size of a page
    size_t summary_chars = 240;
    double recent_fraction = 0.45;           // share of the budget reserved for the newest items
    double pinned_fraction = 0.20;
    size_t retrieval_candidates = 48;
    int dependency_depth = 2;
    double min_relevance = 0.12;             // normalized; below this a page is not "found"
    size_t warm_pages = 256;
    double hot_bonus = 0.15;                 // keeps the working set (and its KV prefix) stable between requests
    // importance weights
    double w_relevance = 1.0, w_recency = 0.35, w_frequency = 0.12, w_dependents = 0.15, w_pinned = 2.0, w_tool = 0.10, w_verified = 0.20;
};

struct ContextWindowInfo {
    int64_t physical = 0;            // tokens the model processes at once
    int64_t virtual_advertised = 0;  // configured ceiling
    int64_t virtual_capacity = 0;    // what storage/index limits actually allow: min(configured, storage-derived)
    int64_t virtual_used = 0;        // logical tokens held right now
    int64_t resident = 0;            // tokens currently materialized (last working set)
    std::string to_json() const;
};

// ---- selection ----
struct SelectionRequest {
    std::string query;                          // what the model has to answer now
    int64_t budget_tokens = 4096;               // physical budget for context (prompt minus output/evidence reserve)
    std::set<uint64_t> current_items;           // items of the present request: always materialized, never summarized
    context::SecurityScope requester;
    std::vector<std::string> extra_queries;     // e.g. recovery asks for more
};

struct WorkingSet {
    std::vector<Message> messages;
    std::vector<uint64_t> page_ids;             // in prompt order
    std::vector<std::string> page_hashes;
    int64_t tokens = 0;
    size_t paged_in = 0;                        // pages not in the previous working set
    size_t paged_out = 0;                       // pages that left it
    size_t summaries_used = 0;
    size_t retrieved = 0;
    double retrieval_confidence = 0.0;          // best normalized relevance of an old page
    double coverage = 1.0;                      // share of the query's named entities found in the working set
    bool references_earlier = false;            // the request points back at earlier context
    std::vector<std::string> unavailable;       // required context that could not be recovered (never fabricated)
    std::vector<std::string> trace;
};

struct VirtualContextStats {
    uint64_t pages = 0, items = 0, logical_tokens = 0, content_bytes = 0, resident_bytes = 0, spilled_pages = 0;
    uint64_t tier_pages[4] = {0, 0, 0, 0};
    uint64_t selections = 0, page_ins = 0, page_outs = 0, backend_failures = 0;
    uint64_t search_queries = 0, rejected_items = 0;
};

struct SearchHit {
    uint64_t page_id = 0;
    double score = 0.0;          // normalized 0..1
    Provenance provenance = Provenance::kInternalContext;
    std::string source;
    std::string text;            // original content
    std::string content_hash;
};

struct NewItem {
    ItemKind kind = ItemKind::kUser;
    std::string role = "user";
    std::string source = "conversation";
    std::string content;
    Provenance provenance = Provenance::kInternalContext;
    bool pinned = false;
    int priority = 0;
    std::vector<uint64_t> depends_on_items;
    context::SharingScope sharing = context::SharingScope::PRIVATE;
    bool bypass_capacity = false;   // items of the request being served are never rejected
};

class VirtualContext {
public:
    VirtualContext(std::string context_id, context::SecurityScope owner, VirtualContextConfig cfg, std::shared_ptr<IPageBackend> backend = nullptr);
    ~VirtualContext();

    const std::string& id() const { return id_; }
    const context::SecurityScope& owner() const { return owner_; }

    // Adds an item at the end of the logical context. Returns the item id. Content is chunked into pages, indexed and
    // summarized; the original is kept untouched.
    uint64_t append(const NewItem& item);

    // Agents resend their whole history on every request. Matches `messages` against what is already stored and appends
    // only what is new. Returns one item id per message, in order (0 = could not be stored: capacity). Messages after the
    // agent's last assistant turn are the present request and are stored even past the capacity.
    std::vector<uint64_t> ingest(const std::vector<Message>& messages, std::set<uint64_t>* newly_added = nullptr);

    void pin(uint64_t item_id, bool pinned = true);
    void note_used_as_evidence(uint64_t page_id);

    // The minimum useful working set for a request. Updates tiers and access statistics.
    WorkingSet select(const SelectionRequest& req);

    // Authorized search over this context (original content, never summaries). Provenance can be filtered.
    std::vector<SearchHit> search(const context::SecurityScope& requester, const std::string& query, size_t top_k,
                                  int provenance_filter = -1) const;

    // Spills cold page content beyond the resident budget to the backend. Reversible; returns pages spilled.
    size_t compact();

    // Persists every page (content to the backend, metadata to a manifest) and restores it later.
    bool save(const std::string& manifest_path);
    // Saves only when something changed since the last save.
    bool save_if_dirty(const std::string& manifest_path);
    static std::shared_ptr<VirtualContext> load(const std::string& manifest_path, VirtualContextConfig cfg, std::shared_ptr<IPageBackend> backend);

    ContextWindowInfo window_info() const;
    VirtualContextStats stats() const;
    // Original content of a page (reads the backend if it was spilled). False when it cannot be recovered.
    bool page_content(uint64_t page_id, std::string& out) const;
    bool page_info(uint64_t page_id, ContextPage& out) const;
    std::vector<uint64_t> item_pages(uint64_t item_id) const;
    // KV: how much of the previous physical prompt can be kept for this new prompt (exact-prefix, identity checked).
    KvReusePlan plan_kv(const KvIdentity& id, const std::vector<int32_t>& prompt_ids);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string id_;
    context::SecurityScope owner_;
};

// ---- many contexts, one per logical session ----
class VirtualContextStore {
public:
    // storage_dir non-empty: content lives in <dir>/blobs (content addressed), each context's manifest in <dir>; an
    // existing manifest is loaded when its context is opened, so a session survives a restart.
    explicit VirtualContextStore(VirtualContextConfig cfg, std::shared_ptr<IPageBackend> backend = nullptr, std::string storage_dir = "");
    // Writes the context's manifest if it changed (no-op without a storage dir).
    bool persist(const std::shared_ptr<VirtualContext>& vc);
    // context_id is derived from every scope field: two sessions never share a context by accident.
    static std::string context_id(const context::SecurityScope& s);
    std::shared_ptr<VirtualContext> open(const context::SecurityScope& scope);
    size_t size() const;
    const VirtualContextConfig& config() const { return cfg_; }

private:
    VirtualContextConfig cfg_;
    std::shared_ptr<IPageBackend> backend_;
    std::string dir_;
    mutable std::mutex mu_;
    std::map<std::string, std::shared_ptr<VirtualContext>> contexts_;
};

} // namespace strata::rt
