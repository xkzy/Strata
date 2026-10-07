// include/strata/rt/retrieval.hpp - hybrid retrieval with multi-level caches (internal to the runtime; never a tool)
//
// One HybridIndex per virtual context. Documents are the context's pages. A query is classified, routed to the cheapest
// retrievers that can answer it (exact identifier lookup, lexical BM25, vector similarity, structural code relations),
// the scored candidates are normalized and fused, and the result is cached at several levels. Nothing here is visible to
// the agent: the virtual context calls it from select() / search(), and verification reads through the same path.
//
// What this is and is not:
//   * the default embedder (HashedNgramEmbedder) is deterministic character-n-gram hashing: it finds near-spellings and
//     shared vocabulary, NOT paraphrases. A real embedding model plugs in through IEmbedder (id + version are part of
//     every cache key); none ships with the engine today.
//   * authorization is applied by the caller's access function BEFORE a document is scored, and again when a cached
//     result is returned. A shared content hash never grants access.
//   * every cache entry carries the index version; an append invalidates ranked results and candidates (BM25 statistics change).
//     A negative entry ("nothing found") survives appends that add none of the terms it depends on.
//   * L6 caches formatted text only: tokenization lives in the Go server, so token ids cannot be cached here.
#pragma once

#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::rt {

// ---- text helpers shared by the index and the virtual context ----
namespace text {
std::string lower(std::string s);
const std::set<std::string>& stopwords();
std::vector<std::string> raw_tokens(const std::string& text);          // letters/digits plus identifier and path characters
bool looks_like_entity(const std::string& token);                      // snake_case, camelCase, paths, v2, sha256 ...
std::vector<std::string> split_identifier(const std::string& token);   // process_packet / processPacket -> process, packet
std::vector<std::string> index_terms(const std::string& text);         // lowercase tokens + identifier pieces
std::vector<std::string> entities_of(const std::string& text, size_t cap = 64);
std::string normalize_symbol(const std::string& token);                // lowercase, '_' and '-' removed: process_packet == processPacket
} // namespace text

// ---- query classes and routing ----
enum class QueryClass { kExact, kLexical, kSemantic, kCode, kStructural, kConversational, kNumerical, kTemporal, kMultiHop, kUnknown };
const char* to_string(QueryClass c);

enum Retriever : unsigned { kRetrieverExact = 1, kRetrieverLexical = 2, kRetrieverVector = 4, kRetrieverStructural = 8 };

struct QueryPlan {
    QueryClass cls = QueryClass::kUnknown;
    unsigned retrievers = 0;                  // Retriever bits to run first
    std::vector<std::string> identifiers;     // exact-lookup keys found in the query (normalized)
    std::vector<std::string> expanded;        // deterministic variants: snake_case / camelCase pieces, path components
    std::string structural_op;                // "defs" | "callers" | "callees" | "imports" | ""
    std::string structural_target;            // normalized symbol
};
QueryPlan plan_query(const std::string& query);

// ---- embeddings ----
class IEmbedder {
public:
    virtual ~IEmbedder() = default;
    virtual std::string id() const = 0;
    virtual std::string version() const = 0;
    virtual size_t dim() const = 0;
    virtual std::vector<float> embed(const std::string& text) const = 0;   // unit length
};
class HashedNgramEmbedder : public IEmbedder {
public:
    explicit HashedNgramEmbedder(size_t dim = 192) : dim_(dim) {}
    std::string id() const override { return "hashed-ngram"; }
    std::string version() const override { return "1"; }
    size_t dim() const override { return dim_; }
    std::vector<float> embed(const std::string& text) const override;
private:
    size_t dim_;
};

// L4: embeddings by content, embedder and version. Shared by every context of a store; holds vectors only (never results).
class EmbeddingCache {
public:
    explicit EmbeddingCache(size_t max_entries = 1 << 16) : max_(max_entries) {}
    std::shared_ptr<const std::vector<float>> get_or_compute(const std::string& content_hash, const IEmbedder& e, const std::string& content,
                                                             const std::string& chunking_version, bool* hit = nullptr);
    size_t size() const;
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
private:
    mutable std::mutex mu_;
    size_t max_;
    std::unordered_map<std::string, std::shared_ptr<const std::vector<float>>> map_;
    std::list<std::string> order_;   // FIFO eviction (embeddings are cheap to rebuild and not tied to any scope)
    uint64_t hits_ = 0, misses_ = 0;
};

// ---- configuration ----
struct RetrievalConfig {
    // fusion: fused = lexical*w_lexical + exact*w_exact + vector*w_vector + structural*w_structural (+ locality / authority bonuses), clamped to 1
    double w_lexical = 1.0, w_exact = 0.6, w_vector = 0.35, w_structural = 0.3, w_locality = 0.10, w_authority = 0.05;
    double vector_floor = 0.15;               // cosine below this carries no signal (measured: unrelated text 0.02-0.07, near-spellings of a long page 0.22-0.5)
    size_t candidate_limit = 400;             // coarse stage: at most this many candidates reach fusion
    size_t result_limit = 64;
    size_t vector_scan_limit = 20000;         // documents the vector retriever scans (newest first)
    bool lexical_only = false;                // baseline for benchmarks: every query is plain BM25
    bool adaptive = true;                     // retriever weights move with measured usefulness (bounded, deterministic)
    size_t query_cache_entries = 512;         // L2/L3/L5
    size_t negative_cache_entries = 256;
    size_t materialized_cache_bytes = 8u << 20;   // L6
    // retrieval loop guard
    int loop_repeat_limit = 3;
    size_t l2_admit_min_docs = 32;            // admission: on a tiny index a lexical/exact query is cheaper to redo than to cache
};

struct RetrievalMetrics {
    uint64_t queries = 0;
    uint64_t l0_hits = 0, l1_hits = 0, l2_hits = 0, l3_hits = 0, l5_hits = 0, l6_hits = 0, negative_hits = 0;
    uint64_t exact_runs = 0, lexical_runs = 0, vector_runs = 0, structural_runs = 0;
    uint64_t exact_hits = 0, lexical_hits = 0, vector_hits = 0, structural_hits = 0;   // runs that produced candidates
    uint64_t exact_short_circuits = 0;        // exact lookup answered alone
    uint64_t widened = 0;                     // a cheap plan found nothing and the next retrievers ran
    uint64_t embedding_hits = 0, embedding_misses = 0;
    uint64_t duplicate_candidates = 0;        // the same document returned by more than one retriever
    uint64_t loop_throttled = 0;
    double retrieval_ms = 0, rerank_ms = 0, embedding_ms = 0;
    uint64_t class_counts[10] = {0};
    std::string to_json() const;
};

// ---- the index ----
struct Candidate {
    uint32_t doc = 0;
    double score = 0;                         // fused, 0..1
    double lexical = 0, exact = 0, vector = 0, structural = 0;   // normalized per-retriever signals
    unsigned methods = 0;                     // Retriever bits that returned it
};

struct HybridQuery {
    std::string text;
    std::vector<std::string> extra;                       // more query strings (recovery); scored as max
    std::string scope_key;                                // requester scope: part of every cache key
    std::function<bool(uint32_t)> allowed;                // authorization; called before a document is scored
    std::string locality_prefix;                          // sources under this prefix get a small bonus (never create relevance)
    std::vector<std::string> versions;                    // extra cache-key parts (ranking/policy versions)
    std::shared_ptr<void> request_scope;                  // reserved
};

// L0: one request's own cache (same query text asked twice in a request costs one search)
struct RequestCache {
    std::unordered_map<std::string, std::vector<Candidate>> results;
};

struct DocInfo {
    std::string source;       // file path / tool / "conversation"
    double authority = 0.0;   // 0..1: pinned / system / high priority content
};

class HybridIndex {
public:
    HybridIndex(RetrievalConfig cfg, std::shared_ptr<IEmbedder> embedder, std::shared_ptr<EmbeddingCache> embeddings);

    // Adds document `idx` (dense, increasing; gaps are allowed for documents that could not be loaded).
    void add(uint32_t idx, const std::string& content, const std::string& content_hash, const DocInfo& info);
    void reserve_gap(uint32_t idx);                       // a document that exists but has no searchable content
    size_t size() const { return docs_.size(); }
    uint64_t version() const { return version_; }
    bool document_present(uint32_t idx) const { return idx < docs_.size() && docs_[idx].present; }
    bool has_term(const std::string& term) const { return postings_.count(term) > 0; }

    // Runs a query. `rc` (optional) is the request-local cache.
    std::vector<Candidate> search(const HybridQuery& q, RequestCache* rc = nullptr);
    // Feedback: these candidates were used (adapts retriever weights when cfg.adaptive).
    void note_used(const std::vector<Candidate>& used);

    // L6: formatted, model-ready text blocks by (content hash, format key). Text only; token ids live in the Go server.
    bool materialized_get(const std::string& key, std::string& out);
    void materialized_put(const std::string& key, const std::string& block);

    // Retrieval-loop guard: the same (query, candidate set) without new evidence. Returns the number of repeats.
    int observe_state(const std::string& query_hash, const std::string& candidate_hash, bool new_evidence);
    bool loop_throttled(const std::string& query_hash, const std::string& candidate_hash) const;

    RetrievalMetrics metrics() const;
    const RetrievalConfig& config() const { return cfg_; }

private:
    struct Doc { uint32_t len = 0; std::string hash, source; double authority = 0; bool present = false; };
    struct CacheEntry { uint64_t version = 0; std::vector<Candidate> cands; uint32_t hits = 0; double cost_ms = 0; std::vector<std::string> terms; };   // terms: what a negative entry depends on

    std::vector<Candidate> run(const HybridQuery& q, const std::string& text);
    void exact_lookup(const QueryPlan& plan, const HybridQuery& q, std::unordered_map<uint32_t, Candidate>& out);
    void lexical(const std::string& text, const HybridQuery& q, std::unordered_map<uint32_t, Candidate>& out);
    void vector(const std::string& text, const HybridQuery& q, std::unordered_map<uint32_t, Candidate>& out);
    void structural(const QueryPlan& plan, const HybridQuery& q, std::unordered_map<uint32_t, Candidate>& out);
    void ensure_vectors();
    double weight(unsigned retriever_bit) const;
    std::string key_for(const HybridQuery& q, const std::string& text) const;
    void cache_put(std::unordered_map<std::string, CacheEntry>& m, std::list<std::string>& order, size_t cap, const std::string& key, std::vector<Candidate> v, double cost_ms,
                   std::vector<std::string> terms = {});
    std::string ranking_version() const;
    bool terms_absent(const std::vector<std::string>& terms) const;

    RetrievalConfig cfg_;
    std::shared_ptr<IEmbedder> embedder_;
    std::shared_ptr<EmbeddingCache> embeddings_;
    uint64_t version_ = 0;
    std::vector<Doc> docs_;
    std::unordered_map<std::string, std::vector<std::pair<uint32_t, uint16_t>>> postings_;    // lexical
    double total_len_ = 0;
    std::unordered_map<std::string, std::vector<uint32_t>> exact_;                            // L1: normalized identifier -> docs
    std::unordered_map<std::string, std::vector<uint32_t>> defs_, calls_, imports_;           // structural
    std::vector<std::shared_ptr<const std::vector<float>>> vectors_;                          // lazy
    std::vector<std::string> contents_for_vectors_;                                           // kept only until embedded
    size_t embedded_upto_ = 0;

    mutable std::mutex mu_;
    std::unordered_map<std::string, CacheEntry> l2_, l3_, neg_;                               // ranked results; raw candidates; negative results
    std::list<std::string> l2_order_, l3_order_, neg_order_;
    std::unordered_map<std::string, std::string> l6_;
    std::list<std::string> l6_order_;
    size_t l6_bytes_ = 0;
    std::unordered_map<std::string, int> loop_;
    double adapt_used_[4] = {0, 0, 0, 0}, adapt_runs_[4] = {0, 0, 0, 0};
    RetrievalMetrics m_;
};

} // namespace strata::rt
