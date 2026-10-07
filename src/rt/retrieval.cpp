// src/rt/retrieval.cpp - hybrid retrieval with multi-level caches (see retrieval.hpp)
#include "strata/rt/retrieval.hpp"

#include "strata/rt/types.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace strata::rt {

// =====================================================================================================================
// text helpers (moved here from virtual_context.cpp)
// =====================================================================================================================
namespace text {

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

const std::set<std::string>& stopwords() {
    static const std::set<std::string> w = {"the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had", "her", "was", "one", "our",
        "out", "has", "have", "this", "that", "with", "from", "they", "will", "what", "when", "which", "their", "there", "would", "about", "into",
        "than", "then", "them", "these", "those", "been", "were", "your", "also", "its", "how", "who", "why", "did", "does", "use", "using", "used",
        "should", "could", "please", "make", "need", "want", "like", "just", "some", "more", "very", "let", "get", "set", "new", "now", "yes"};
    return w;
}

static bool is_word_char(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == '/' || c == '-'; }

// Raw tokens: letters/digits plus the characters that make up identifiers and paths.
std::vector<std::string> raw_tokens(const std::string& text) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && !is_word_char(text[i])) ++i;
        size_t j = i;
        while (j < text.size() && is_word_char(text[j])) ++j;
        if (j > i) {
            size_t b = i, e = j;
            while (e > b && (text[e - 1] == '.' || text[e - 1] == '-' || text[e - 1] == '/')) --e;   // trailing punctuation
            while (b < e && (text[b] == '.' || text[b] == '-')) ++b;
            if (e > b) out.push_back(text.substr(b, e - b));
        }
        i = j;
    }
    return out;
}

bool looks_like_entity(const std::string& t) {
    if (t.size() < 3) return false;
    bool under = false, dot = false, slash = false, digit = false, alpha = false;
    int caps = 0;
    for (size_t i = 0; i < t.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(t[i]);
        if (c == '_') under = true;
        else if (c == '.' && i + 1 < t.size() && std::isalpha(static_cast<unsigned char>(t[i + 1])) && i > 0) dot = true;
        else if (c == '/') slash = true;
        else if (std::isdigit(c)) digit = true;
        else if (std::isalpha(c)) { alpha = true; if (std::isupper(c) && i > 0) ++caps; }
    }
    if (under || slash) return alpha;
    if (dot) return alpha;
    if (caps >= 1 && alpha) return true;               // camelCase / CamelCase
    if (digit && alpha) return true;                   // v2, sha256, port8080
    return false;
}

std::vector<std::string> split_identifier(const std::string& t) {
    std::vector<std::string> parts;
    std::string cur;
    auto flush = [&]() { if (cur.size() >= 2) parts.push_back(lower(cur)); cur.clear(); };
    for (size_t i = 0; i < t.size(); ++i) {
        char c = t[i];
        if (c == '_' || c == '.' || c == '/' || c == '-') { flush(); continue; }
        if (!cur.empty() && std::isupper(static_cast<unsigned char>(c)) && std::islower(static_cast<unsigned char>(cur.back()))) flush();
        cur += c;
    }
    flush();
    return parts;
}

// index terms: whole lowercase tokens plus the pieces of identifiers/paths
std::vector<std::string> index_terms(const std::string& text) {
    std::vector<std::string> out;
    for (const auto& t : raw_tokens(text)) {
        const std::string l = lower(t);
        const bool entity = looks_like_entity(t);
        if (l.size() >= 2 && !stopwords().count(l)) out.push_back(l);
        if (entity || l.find_first_of("_./-") != std::string::npos)
            for (auto& p : split_identifier(t)) if (p != l && !stopwords().count(p)) out.push_back(p);
    }
    return out;
}

std::vector<std::string> entities_of(const std::string& text, size_t cap) {
    std::vector<std::string> out;
    std::set<std::string> seen;
    for (const auto& t : raw_tokens(text)) {
        if (!looks_like_entity(t)) continue;
        std::string l = lower(t);
        if (seen.insert(l).second) { out.push_back(l); if (out.size() >= cap) break; }
    }
    return out;
}

std::string normalize_symbol(const std::string& token) {
    std::string out;
    for (char c : token) if (c != '_' && c != '-') out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return out;
}

} // namespace text

using namespace text;

namespace {

const char* kClassNames[] = {"EXACT", "LEXICAL", "SEMANTIC", "CODE", "STRUCTURAL", "CONVERSATIONAL", "NUMERICAL", "TEMPORAL", "MULTI_HOP", "UNKNOWN"};

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

bool is_ident_char(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

bool is_hex_hash(const std::string& t) {
    if (t.size() < 7 || t.size() > 64) return false;
    bool digit = false, letter = false;
    for (char c : t) {
        if (std::isdigit(static_cast<unsigned char>(c))) digit = true;
        else if ((c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) letter = true;
        else return false;
    }
    return digit && letter;
}

bool is_error_code(const std::string& t) {   // E1234, ERR_TIMEOUT, ECONNRESET, 0x80070005
    if (t.size() < 3) return false;
    if (t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X')) return true;
    bool upper = false, lower_c = false;
    for (char c : t) {
        if (std::islower(static_cast<unsigned char>(c))) lower_c = true;
        if (std::isupper(static_cast<unsigned char>(c))) upper = true;
    }
    return upper && !lower_c && t.size() >= 4 && (t.find('_') != std::string::npos || std::any_of(t.begin(), t.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); }));
}

// camelCase <-> snake_case variants of an identifier (full tokens, lowercased by the lexical index)
std::vector<std::string> variants_of(const std::string& id) {
    std::vector<std::string> out;
    const auto parts = split_identifier(id);
    if (parts.size() < 2) return out;
    std::string snake, camel;
    for (size_t i = 0; i < parts.size(); ++i) {
        snake += (i ? "_" : "") + parts[i];
        std::string p = parts[i];
        if (i) p[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(p[0])));
        camel += p;
    }
    if (snake != lower(id)) out.push_back(snake);
    if (camel != id) out.push_back(lower(camel));
    return out;
}

bool has_any(const std::string& hay, std::initializer_list<const char*> needles) {
    for (const char* n : needles) if (hay.find(n) != std::string::npos) return true;
    return false;
}

bool is_keyword(const std::string& w) {
    static const std::set<std::string> k = {"if", "for", "while", "switch", "return", "sizeof", "catch", "else", "do", "case", "new", "delete", "throw", "typeof", "defined"};
    return k.count(w) > 0;
}

} // namespace

const char* to_string(QueryClass c) { return kClassNames[static_cast<int>(c)]; }

// =====================================================================================================================
// query planning (deterministic: no model, no unbounded scans)
// =====================================================================================================================
QueryPlan plan_query(const std::string& query_in) {
    QueryPlan plan;
    const std::string query = query_in.size() > 4096 ? query_in.substr(0, 4096) : query_in;
    const std::string lq = lower(query);
    std::set<std::string> ids;
    size_t plain_words = 0, digits = 0, letters = 0;
    for (const auto& t : raw_tokens(query)) {
        const std::string l = lower(t);
        const bool url = t.find("://") != std::string::npos;
        if (looks_like_entity(t) || is_hex_hash(t) || is_error_code(t) || url) { if (ids.insert(normalize_symbol(t)).second) plan.identifiers.push_back(normalize_symbol(t)); }
        else if (l.size() >= 2 && !stopwords().count(l)) ++plain_words;
        for (char c : t) { if (std::isdigit(static_cast<unsigned char>(c))) ++digits; else if (std::isalpha(static_cast<unsigned char>(c))) ++letters; }
    }
    // call syntax: process_packet() / handle(x) names a function even when it is a plain word
    for (size_t i = 0; i + 1 < query.size(); ++i) {
        if (query[i] != '(' || i == 0 || !is_ident_char(query[i - 1])) continue;
        size_t b = i;
        while (b > 0 && is_ident_char(query[b - 1])) --b;
        const std::string name = query.substr(b, i - b);
        if (name.size() >= 3 && !is_keyword(lower(name)) && ids.insert(normalize_symbol(name)).second) plan.identifiers.push_back(normalize_symbol(name));
    }
    for (const auto& t : raw_tokens(query)) {   // deterministic variants and pieces
        if (!looks_like_entity(t)) continue;
        for (auto& v : variants_of(t)) plan.expanded.push_back(v);
        for (auto& p : split_identifier(t)) plan.expanded.push_back(p);
    }
    std::sort(plan.expanded.begin(), plan.expanded.end());
    plan.expanded.erase(std::unique(plan.expanded.begin(), plan.expanded.end()), plan.expanded.end());

    // structural cues
    auto target = [&]() -> std::string { return plan.identifiers.empty() ? std::string() : plan.identifiers.front(); };
    if (has_any(lq, {"who calls", "callers of", "where is", "where does"}) && has_any(lq, {" called", "who calls", "callers of", " call "}) && !plan.identifiers.empty()) { plan.structural_op = "callers"; }
    else if (has_any(lq, {"defined", "definition of", "declaration of", "where is the implementation", "implemented"}) && !plan.identifiers.empty()) { plan.structural_op = "defs"; }
    else if (has_any(lq, {"what does", "callees of", "calls what"}) && has_any(lq, {" call", "callees"}) && !plan.identifiers.empty()) { plan.structural_op = "callees"; }
    else if (has_any(lq, {"imports", "includes", "depends on", "who uses module", "imported by"}) && !plan.identifiers.empty()) { plan.structural_op = "imports"; }
    plan.structural_target = target();

    const bool convo = has_any(lq, {"we discussed", "did we", "you said", "i said", "earlier", "you mentioned", "we decided", "we agreed", "last time", "remember when", "we talked"});
    const bool temporal = has_any(lq, {"yesterday", "last week", "last month", "most recent", "latest", "recently", "today", "this morning"});
    const bool multihop = (plan.identifiers.size() >= 2 && has_any(lq, {" between ", " through ", " and then ", " which calls ", " that calls ", "path from"})) || (plain_words > 24 && plan.identifiers.size() >= 2);
    const bool question = lq.rfind("how ", 0) == 0 || lq.rfind("why ", 0) == 0 || lq.rfind("what ", 0) == 0 || lq.rfind("explain", 0) == 0 || lq.rfind("describe", 0) == 0 || lq.find('?') != std::string::npos;

    if (!plan.structural_op.empty()) { plan.cls = QueryClass::kStructural; plan.retrievers = kRetrieverStructural | kRetrieverExact; }
    else if (multihop) { plan.cls = QueryClass::kMultiHop; plan.retrievers = kRetrieverExact | kRetrieverLexical | kRetrieverVector | kRetrieverStructural; }
    else if (convo) { plan.cls = QueryClass::kConversational; plan.retrievers = kRetrieverLexical | kRetrieverVector; }
    else if (!plan.identifiers.empty() && plain_words <= 1) { plan.cls = QueryClass::kExact; plan.retrievers = kRetrieverExact; }
    else if (!plan.identifiers.empty()) { plan.cls = QueryClass::kCode; plan.retrievers = kRetrieverExact | kRetrieverLexical | kRetrieverStructural; }
    else if (digits > letters && digits > 0) { plan.cls = QueryClass::kNumerical; plan.retrievers = kRetrieverExact | kRetrieverLexical; }
    else if (temporal) { plan.cls = QueryClass::kTemporal; plan.retrievers = kRetrieverLexical; }
    else if (question && plain_words >= 4) { plan.cls = QueryClass::kSemantic; plan.retrievers = kRetrieverLexical | kRetrieverVector; }
    else if (plain_words >= 1) { plan.cls = QueryClass::kLexical; plan.retrievers = kRetrieverLexical; }
    else { plan.cls = QueryClass::kUnknown; plan.retrievers = kRetrieverExact | kRetrieverLexical | kRetrieverVector; }
    return plan;
}

// =====================================================================================================================
// embeddings
// =====================================================================================================================
std::vector<float> HashedNgramEmbedder::embed(const std::string& text_in) const {
    std::vector<float> v(dim_, 0.0f);
    const std::string t = lower(text_in.size() > 1 << 16 ? text_in.substr(0, 1 << 16) : text_in);
    auto add = [&](const std::string& g, float w) {
        uint64_t h = 1469598103934665603ULL;
        for (unsigned char c : g) { h ^= c; h *= 1099511628211ULL; }
        const size_t slot = static_cast<size_t>(h % dim_);
        v[slot] += ((h >> 40) & 1) ? w : -w;   // signed hashing keeps unrelated texts near orthogonal
    };
    std::string word;
    auto flush = [&]() {
        if (word.size() >= 2 && !stopwords().count(word)) {
            add("w:" + word, 1.5f);
            const std::string padded = "^" + word + "$";
            for (size_t i = 0; i + 3 <= padded.size(); ++i) add(padded.substr(i, 3), 0.5f);
        }
        word.clear();
    };
    for (char c : t) { if (std::isalnum(static_cast<unsigned char>(c))) word.push_back(c); else flush(); }
    flush();
    double n = 0;
    for (float x : v) n += static_cast<double>(x) * x;
    if (n > 0) { const float inv = static_cast<float>(1.0 / std::sqrt(n)); for (auto& x : v) x *= inv; }
    return v;
}

std::shared_ptr<const std::vector<float>> EmbeddingCache::get_or_compute(const std::string& content_hash, const IEmbedder& e, const std::string& content,
                                                                          const std::string& chunking_version, bool* hit) {
    const std::string key = content_hash + "|" + e.id() + "|" + e.version() + "|" + std::to_string(e.dim()) + "|" + chunking_version;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(key);
        if (it != map_.end()) { ++hits_; if (hit) *hit = true; return it->second; }
    }
    auto v = std::make_shared<const std::vector<float>>(e.embed(content));
    std::lock_guard<std::mutex> lock(mu_);
    ++misses_;
    if (hit) *hit = false;
    if (!map_.count(key)) {
        map_[key] = v;
        order_.push_back(key);
        while (map_.size() > max_ && !order_.empty()) { map_.erase(order_.front()); order_.pop_front(); }
    }
    return v;
}
size_t EmbeddingCache::size() const { std::lock_guard<std::mutex> lock(mu_); return map_.size(); }

// =====================================================================================================================
// metrics
// =====================================================================================================================
std::string RetrievalMetrics::to_json() const {
    std::ostringstream o;
    auto rate = [](uint64_t a, uint64_t b) { return b ? static_cast<double>(a) / static_cast<double>(b) : 0.0; };
    const uint64_t cache_hits = l0_hits + l2_hits + l3_hits + negative_hits;
    o << "{\"queries\":" << queries << ",\"cache_hit_rate\":" << rate(cache_hits, queries)
      << ",\"l0_hits\":" << l0_hits << ",\"l2_hits\":" << l2_hits << ",\"l3_hits\":" << l3_hits << ",\"rerank_hits\":" << l5_hits
      << ",\"l6_hits\":" << l6_hits << ",\"negative_hits\":" << negative_hits
      << ",\"exact_runs\":" << exact_runs << ",\"lexical_runs\":" << lexical_runs << ",\"vector_runs\":" << vector_runs << ",\"structural_runs\":" << structural_runs
      << ",\"exact_hit_rate\":" << rate(exact_hits, exact_runs) << ",\"linear_hit_rate\":" << rate(lexical_hits, lexical_runs)
      << ",\"vector_hit_rate\":" << rate(vector_hits, vector_runs) << ",\"structural_hit_rate\":" << rate(structural_hits, structural_runs)
      << ",\"exact_short_circuits\":" << exact_short_circuits << ",\"widened\":" << widened
      << ",\"embedding_hits\":" << embedding_hits << ",\"embedding_misses\":" << embedding_misses
      << ",\"duplicate_retrieval_rate\":" << rate(duplicate_candidates, queries) << ",\"retrieval_loops_throttled\":" << loop_throttled
      << ",\"retrieval_ms\":" << retrieval_ms << ",\"rerank_ms\":" << rerank_ms << ",\"embedding_ms\":" << embedding_ms << ",\"classes\":{";
    for (int i = 0; i < 10; ++i) o << (i ? "," : "") << "\"" << kClassNames[i] << "\":" << class_counts[i];
    o << "}}";
    return o.str();
}

// =====================================================================================================================
// the index
// =====================================================================================================================
HybridIndex::HybridIndex(RetrievalConfig cfg, std::shared_ptr<IEmbedder> embedder, std::shared_ptr<EmbeddingCache> embeddings)
    : cfg_(cfg), embedder_(embedder ? std::move(embedder) : std::make_shared<HashedNgramEmbedder>()),
      embeddings_(embeddings ? std::move(embeddings) : std::make_shared<EmbeddingCache>()) {}

void HybridIndex::reserve_gap(uint32_t idx) {
    std::lock_guard<std::mutex> lock(mu_);
    if (docs_.size() <= idx) { docs_.resize(idx + 1); vectors_.resize(idx + 1); contents_for_vectors_.resize(idx + 1); }
    ++version_;
}

void HybridIndex::add(uint32_t idx, const std::string& content, const std::string& content_hash, const DocInfo& info) {
    std::lock_guard<std::mutex> lock(mu_);
    if (docs_.size() <= idx) { docs_.resize(idx + 1); vectors_.resize(idx + 1); contents_for_vectors_.resize(idx + 1); }
    ++version_;
    Doc& d = docs_[idx];
    d.present = true; d.hash = content_hash; d.source = info.source; d.authority = info.authority;

    // lexical
    std::unordered_map<std::string, uint16_t> tf;
    uint32_t len = 0;
    for (auto& t : index_terms(content)) { auto& c = tf[t]; if (c < 65535) ++c; ++len; }
    for (auto& kv : tf) postings_[kv.first].push_back({idx, kv.second});
    d.len = len;
    total_len_ += len;

    // L1 exact: identifiers, paths and the document's own source
    auto add_exact = [&](const std::string& key) {
        if (key.size() < 3) return;
        auto& v = exact_[key];
        if (v.empty() || v.back() != idx) v.push_back(idx);
    };
    size_t n = 0;
    for (const auto& t : raw_tokens(content)) {
        if (++n > 20000) break;
        if (looks_like_entity(t) || is_hex_hash(t) || is_error_code(t)) {
            add_exact(normalize_symbol(t));
            const size_t slash = t.rfind('/');
            if (slash != std::string::npos) add_exact(normalize_symbol(t.substr(slash + 1)));
        }
    }
    if (!info.source.empty() && info.source != "conversation") {
        add_exact(normalize_symbol(info.source));
        const size_t slash = info.source.rfind('/');
        if (slash != std::string::npos) add_exact(normalize_symbol(info.source.substr(slash + 1)));
    }

    // structural: definitions, call sites, imports (bounded scan; code-like text only yields anything)
    static const std::set<std::string> def_words = {"def", "class", "struct", "enum", "interface", "fn", "func", "function", "namespace", "trait", "impl", "type", "union"};
    size_t pos = 0, lines = 0, syms = 0;
    const size_t limit = std::min<size_t>(content.size(), 1u << 16);
    auto add_to = [&](std::unordered_map<std::string, std::vector<uint32_t>>& m, const std::string& name) {
        if (name.size() < 2 || ++syms > 400) return;
        auto& v = m[normalize_symbol(name)];
        if (v.empty() || v.back() != idx) v.push_back(idx);
    };
    while (pos < limit && lines < 4000) {
        size_t e = content.find('\n', pos);
        if (e == std::string::npos || e > limit) e = limit;
        const std::string line = content.substr(pos, e - pos);
        pos = e + 1; ++lines;
        size_t b = line.find_first_not_of(" \t");
        if (b == std::string::npos) continue;
        const std::string t = line.substr(b);
        if (t.rfind("//", 0) == 0 || t.rfind("#!", 0) == 0) continue;
        std::string def_name;
        // imports
        if (t.rfind("#include", 0) == 0) {
            size_t q = t.find_first_of("<\"");
            if (q != std::string::npos) { size_t r = t.find_first_of(">\"", q + 1); if (r != std::string::npos) { std::string inc = t.substr(q + 1, r - q - 1); add_to(imports_, inc); const size_t sl = inc.rfind('/'); if (sl != std::string::npos) add_to(imports_, inc.substr(sl + 1)); } }
            continue;
        }
        if (t.rfind("import ", 0) == 0 || t.rfind("from ", 0) == 0 || t.rfind("use ", 0) == 0) {
            std::istringstream ls(t);
            std::string w1, w2;
            ls >> w1 >> w2;
            while (!w2.empty() && (w2.back() == ';' || w2.back() == ',')) w2.pop_back();
            if (!w2.empty() && w2 != "(") add_to(imports_, w2);
            continue;
        }
        // keyword definitions
        {
            std::istringstream ls(t);
            std::string w;
            while (ls >> w && def_words.count(w)) {}   // pub, async ... are not in the set: handled below
            std::istringstream ls2(t);
            std::string a, b2, c;
            ls2 >> a >> b2 >> c;
            auto ident = [](std::string s) { size_t i = 0; while (i < s.size() && is_ident_char(s[i])) ++i; return s.substr(0, i); };
            if (def_words.count(a) && !b2.empty()) def_name = ident(b2);
            else if ((a == "pub" || a == "async" || a == "export" || a == "static" || a == "public") && def_words.count(b2) && !c.empty()) def_name = ident(c);
            else if ((a == "pub" || a == "async" || a == "export") && (b2 == "async" || b2 == "pub") && !c.empty()) { std::string d4; ls2 >> d4; def_name = ident(d4); }
        }
        // C-like definition at column 0: "int process_packet(struct pkt *p) {"
        const size_t paren = t.find('(');
        if (def_name.empty() && b == 0 && paren != std::string::npos && paren > 2 && t[0] != '#' && t.find(';') == std::string::npos) {
            size_t nb = paren;
            while (nb > 0 && is_ident_char(t[nb - 1])) --nb;
            if (nb < paren && nb > 0 && (t[nb - 1] == ' ' || t[nb - 1] == '*' || t[nb - 1] == '&' || t[nb - 1] == ':') && !is_keyword(t.substr(nb, paren - nb))) def_name = t.substr(nb, paren - nb);
        }
        if (!def_name.empty()) { add_to(defs_, def_name); add_exact(normalize_symbol(def_name)); }
        // call sites
        for (size_t i = 0; i < t.size(); ++i) {
            if (t[i] != '(' || i == 0 || !is_ident_char(t[i - 1])) continue;
            size_t cb = i;
            while (cb > 0 && is_ident_char(t[cb - 1])) --cb;
            const std::string name = t.substr(cb, i - cb);
            if (name.size() >= 2 && !is_keyword(lower(name)) && name != def_name) add_to(calls_, name);
        }
    }
    contents_for_vectors_[idx] = content;   // embedded lazily, only when a vector retriever first runs
}

void HybridIndex::ensure_vectors() {
    if (embedded_upto_ >= docs_.size()) return;
    const double t0 = now_ms();
    for (; embedded_upto_ < docs_.size(); ++embedded_upto_) {
        const Doc& d = docs_[embedded_upto_];
        if (!d.present) continue;
        bool hit = false;
        vectors_[embedded_upto_] = embeddings_->get_or_compute(d.hash, *embedder_, contents_for_vectors_[embedded_upto_], "1", &hit);
        ++(hit ? m_.embedding_hits : m_.embedding_misses);
        std::string().swap(contents_for_vectors_[embedded_upto_]);
    }
    m_.embedding_ms += now_ms() - t0;
}

double HybridIndex::weight(unsigned bit) const {
    double base = bit == kRetrieverExact ? cfg_.w_exact : bit == kRetrieverLexical ? cfg_.w_lexical : bit == kRetrieverVector ? cfg_.w_vector : cfg_.w_structural;
    if (!cfg_.adaptive) return base;
    const int i = bit == kRetrieverExact ? 0 : bit == kRetrieverLexical ? 1 : bit == kRetrieverVector ? 2 : 3;
    if (adapt_runs_[i] < 20) return base;
    const double rate = std::min(1.0, adapt_used_[i] / adapt_runs_[i]);
    return base * (0.75 + 0.5 * rate);   // bounded: at most +-25%
}

bool HybridIndex::terms_absent(const std::vector<std::string>& terms) const {
    for (const auto& t : terms)
        if (postings_.count(t) || exact_.count(t) || defs_.count(t) || calls_.count(t) || imports_.count(t)) return false;
    return true;
}

std::string HybridIndex::ranking_version() const {
    char buf[96];
    std::snprintf(buf, sizeof buf, "%.3f/%.3f/%.3f/%.3f", weight(kRetrieverExact), weight(kRetrieverLexical), weight(kRetrieverVector), weight(kRetrieverStructural));
    return buf;
}

void HybridIndex::note_used(const std::vector<Candidate>& used) {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& c : used) {
        if (c.methods & kRetrieverExact) adapt_used_[0] += 1;
        if (c.methods & kRetrieverLexical) adapt_used_[1] += 1;
        if (c.methods & kRetrieverVector) adapt_used_[2] += 1;
        if (c.methods & kRetrieverStructural) adapt_used_[3] += 1;
    }
}

// ---- retrievers ----
void HybridIndex::exact_lookup(const QueryPlan& plan, const HybridQuery& q, std::unordered_map<uint32_t, Candidate>& out) {
    ++m_.exact_runs;
    std::unordered_map<uint32_t, int> hits;
    for (const auto& id : plan.identifiers) {
        auto it = exact_.find(id);
        if (it == exact_.end()) continue;
        for (uint32_t d : it->second) if (!q.allowed || q.allowed(d)) ++hits[d];
    }
    if (hits.empty()) return;
    ++m_.exact_hits;
    for (auto& kv : hits) {
        Candidate& c = out[kv.first];
        c.doc = kv.first;
        c.exact = static_cast<double>(kv.second) / static_cast<double>(std::max<size_t>(1, plan.identifiers.size()));
        c.methods |= kRetrieverExact;
    }
}

void HybridIndex::lexical(const std::string& text, const HybridQuery& q, std::unordered_map<uint32_t, Candidate>& out) {
    ++m_.lexical_runs;
    if (docs_.empty()) return;
    const QueryPlan plan = plan_query(text);
    std::set<std::string> qterms, expanded;
    for (auto& t : index_terms(text)) qterms.insert(t);
    for (auto& t : plan.expanded) if (!qterms.count(t)) expanded.insert(t);
    std::set<std::string> ents;
    for (auto& e : entities_of(text)) ents.insert(e);
    const double n = static_cast<double>(docs_.size());
    const double avg = std::max(1.0, total_len_ / n);
    const double k1 = 1.2, b = 0.75;
    double max_possible = 0;
    std::unordered_map<uint32_t, double> scores;
    auto run_terms = [&](const std::set<std::string>& terms, double boost) {
        for (const auto& t : terms) {
            auto it = postings_.find(t);
            if (it == postings_.end()) continue;
            const auto& pl = it->second;
            if (docs_.size() > 200 && pl.size() > docs_.size() / 2) continue;   // carries no information
            const double idf = std::log(1.0 + (n - static_cast<double>(pl.size()) + 0.5) / (static_cast<double>(pl.size()) + 0.5));
            const double w = boost * idf * (ents.count(t) ? 1.6 : 1.0);
            max_possible += w * (k1 + 1);
            for (const auto& pr : pl) {
                if (q.allowed && !q.allowed(pr.first)) continue;   // authorization before scoring
                const double tf = pr.second, dl = docs_[pr.first].len;
                scores[pr.first] += w * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg));
            }
        }
    };
    run_terms(qterms, 1.0);
    run_terms(expanded, 0.5);
    if (scores.empty() || max_possible <= 0) return;
    ++m_.lexical_hits;
    std::vector<std::pair<double, uint32_t>> ranked;
    ranked.reserve(scores.size());
    for (auto& kv : scores) ranked.push_back({std::min(1.0, kv.second / max_possible), kv.first});
    std::sort(ranked.begin(), ranked.end(), [](auto& a, auto& b2) { return a.first != b2.first ? a.first > b2.first : a.second < b2.second; });
    if (ranked.size() > cfg_.candidate_limit) ranked.resize(cfg_.candidate_limit);   // coarse stage
    for (auto& r : ranked) { Candidate& c = out[r.second]; c.doc = r.second; c.lexical = std::max(c.lexical, r.first); c.methods |= kRetrieverLexical; }
}

void HybridIndex::vector(const std::string& text, const HybridQuery& q, std::unordered_map<uint32_t, Candidate>& out) {
    ++m_.vector_runs;
    if (docs_.empty()) return;
    ensure_vectors();
    const double t0 = now_ms();
    bool hit = false;
    auto qv = embeddings_->get_or_compute(text::lower(text), *embedder_, text, "query", &hit);
    ++(hit ? m_.embedding_hits : m_.embedding_misses);
    std::vector<std::pair<double, uint32_t>> ranked;
    size_t scanned = 0;
    for (size_t i = docs_.size(); i-- > 0 && scanned < cfg_.vector_scan_limit;) {
        if (!docs_[i].present || !vectors_[i]) continue;
        if (q.allowed && !q.allowed(static_cast<uint32_t>(i))) continue;
        ++scanned;
        const auto& dv = *vectors_[i];
        double dot = 0;
        for (size_t k = 0; k < dv.size() && k < qv->size(); ++k) dot += static_cast<double>(dv[k]) * (*qv)[k];
        const double s = (dot - cfg_.vector_floor) / (1.0 - cfg_.vector_floor);
        if (s > 0) ranked.push_back({std::min(1.0, s), static_cast<uint32_t>(i)});
    }
    m_.embedding_ms += now_ms() - t0;
    if (ranked.empty()) return;
    ++m_.vector_hits;
    std::sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    if (ranked.size() > cfg_.candidate_limit) ranked.resize(cfg_.candidate_limit);
    for (auto& r : ranked) { Candidate& c = out[r.second]; c.doc = r.second; c.vector = r.first; c.methods |= kRetrieverVector; }
}

void HybridIndex::structural(const QueryPlan& plan, const HybridQuery& q, std::unordered_map<uint32_t, Candidate>& out) {
    ++m_.structural_runs;
    const std::string& t = plan.structural_target;
    if (t.empty() && plan.structural_op.empty()) {   // code query without a relation: documents that define or call the named symbols
        std::unordered_map<uint32_t, double> s;
        for (const auto& id : plan.identifiers) {
            auto d = defs_.find(id);
            if (d != defs_.end()) for (uint32_t x : d->second) s[x] = std::max(s[x], 1.0);
        }
        bool any = false;
        for (auto& kv : s) if (!q.allowed || q.allowed(kv.first)) { Candidate& c = out[kv.first]; c.doc = kv.first; c.structural = kv.second; c.methods |= kRetrieverStructural; any = true; }
        if (any) ++m_.structural_hits;
        return;
    }
    const std::unordered_map<std::string, std::vector<uint32_t>>* m = nullptr;
    double s = 1.0;
    if (plan.structural_op == "defs" || plan.structural_op == "callees") m = &defs_;   // a callee list lives in the defining document
    else if (plan.structural_op == "callers") { m = &calls_; s = 0.9; }
    else if (plan.structural_op == "imports") m = &imports_;
    if (!m) return;
    auto it = m->find(t);
    bool any = false;
    if (it != m->end())
        for (uint32_t d : it->second) {
            if (q.allowed && !q.allowed(d)) continue;
            Candidate& c = out[d]; c.doc = d; c.structural = std::max(c.structural, s); c.methods |= kRetrieverStructural; any = true;
        }
    if (any) ++m_.structural_hits;
}

// ---- search with caches ----
std::string HybridIndex::key_for(const HybridQuery& q, const std::string& text) const {
    std::string norm, prev;
    for (char c : text) {   // normalized query: lowercase, whitespace collapsed
        const char l = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (std::isspace(static_cast<unsigned char>(l))) { if (!norm.empty() && norm.back() != ' ') norm.push_back(' '); }
        else norm.push_back(l);
    }
    while (!norm.empty() && norm.back() == ' ') norm.pop_back();
    std::string k = q.scope_key + "\x1f" + norm + "\x1f" + q.locality_prefix;
    for (const auto& v : q.versions) k += "\x1f" + v;
    return k;
}

void HybridIndex::cache_put(std::unordered_map<std::string, CacheEntry>& m, std::list<std::string>& order, size_t cap, const std::string& key, std::vector<Candidate> v, double cost_ms,
                            std::vector<std::string> terms) {
    if (cap == 0) return;
    if (!m.count(key)) order.push_back(key);
    CacheEntry e;
    e.version = version_; e.cands = std::move(v); e.cost_ms = cost_ms; e.terms = std::move(terms);
    m[key] = std::move(e);
    while (m.size() > cap && !order.empty()) {
        // not LRU alone: among the oldest few, evict the entry that is cheapest to recompute and least used per byte
        auto victim = order.begin();
        double best = 1e300;
        size_t seen = 0;
        for (auto it = order.begin(); it != order.end() && seen < 8; ++it, ++seen) {
            auto f = m.find(*it);
            if (f == m.end()) { victim = it; best = -1; break; }
            const double keep = (f->second.hits + 1.0) * (f->second.cost_ms + 0.01) / (static_cast<double>(f->second.cands.size()) + 1.0);
            if (keep < best) { best = keep; victim = it; }
        }
        m.erase(*victim);
        order.erase(victim);
    }
}

std::vector<Candidate> HybridIndex::run(const HybridQuery& q, const std::string& text) {
    const double t0 = now_ms();
    QueryPlan plan = plan_query(text);
    if (cfg_.lexical_only) plan.retrievers = kRetrieverLexical;
    ++m_.class_counts[static_cast<int>(plan.cls)];
    std::unordered_map<uint32_t, Candidate> pool;
    unsigned ran = 0;
    auto run_set = [&](unsigned set) {
        if ((set & kRetrieverExact) && !(ran & kRetrieverExact)) { exact_lookup(plan, q, pool); ran |= kRetrieverExact; }
        if ((set & kRetrieverStructural) && !(ran & kRetrieverStructural)) { structural(plan, q, pool); ran |= kRetrieverStructural; }
        if ((set & kRetrieverLexical) && !(ran & kRetrieverLexical)) { lexical(text, q, pool); ran |= kRetrieverLexical; }
        if ((set & kRetrieverVector) && !(ran & kRetrieverVector)) { vector(text, q, pool); ran |= kRetrieverVector; }
    };
    run_set(plan.retrievers);
    if (plan.cls == QueryClass::kExact && !pool.empty()) ++m_.exact_short_circuits;
    // cost-aware widening: the cheap plan found nothing, so the next retrievers run
    if (pool.empty() && !cfg_.lexical_only) {
        const unsigned next = (plan.retrievers & kRetrieverVector) ? 0u : (kRetrieverLexical | kRetrieverVector);
        const unsigned todo = next & ~ran;
        if (todo) { ++m_.widened; run_set(kRetrieverLexical); if (pool.empty()) run_set(kRetrieverVector); }
    }
    m_.retrieval_ms += now_ms() - t0;

    // adapt bookkeeping: which retrievers produced candidates
    unsigned produced = 0;
    for (auto& kv : pool) produced |= kv.second.methods;
    if (produced & kRetrieverExact) adapt_runs_[0] += 1;
    if (produced & kRetrieverLexical) adapt_runs_[1] += 1;
    if (produced & kRetrieverVector) adapt_runs_[2] += 1;
    if (produced & kRetrieverStructural) adapt_runs_[3] += 1;

    std::vector<Candidate> out;
    out.reserve(pool.size());
    for (auto& kv : pool) {
        unsigned bits = kv.second.methods;
        if ((bits & (bits - 1)) != 0) ++m_.duplicate_candidates;   // returned by more than one retriever; counted once
        out.push_back(kv.second);
    }
    return out;   // raw, unfused: fusion happens in search() so that L3 can be re-ranked cheaply
}

std::vector<Candidate> HybridIndex::search(const HybridQuery& q, RequestCache* rc) {
    std::lock_guard<std::mutex> lock(mu_);
    ++m_.queries;
    std::vector<std::string> texts = {q.text};
    for (const auto& e : q.extra) texts.push_back(e);
    const std::string rver = ranking_version();
    std::unordered_map<uint32_t, Candidate> merged;

    for (const auto& text : texts) {
        const std::string base = key_for(q, text);
        const std::string full = base + "\x1e" + rver;
        const std::string l0key = full + "\x1e" + std::to_string(version_);
        std::vector<Candidate> ranked;
        bool have = false;

        if (rc) {
            auto it = rc->results.find(l0key);
            if (it != rc->results.end()) { ranked = it->second; have = true; ++m_.l0_hits; }
        }
        if (!have) {   // negative cache: a definitive "nothing" for this exact state of the index
            auto n = neg_.find(base);
            if (n != neg_.end() && n->second.version != version_ && terms_absent(n->second.terms)) n->second.version = version_;   // the appends added none of its terms
            if (n != neg_.end() && n->second.version == version_) { ++n->second.hits; ++m_.negative_hits; have = true; }
        }
        if (!have) {
            auto h = l2_.find(full);
            if (h != l2_.end() && h->second.version == version_) {
                ranked = h->second.cands;
                ++h->second.hits; ++m_.l2_hits; have = true;
                // defense in depth: a cached result is re-authorized when it is returned
                ranked.erase(std::remove_if(ranked.begin(), ranked.end(), [&](const Candidate& c) { return q.allowed && !q.allowed(c.doc); }), ranked.end());
            }
        }
        std::vector<Candidate> raw;
        double cost = 0;
        if (!have) {
            auto r3 = l3_.find(base);
            if (r3 != l3_.end() && r3->second.version == version_) {
                raw = r3->second.cands;
                ++r3->second.hits; ++m_.l3_hits; ++m_.l5_hits;   // the ranking policy changed: candidates reused, only fusion repeats
                raw.erase(std::remove_if(raw.begin(), raw.end(), [&](const Candidate& c) { return q.allowed && !q.allowed(c.doc); }), raw.end());
            } else {
                const double t0 = now_ms();
                raw = run(q, text);
                cost = now_ms() - t0;
                if (docs_.size() >= cfg_.l2_admit_min_docs || (plan_query(text).retrievers & kRetrieverVector)) cache_put(l3_, l3_order_, cfg_.query_cache_entries, base, raw, cost);
            }
            // fusion: normalized signals, configurable adaptive weights, small bonuses that never create relevance
            const double t1 = now_ms();
            for (auto& c : raw) {
                double s = c.lexical * weight(kRetrieverLexical) + c.exact * weight(kRetrieverExact) + c.vector * weight(kRetrieverVector) + c.structural * weight(kRetrieverStructural);
                if (s > 0) {
                    if (!q.locality_prefix.empty() && docs_[c.doc].source.compare(0, q.locality_prefix.size(), q.locality_prefix) == 0) s += cfg_.w_locality;
                    s += cfg_.w_authority * docs_[c.doc].authority;
                }
                c.score = std::min(1.0, s);
            }
            raw.erase(std::remove_if(raw.begin(), raw.end(), [](const Candidate& c) { return c.score <= 0; }), raw.end());
            std::sort(raw.begin(), raw.end(), [](const Candidate& a, const Candidate& b) { return a.score != b.score ? a.score > b.score : a.doc < b.doc; });
            if (raw.size() > cfg_.result_limit) raw.resize(cfg_.result_limit);
            m_.rerank_ms += now_ms() - t1;
            ranked = raw;
            if (ranked.empty()) {
                const QueryPlan np = plan_query(text);
                std::vector<std::string> terms = index_terms(text);
                for (auto& t : np.identifiers) terms.push_back(t);
                for (auto& t : np.expanded) terms.push_back(t);
                cache_put(neg_, neg_order_, cfg_.negative_cache_entries, base, {}, cost, std::move(terms));
            }
            else if (docs_.size() >= cfg_.l2_admit_min_docs) cache_put(l2_, l2_order_, cfg_.query_cache_entries, full, ranked, cost);
        }
        if (rc) rc->results[l0key] = ranked;
        for (auto& c : ranked) {
            auto it = merged.find(c.doc);
            if (it == merged.end()) merged[c.doc] = c;
            else if (c.score > it->second.score) { unsigned m2 = it->second.methods | c.methods; it->second = c; it->second.methods = m2; }
        }
    }
    std::vector<Candidate> out;
    out.reserve(merged.size());
    for (auto& kv : merged) out.push_back(kv.second);
    std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) { return a.score != b.score ? a.score > b.score : a.doc < b.doc; });
    return out;
}

// ---- L6 and the loop guard ----
bool HybridIndex::materialized_get(const std::string& key, std::string& out) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = l6_.find(key);
    if (it == l6_.end()) return false;
    out = it->second;
    ++m_.l6_hits;
    return true;
}
void HybridIndex::materialized_put(const std::string& key, const std::string& block) {
    std::lock_guard<std::mutex> lock(mu_);
    if (block.size() > cfg_.materialized_cache_bytes / 4) return;   // admission: one huge block must not evict everything else
    if (!l6_.count(key)) { l6_order_.push_back(key); l6_bytes_ += block.size(); } else l6_bytes_ += block.size() - l6_[key].size();
    l6_[key] = block;
    while (l6_bytes_ > cfg_.materialized_cache_bytes && !l6_order_.empty()) {
        auto f = l6_.find(l6_order_.front());
        if (f != l6_.end()) { l6_bytes_ -= f->second.size(); l6_.erase(f); }
        l6_order_.pop_front();
    }
}

int HybridIndex::observe_state(const std::string& query_hash, const std::string& candidate_hash, bool new_evidence) {
    std::lock_guard<std::mutex> lock(mu_);
    const std::string k = query_hash + "|" + candidate_hash;
    if (new_evidence) { loop_.erase(k); return 0; }
    if (loop_.size() > 4096) loop_.clear();
    const int n = ++loop_[k];
    if (n >= cfg_.loop_repeat_limit) ++m_.loop_throttled;
    return n;
}
bool HybridIndex::loop_throttled(const std::string& query_hash, const std::string& candidate_hash) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = loop_.find(query_hash + "|" + candidate_hash);
    return it != loop_.end() && it->second >= cfg_.loop_repeat_limit;
}

RetrievalMetrics HybridIndex::metrics() const {
    std::lock_guard<std::mutex> lock(mu_);
    return m_;
}

} // namespace strata::rt
