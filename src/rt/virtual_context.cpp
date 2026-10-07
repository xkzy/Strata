// src/rt/virtual_context.cpp - virtual context window (see virtual_context.hpp)
#include "strata/rt/virtual_context.hpp"

#include "strata/rt/retrieval.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <list>
#include <sstream>

namespace strata::rt {

namespace {

using namespace text;

size_t est_tokens(const std::string& s) { return (s.size() + 3) / 4; }

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

bool references_earlier_context(const std::string& query) {
    const std::string q = lower(query);
    // phrases that point back at the session; bare words like "before" or "originally" occur in ordinary requests
    static const char* cues[] = {"we discussed", "did we discuss", "we discuss earlier", "did we decide", "did we agree", "did i say", "did you say", "did i mention", "you said", "i said", "i told you", "you told me", "as i mentioned", "as we discussed", "last time we",
                                 "mentioned above", "we agreed", "we decided", "discussed earlier", "mentioned earlier", "said earlier", "decided earlier",
                                 "agreed earlier", "earlier we", "earlier you", "earlier i ", "back when we", "you suggested", "you proposed",
                                 "remember when", "previously discussed", "previously mentioned", "previously decided", "the one from earlier", "from our earlier",
                                 "we talked about", "you mentioned", "i mentioned", "we used before", "we chose", "we picked"};
    for (const char* c : cues) if (q.find(c) != std::string::npos) return true;
    return false;
}

std::string make_summary(const std::string& text, size_t max_chars) {
    std::string t = trim(text);
    // first sentence-ish piece
    size_t cut = std::string::npos;
    for (size_t i = 0; i < t.size() && i < max_chars * 2; ++i) {
        if ((t[i] == '.' || t[i] == '!' || t[i] == '?' || t[i] == '\n') && i >= 20) { cut = i + 1; break; }
    }
    std::string s = cut == std::string::npos ? t : t.substr(0, cut);
    if (s.size() > max_chars) { s.resize(max_chars); s += "..."; }
    for (auto& c : s) if (c == '\n' || c == '\r') c = ' ';
    return trim(s);
}

std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '\\') o += "\\\\";
        else if (c == '\t') o += "\\t";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else o += c;
    }
    return o;
}
std::string unesc(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char n = s[++i];
            o += n == 't' ? '\t' : n == 'n' ? '\n' : n == 'r' ? '\r' : n;
        } else o += s[i];
    }
    return o;
}
std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> f;
    size_t b = 0;
    while (true) {
        size_t e = line.find('\t', b);
        if (e == std::string::npos) { f.push_back(unesc(line.substr(b))); break; }
        f.push_back(unesc(line.substr(b, e - b)));
        b = e + 1;
    }
    return f;
}

ItemKind kind_for_role(const std::string& role) {
    if (role == "system") return ItemKind::kSystem;
    if (role == "assistant") return ItemKind::kAssistant;
    if (role == "tool") return ItemKind::kToolResult;
    return ItemKind::kUser;
}

bool directive_text(const std::string& t) {
    const std::string l = lower(trim(t)).substr(0, 40);
    return l.rfind("remember", 0) == 0 || l.rfind("important:", 0) == 0 || l.rfind("decision:", 0) == 0 || l.rfind("rule:", 0) == 0 ||
           l.rfind("always ", 0) == 0 || l.rfind("never ", 0) == 0 || l.rfind("pin:", 0) == 0;
}

// Persisted context is private agent memory: directories 0700, files 0600, written to a temp name and renamed.
bool make_private_dir(const std::string& dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec || !fs::is_directory(dir, ec)) return false;
#ifndef _WIN32
    struct stat sb;
    if (::stat(dir.c_str(), &sb) != 0 || sb.st_uid != ::geteuid()) return false;   // never use a directory someone else owns
    if ((sb.st_mode & 077) != 0 && ::chmod(dir.c_str(), 0700) != 0) return false;
#endif
    return true;
}
bool write_private_file(const std::string& path, const std::string& data) {
#ifndef _WIN32
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) return false;
    size_t off = 0;
    while (off < data.size()) {
        ssize_t n = ::write(fd, data.data() + off, data.size() - off);
        if (n <= 0) { ::close(fd); return false; }
        off += static_cast<size_t>(n);
    }
    return ::close(fd) == 0;
#else
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(data.data(), static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(f);
#endif
}

} // namespace

// ---------------- file backend ----------------
FilePageBackend::FilePageBackend(std::string dir) : dir_(std::move(dir)) {
    (void)make_private_dir(dir_);   // on failure put() fails visibly
}
std::string FilePageBackend::path(const std::string& hash) const { return dir_ + "/" + hash + ".page"; }
bool FilePageBackend::has(const std::string& hash) { std::ifstream f(path(hash), std::ios::binary); return f.good(); }
bool FilePageBackend::put(const std::string& hash, const std::string& content) {
    if (has(hash)) return true;
    const std::string tmp = path(hash) + ".tmp";
    if (!write_private_file(tmp, content)) return false;
    return std::rename(tmp.c_str(), path(hash).c_str()) == 0;
}
bool FilePageBackend::get(const std::string& hash, std::string& content) {
    std::ifstream f(path(hash), std::ios::binary);
    if (!f) return false;
    std::ostringstream ss; ss << f.rdbuf();
    content = ss.str();
    return hash128(content) == hash;   // content addressed: a corrupted file is "unavailable", not wrong text
}

std::string ContextWindowInfo::to_json() const {
    std::ostringstream o;
    o << "{\"context_window\":{\"physical\":" << physical << ",\"virtual\":" << virtual_advertised << ",\"virtual_capacity\":" << virtual_capacity
      << ",\"virtual_used\":" << virtual_used << ",\"resident\":" << resident << "}}";
    return o.str();
}

// ---------------- implementation ----------------
struct VirtualContext::Impl {
    VirtualContextConfig cfg;
    std::shared_ptr<IPageBackend> backend;
    mutable std::mutex mu;

    struct Item {
        uint64_t id = 0;
        ItemKind kind = ItemKind::kUser;
        std::string role, source, key;
        Provenance provenance = Provenance::kInternalContext;
        bool pinned = false;
        int priority = 0;
        context::SharingScope sharing = context::SharingScope::PRIVATE;
        std::vector<uint64_t> pages;
        size_t tokens = 0;
    };

    std::vector<ContextPage> pages;      // page_id = index + 1
    std::vector<Item> items;             // item_id = index + 1
    mutable std::unordered_map<std::string, std::string> resident;     // hash -> content
    mutable uint64_t resident_bytes = 0;
    std::set<std::string> spilled;       // hashes known to live only in the backend
    std::unique_ptr<HybridIndex> index;                        // exact / lexical / vector / structural retrieval + caches (retrieval.hpp)
    std::unordered_map<std::string, uint64_t> entity_first;   // entity -> first page id
    std::vector<uint32_t> dependents;                          // per page idx
    std::unordered_map<std::string, std::vector<uint64_t>> msg_index;   // role|hash -> item ids
    std::set<uint64_t> prev_hot;
    int64_t last_resident_tokens = 0;
    uint64_t logical_tokens = 0, content_bytes = 0;
    mutable VirtualContextStats st;
    KvIdentity kv_id;
    std::vector<int32_t> kv_tokens;
    context::SecurityScope owner;
    bool dirty = false;
    std::set<std::string> persisted;   // hashes already written to the backend

    ContextPage& pg(uint64_t id) { return pages[id - 1]; }
    const ContextPage& pg(uint64_t id) const { return pages[id - 1]; }

    int64_t capacity() const {
        // each logical token costs about 4 bytes of content plus ~4 bytes of index and metadata
        const int64_t by_storage = static_cast<int64_t>(cfg.storage_budget_bytes / 8);
        return std::min(cfg.advertised_virtual_tokens, by_storage);
    }

    bool content_locked(const ContextPage& p, std::string& out) const {
        auto it = resident.find(p.content_hash);
        if (it != resident.end()) { out = it->second; return true; }
        if (backend) {
            std::string c;
            if (backend->get(p.content_hash, c)) {
                resident_bytes += c.size();
                out = c;
                resident[p.content_hash] = std::move(c);
                return true;
            }
        }
        ++st.backend_failures;
        return false;
    }

    std::vector<std::string> split_chunks(const std::string& text) const {
        std::vector<std::string> out;
        const size_t target = std::max<size_t>(64, cfg.chunk_tokens * 4);
        size_t pos = 0;
        while (pos < text.size()) {
            size_t end = std::min(text.size(), pos + target);
            if (end < text.size()) {
                size_t nl = text.rfind('\n', end);
                if (nl != std::string::npos && nl > pos + target / 2) end = nl + 1;
                else {
                    size_t sp = text.find_last_of(" \t", end);
                    if (sp != std::string::npos && sp > pos + target / 2) end = sp + 1;
                }
            }
            out.push_back(text.substr(pos, end - pos));
            pos = end;
        }
        return out;
    }

    void index_page(const ContextPage& p, const std::string& text) {
        DocInfo info;
        info.source = p.source;
        info.authority = p.pinned || p.kind == ItemKind::kSystem ? 1.0 : std::min(1.0, std::max(0.0, p.priority / 10.0));
        index->add(static_cast<uint32_t>(p.page_id - 1), text, p.content_hash, info);
    }

    void add_dependencies(ContextPage& p, const std::string& text) {
        std::set<uint64_t> deps(p.dependency.begin(), p.dependency.end());
        for (const auto& e : entities_of(text, 12)) {
            auto it = entity_first.find(e);
            if (it == entity_first.end()) { entity_first[e] = p.page_id; continue; }
            if (pg(it->second).item_id != p.item_id) deps.insert(it->second);
        }
        p.dependency.assign(deps.begin(), deps.end());
        if (p.dependency.size() > 4) p.dependency.resize(4);
        if (dependents.size() < pages.size()) dependents.resize(pages.size(), 0);
        for (uint64_t d : p.dependency) if (d >= 1 && d <= dependents.size()) ++dependents[d - 1];
    }

    std::string msg_key(const std::string& role, const std::string& content) const { return role + "\x1f" + hash128(trim(content)); }

    uint64_t append_locked(const NewItem& ni) {
        const size_t toks = est_tokens(ni.content);
        if (!ni.bypass_capacity && static_cast<int64_t>(logical_tokens + toks) > capacity()) { ++st.rejected_items; return 0; }
        Item it;
        it.id = items.size() + 1;
        it.kind = ni.kind; it.role = ni.role; it.source = ni.source; it.provenance = ni.provenance;
        it.priority = ni.priority; it.sharing = ni.sharing;
        it.pinned = ni.pinned || ni.kind == ItemKind::kSystem || (ni.kind == ItemKind::kUser && directive_text(ni.content));
        it.key = msg_key(ni.role, ni.content);
        it.tokens = toks;
        if (ni.kind == ItemKind::kSystem) {   // a newer system prompt from the same source supersedes the older one (kept, no longer pinned)
            for (auto& o : items) if (o.kind == ItemKind::kSystem && o.source == ni.source && o.pinned) {
                o.pinned = false;
                for (uint64_t pid : o.pages) pg(pid).pinned = false;
            }
        }

        std::set<uint64_t> deps;
        for (uint64_t d : ni.depends_on_items) if (d >= 1 && d <= items.size() && !items[d - 1].pages.empty()) deps.insert(items[d - 1].pages.front());
        // conversation structure: a reply depends on what it answers; a tool result on the call before it
        if (!items.empty()) {
            if (ni.kind == ItemKind::kAssistant) { for (size_t k = items.size(); k-- > 0;) if (items[k].kind == ItemKind::kUser && !items[k].pages.empty()) { deps.insert(items[k].pages.front()); break; } }
            if (ni.kind == ItemKind::kToolResult) { for (size_t k = items.size(); k-- > 0;) if (items[k].kind == ItemKind::kAssistant && !items[k].pages.empty()) { deps.insert(items[k].pages.front()); break; } }
        }

        auto chunks = split_chunks(ni.content);
        if (chunks.empty()) chunks.push_back("");
        uint32_t ci = 0;
        for (auto& ch : chunks) {
            ContextPage p;
            p.page_id = pages.size() + 1;
            p.item_id = it.id;
            p.sequence_position = p.page_id;
            p.token_begin = logical_tokens;
            p.tokens = est_tokens(ch);
            logical_tokens += p.tokens;
            p.token_end = logical_tokens;
            p.content_hash = hash128(ch);
            p.source = ni.source; p.provenance = ni.provenance; p.kind = ni.kind; p.role = ni.role;
            p.pinned = it.pinned; p.priority = ni.priority;
            p.chunk_index = ci++; p.chunk_count = static_cast<uint32_t>(chunks.size());
            p.sharing = ni.sharing;
            p.owner = owner;
            p.summary = make_summary(ch, cfg.summary_chars);
            p.tier = ContextTier::kWarm;
            p.dependency.assign(deps.begin(), deps.end());
            pages.push_back(p);
            it.pages.push_back(p.page_id);
            // content (deduplicated by hash)
            if (!resident.count(p.content_hash) && !spilled.count(p.content_hash)) { resident[p.content_hash] = ch; resident_bytes += ch.size(); content_bytes += ch.size(); }
            index_page(pg(p.page_id), ch);
            add_dependencies(pg(p.page_id), ch);
        }
        msg_index[it.key].push_back(it.id);
        items.push_back(std::move(it));
        dirty = true;
        if (resident_bytes > cfg.resident_bytes + cfg.resident_bytes / 4) compact_locked();
        return items.back().id;
    }

    // ---- retrieval: the hybrid index decides which retrievers run; authorization is applied before anything is scored ----
    static std::string scope_key(const context::SecurityScope& r) {
        std::vector<std::string> perms(r.permissions.begin(), r.permissions.end());
        std::sort(perms.begin(), perms.end());
        std::string k = r.tenant_id + "\x1f" + r.user_id + "\x1f" + r.workspace_id + "\x1f" + r.agent_id + "\x1f" + r.session_id + "\x1f" + std::to_string(static_cast<int>(r.sharing_scope));
        for (const auto& p : perms) k += "\x1f" + p;
        return k;
    }
    std::vector<Candidate> retrieve(const std::string& query, const context::SecurityScope& requester, const std::vector<std::string>& extra = {}, RequestCache* rc = nullptr) const {
        HybridQuery q;
        q.text = query;
        q.extra = extra;
        q.scope_key = scope_key(requester);
        q.allowed = [this, &requester](uint32_t idx) { return idx < pages.size() && requester.can_access(pages[idx].owner, pages[idx].sharing); };
        return index->search(q, rc);
    }

    double importance(const ContextPage& p, double rel, uint64_t newest, bool was_hot) const {
        const double age = static_cast<double>(newest - std::min(newest, p.sequence_position));
        const double recency = 1.0 / (1.0 + age / 64.0);
        const uint32_t deps = p.page_id - 1 < dependents.size() ? dependents[p.page_id - 1] : 0;
        double s = cfg.w_relevance * rel + cfg.w_recency * recency + cfg.w_frequency * std::log1p(p.access_frequency) +
                   cfg.w_dependents * std::log1p(deps) + cfg.w_verified * std::log1p(p.verification_refs) + 0.1 * p.priority;
        if (p.pinned) s += cfg.w_pinned;
        if ((p.kind == ItemKind::kToolResult || p.kind == ItemKind::kFile) && deps > 0) s += cfg.w_tool;
        if (was_hot) s += cfg.hot_bonus;
        return s;
    }

    size_t compact_locked() {
        if (!backend) return 0;
        std::vector<std::pair<double, const ContextPage*>> cold;
        const uint64_t newest = pages.size();
        for (const auto& p : pages) {
            if (p.pinned || prev_hot.count(p.page_id) || !resident.count(p.content_hash)) continue;
            cold.push_back({importance(p, 0, newest, false), &p});
        }
        std::sort(cold.begin(), cold.end(), [](auto& a, auto& b) { return a.first < b.first; });
        size_t spilled_pages = 0;
        for (auto& c : cold) {
            if (resident_bytes <= cfg.resident_bytes) break;
            const std::string& h = c.second->content_hash;
            auto it = resident.find(h);
            if (it == resident.end()) continue;
            if (!backend->put(h, it->second)) { ++st.backend_failures; continue; }
            resident_bytes -= it->second.size();
            resident.erase(it);
            spilled.insert(h);
            ++spilled_pages;
        }
        for (auto& p : pages) if (spilled.count(p.content_hash) && !resident.count(p.content_hash) && p.tier != ContextTier::kHot) p.tier = ContextTier::kArchive;
        return spilled_pages;
    }
};

VirtualContext::VirtualContext(std::string context_id, context::SecurityScope owner, VirtualContextConfig cfg, std::shared_ptr<IPageBackend> backend, std::shared_ptr<EmbeddingCache> embeddings)
    : impl_(new Impl), id_(std::move(context_id)), owner_(std::move(owner)) {
    impl_->cfg = std::move(cfg);
    impl_->index = std::make_unique<HybridIndex>(impl_->cfg.retrieval, nullptr, std::move(embeddings));
    impl_->backend = std::move(backend);
    impl_->owner = owner_;
}
VirtualContext::~VirtualContext() = default;

uint64_t VirtualContext::append(const NewItem& item) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->append_locked(item);
}

std::vector<uint64_t> VirtualContext::ingest(const std::vector<Message>& messages, std::set<uint64_t>* newly_added) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    std::vector<uint64_t> ids;
    uint64_t cursor = 0;
    size_t current_from = 0;   // the present request: everything after the last assistant turn
    for (size_t i = messages.size(); i-- > 0;) if (messages[i].role == "assistant") { current_from = i + 1; break; }
    for (size_t i = 0; i < messages.size(); ++i) {
        const Message& m = messages[i];
        const std::string key = impl_->msg_key(m.role, m.content);
        uint64_t found = 0;
        auto it = impl_->msg_index.find(key);
        if (it != impl_->msg_index.end()) {
            auto lb = std::upper_bound(it->second.begin(), it->second.end(), cursor);
            if (lb != it->second.end()) found = *lb;
        }
        // the newest message is new unless it is an exact retry of the very last stored item
        if (found && i + 1 == messages.size() && found != impl_->items.size()) found = 0;
        if (!found) {
            NewItem ni;
            ni.role = m.role == "tool" || m.role == "assistant" || m.role == "system" ? m.role : "user";
            ni.kind = kind_for_role(m.role);
            ni.source = ni.kind == ItemKind::kToolResult ? (m.name.empty() ? "tool" : m.name) : "conversation";
            ni.content = m.content;
            ni.sharing = context::SharingScope::PRIVATE;
            ni.bypass_capacity = i >= current_from;
            found = impl_->append_locked(ni);
            if (found && newly_added) newly_added->insert(found);
        }
        ids.push_back(found);
        if (found) cursor = std::max(cursor, found);
    }
    return ids;
}

void VirtualContext::pin(uint64_t item_id, bool pinned) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (item_id < 1 || item_id > impl_->items.size()) return;
    auto& it = impl_->items[item_id - 1];
    it.pinned = pinned;
    for (uint64_t pid : it.pages) impl_->pg(pid).pinned = pinned;
    impl_->dirty = true;
}

void VirtualContext::note_used_as_evidence(uint64_t page_id) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (page_id >= 1 && page_id <= impl_->pages.size()) ++impl_->pg(page_id).verification_refs;
}

bool VirtualContext::page_content(uint64_t page_id, std::string& out) const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (page_id < 1 || page_id > impl_->pages.size()) return false;
    return impl_->content_locked(impl_->pg(page_id), out);
}
bool VirtualContext::page_info(uint64_t page_id, ContextPage& out) const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (page_id < 1 || page_id > impl_->pages.size()) return false;
    out = impl_->pg(page_id);
    return true;
}
std::vector<uint64_t> VirtualContext::item_pages(uint64_t item_id) const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (item_id < 1 || item_id > impl_->items.size()) return {};
    return impl_->items[item_id - 1].pages;
}

std::vector<SearchHit> VirtualContext::search(const context::SecurityScope& requester, const std::string& query, size_t top_k, int prov) const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    ++impl_->st.search_queries;
    auto cands = impl_->retrieve(query, requester);
    std::vector<std::pair<double, uint32_t>> ranked;
    for (auto& c : cands) {
        const ContextPage& p = impl_->pages[c.doc];
        if (prov >= 0 && static_cast<int>(p.provenance) != prov) continue;
        ranked.push_back({c.score, c.doc});
    }
    std::sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    std::vector<SearchHit> out;
    for (auto& r : ranked) {
        if (out.size() >= top_k) break;
        const ContextPage& p = impl_->pages[r.second];
        SearchHit h;
        h.page_id = p.page_id; h.score = r.first; h.provenance = p.provenance; h.source = p.source; h.content_hash = p.content_hash;
        if (!impl_->content_locked(p, h.text)) continue;   // unrecoverable content is not returned (and not invented)
        out.push_back(std::move(h));
    }
    return out;
}

WorkingSet VirtualContext::select(const SelectionRequest& req) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    Impl& m = *impl_;
    WorkingSet ws;
    ++m.st.selections;
    const uint64_t newest = m.pages.size();
    const int64_t budget = std::max<int64_t>(256, req.budget_tokens);
    ws.references_earlier = references_earlier_context(req.query);

    std::map<uint64_t, bool> chosen;      // page id -> as_summary
    int64_t used = 0;
    auto cost = [&](const ContextPage& p, bool summary) -> int64_t { return static_cast<int64_t>(summary ? est_tokens(p.summary) : p.tokens) + 4; };
    auto authorized = [&](const ContextPage& p) { return req.requester.can_access(p.owner, p.sharing); };
    auto take = [&](uint64_t pid, bool summary) -> bool {
        const ContextPage& p = m.pg(pid);
        if (!authorized(p)) return false;
        auto it = chosen.find(pid);
        if (it != chosen.end()) {
            if (!it->second || summary) return true;           // already there in an equal or better form
            used -= cost(p, true); used += cost(p, false); it->second = false;   // upgrade summary -> original
            return true;
        }
        chosen[pid] = summary;
        used += cost(p, summary);
        return true;
    };
    auto fits = [&](const ContextPage& p, bool summary, int64_t limit) { return used + cost(p, summary) <= limit; };

    // relevance of every authorized page to the query (and to recovery's extra queries)
    std::unordered_map<uint32_t, double> rel;
    std::unordered_map<uint32_t, Candidate> rel_cands;
    for (auto& c : m.retrieve(req.query, req.requester, req.extra_queries)) { rel[c.doc] = c.score; rel_cands[c.doc] = c; }
    auto rel_of = [&](uint64_t pid) { auto it = rel.find(static_cast<uint32_t>(pid - 1)); return it == rel.end() ? 0.0 : it->second; };

    // A: the present request. Small items are mandatory; huge ones (a 500K-line tool result) contribute their head and
    // their relevant regions only - the rest stays in the logical context.
    const int64_t huge = budget / 4;
    std::set<uint64_t> big_current;
    for (uint64_t iid : req.current_items) {
        if (iid < 1 || iid > m.items.size()) continue;
        const auto& it = m.items[iid - 1];
        if (static_cast<int64_t>(it.tokens) <= huge) { for (uint64_t pid : it.pages) take(pid, false); }
        else {
            big_current.insert(iid);
            if (!it.pages.empty()) take(it.pages.front(), false);
        }
    }
    // B: pinned context
    {
        const int64_t limit = used + static_cast<int64_t>(budget * m.cfg.pinned_fraction);
        std::vector<const Impl::Item*> pinned;
        for (const auto& it : m.items) if (it.pinned && !req.current_items.count(it.id)) pinned.push_back(&it);
        std::stable_sort(pinned.begin(), pinned.end(), [](auto* a, auto* b) { return a->priority > b->priority; });
        for (auto* it : pinned)
            for (uint64_t pid : it->pages) {
                const ContextPage& p = m.pg(pid);
                if (fits(p, false, limit)) take(pid, false);
                else if (fits(p, true, limit)) take(pid, true);
                else ws.trace.push_back("pinned page " + std::to_string(pid) + " does not fit");
            }
    }
    // C: recent tail, newest first, until its share of the budget is used
    {
        const int64_t limit = used + static_cast<int64_t>(budget * m.cfg.recent_fraction);
        for (size_t k = m.items.size(); k-- > 0;) {
            const auto& it = m.items[k];
            if (req.current_items.count(it.id) || static_cast<int64_t>(it.tokens) > huge) continue;
            bool all = true;
            for (uint64_t pid : it.pages) if (!chosen.count(pid) && !fits(m.pg(pid), false, limit)) all = false;
            if (!all) break;
            for (uint64_t pid : it.pages) take(pid, false);
        }
    }
    // D: relevant older pages (+ their dependencies), strongest first
    {
        std::vector<std::pair<double, uint64_t>> cand;
        for (auto& kv : rel) {
            const ContextPage& p = m.pages[kv.first];
            if (chosen.count(p.page_id)) continue;
            const bool of_big_current = big_current.count(p.item_id) > 0;
            if (kv.second < m.cfg.min_relevance && !of_big_current) continue;
            cand.push_back({m.importance(p, kv.second, newest, m.prev_hot.count(p.page_id) > 0) + (of_big_current ? 0.5 : 0.0), p.page_id});
        }
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
        if (cand.size() > m.cfg.retrieval_candidates) cand.resize(m.cfg.retrieval_candidates);
        for (auto& c : cand) {
            const ContextPage& p = m.pg(c.second);
            if (chosen.count(p.page_id)) continue;
            // the page and the pages it depends on travel together when they fit
            std::vector<uint64_t> unit{p.page_id};
            std::vector<uint64_t> frontier = p.dependency;
            for (int depth = 0; depth < m.cfg.dependency_depth && !frontier.empty(); ++depth) {
                std::vector<uint64_t> next;
                for (uint64_t d : frontier) {
                    if (d < 1 || d > m.pages.size() || chosen.count(d) || std::find(unit.begin(), unit.end(), d) != unit.end()) continue;
                    if (!authorized(m.pg(d))) continue;
                    unit.push_back(d);
                    for (uint64_t dd : m.pg(d).dependency) next.push_back(dd);
                }
                frontier = std::move(next);
            }
            int64_t unit_cost = 0;
            for (uint64_t u : unit) unit_cost += cost(m.pg(u), false);
            if (used + unit_cost <= budget) { for (uint64_t u : unit) take(u, false); ws.retrieved += unit.size(); }
            else if (fits(p, false, budget)) {
                take(p.page_id, false); ++ws.retrieved;
                for (size_t k = 1; k < unit.size(); ++k) if (fits(m.pg(unit[k]), true, budget)) take(unit[k], true);   // dependencies as summaries
            } else if (fits(p, true, budget)) { take(p.page_id, true); ++ws.retrieved; }
        }
        {   // usefulness feedback: the retrievers that produced pages that made it into the working set
            std::vector<Candidate> used_c;
            for (auto& kv : chosen) { auto it = rel_cands.find(static_cast<uint32_t>(kv.first - 1)); if (it != rel_cands.end()) used_c.push_back(it->second); }
            if (!used_c.empty()) m.index->note_used(used_c);
        }
    }

    // retrieval confidence / coverage / unavailable context
    double best_old = 0;
    for (auto& kv : rel) {
        const ContextPage& p = m.pages[kv.first];
        if (req.current_items.count(p.item_id)) continue;
        best_old = std::max(best_old, kv.second);
    }
    ws.retrieval_confidence = best_old;
    {
        auto qents = entities_of(req.query);
        size_t found = 0;
        std::set<std::string> chosen_terms;
        for (auto& c : chosen) { std::string t; if (m.content_locked(m.pg(c.first), t)) for (auto& x : entities_of(t)) chosen_terms.insert(x); }
        for (auto& e : qents) {
            if (chosen_terms.count(e)) ++found;
            else if (ws.references_earlier && !m.index->has_term(e)) ws.unavailable.push_back("no stored context mentions '" + e + "'");
        }
        ws.coverage = qents.empty() ? 1.0 : static_cast<double>(found) / static_cast<double>(qents.size());
        if (ws.references_earlier && best_old < m.cfg.unavailable_floor && m.items.size() > req.current_items.size())
            ws.unavailable.push_back("the earlier context the request refers to could not be found");
        if (ws.references_earlier && m.items.size() <= req.current_items.size())
            ws.unavailable.push_back("there is no earlier context");
    }

    // materialize in logical order; pages whose original cannot be read are dropped and reported, never replaced by guesses
    std::vector<uint64_t> order;
    for (auto& c : chosen) order.push_back(c.first);
    std::sort(order.begin(), order.end(), [&](uint64_t a, uint64_t b) { return m.pg(a).sequence_position < m.pg(b).sequence_position; });
    std::set<uint64_t> now_hot;
    int64_t tokens = 0;
    uint64_t last_item = 0;
    uint64_t last_page = 0;
    for (uint64_t pid : order) {
        ContextPage& p = m.pg(pid);
        std::string text;
        bool as_summary = chosen[pid];
        if (as_summary) text = p.summary;
        else if (!m.content_locked(p, text)) {
            ws.unavailable.push_back("page " + std::to_string(pid) + " (" + p.source + ") could not be recovered from storage");
            continue;   // a summary is not a substitute for a missing original
        }
        if (as_summary) ++ws.summaries_used;
        const bool contiguous = last_item == p.item_id && last_page + 1 == pid && !ws.messages.empty() && !as_summary;
        if (contiguous) ws.messages.back().content += text;
        else {
            Message msg;
            msg.role = p.role;
            const bool headed = (p.kind == ItemKind::kFile || p.kind == ItemKind::kToolResult) && !p.source.empty();
            msg.content = headed ? p.source + ":\n" + text : text;
            if (p.kind == ItemKind::kToolResult) msg.name = p.source;
            ws.messages.push_back(std::move(msg));
        }
        last_item = p.item_id; last_page = pid;
        ws.page_ids.push_back(pid);
        ws.page_hashes.push_back(as_summary ? "s:" + p.content_hash : p.content_hash);
        tokens += cost(p, as_summary);
        now_hot.insert(pid);
        p.importance = m.importance(p, rel_of(pid), newest, m.prev_hot.count(pid) > 0);
        p.recency = 1.0 / (1.0 + static_cast<double>(newest - std::min(newest, p.sequence_position)) / 64.0);
        if (!req.current_items.count(p.item_id)) ++p.access_frequency;
        p.tier = ContextTier::kHot;
    }
    ws.tokens = tokens;
    for (uint64_t pid : now_hot) if (!m.prev_hot.count(pid)) ++ws.paged_in;
    for (uint64_t pid : m.prev_hot) if (!now_hot.count(pid)) { ++ws.paged_out; m.pg(pid).tier = m.spilled.count(m.pg(pid).content_hash) && !m.resident.count(m.pg(pid).content_hash) ? ContextTier::kArchive : ContextTier::kWarm; }
    m.st.page_ins += ws.paged_in; m.st.page_outs += ws.paged_out;
    m.prev_hot = std::move(now_hot);
    m.last_resident_tokens = tokens;
    ws.trace.push_back("working set: " + std::to_string(ws.page_ids.size()) + " pages, " + std::to_string(tokens) + " tokens; in " + std::to_string(ws.paged_in) +
                       ", out " + std::to_string(ws.paged_out) + ", retrieved " + std::to_string(ws.retrieved));
    return ws;
}

size_t VirtualContext::compact() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->compact_locked();
}

ContextWindowInfo VirtualContext::window_info() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    ContextWindowInfo w;
    w.physical = impl_->cfg.physical_tokens;
    w.virtual_advertised = impl_->cfg.advertised_virtual_tokens;
    w.virtual_capacity = impl_->capacity();
    w.virtual_used = static_cast<int64_t>(impl_->logical_tokens);
    w.resident = impl_->last_resident_tokens;
    return w;
}

int VirtualContext::observe_retrieval(const std::string& query_hash, const std::string& candidate_hash, bool new_evidence) {
    return impl_->index->observe_state(query_hash, candidate_hash, new_evidence);
}
bool VirtualContext::retrieval_throttled(const std::string& query_hash, const std::string& candidate_hash) const {
    return impl_->index->loop_throttled(query_hash, candidate_hash);
}

RetrievalMetrics VirtualContext::retrieval_metrics() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->index->metrics();
}

VirtualContextStats VirtualContext::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    VirtualContextStats s = impl_->st;
    s.pages = impl_->pages.size(); s.items = impl_->items.size(); s.logical_tokens = impl_->logical_tokens;
    s.content_bytes = impl_->content_bytes; s.resident_bytes = impl_->resident_bytes; s.spilled_pages = 0;
    for (const auto& p : impl_->pages) {
        ++s.tier_pages[static_cast<int>(p.tier)];
        if (impl_->spilled.count(p.content_hash) && !impl_->resident.count(p.content_hash)) ++s.spilled_pages;
    }
    return s;
}

KvReusePlan VirtualContext::plan_kv(const KvIdentity& id, const std::vector<int32_t>& prompt_ids) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    KvReusePlan plan = plan_kv_reuse(impl_->kv_id, impl_->kv_tokens, id, prompt_ids);
    impl_->kv_id = id;
    impl_->kv_tokens = prompt_ids;   // the physical KV now holds exactly this working set, nothing else
    return plan;
}

// ---------------- persistence ----------------
bool VirtualContext::save(const std::string& manifest_path) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    Impl& m = *impl_;
    if (!m.backend) return false;
    for (const auto& p : m.pages) {
        if (m.persisted.count(p.content_hash)) continue;
        std::string c;
        if (!m.content_locked(p, c)) return false;
        if (!m.backend->put(p.content_hash, c)) return false;
        m.persisted.insert(p.content_hash);
    }
    const std::string tmp = manifest_path + ".tmp";
    std::ostringstream f;
    {
        const auto& o = owner_;
        f << "H\t" << esc(id_) << '\t' << esc(o.tenant_id) << '\t' << esc(o.user_id) << '\t' << esc(o.workspace_id) << '\t' << esc(o.agent_id) << '\t' << esc(o.session_id) << '\n';
        for (const auto& it : m.items)
            f << "I\t" << it.id << '\t' << static_cast<int>(it.kind) << '\t' << esc(it.role) << '\t' << esc(it.source) << '\t' << static_cast<int>(it.provenance) << '\t'
              << it.pinned << '\t' << it.priority << '\t' << static_cast<int>(it.sharing) << '\t' << it.tokens << '\n';
        for (const auto& p : m.pages) {
            std::string deps;
            for (uint64_t d : p.dependency) deps += (deps.empty() ? "" : ",") + std::to_string(d);
            f << "P\t" << p.page_id << '\t' << p.item_id << '\t' << p.token_begin << '\t' << p.token_end << '\t' << p.content_hash << '\t' << deps << '\t' << esc(p.source) << '\t'
              << p.access_frequency << '\t' << p.verification_refs << '\t' << static_cast<int>(p.provenance) << '\t' << static_cast<int>(p.kind) << '\t' << esc(p.role) << '\t'
              << p.pinned << '\t' << p.priority << '\t' << p.chunk_index << '\t' << p.chunk_count << '\t' << p.tokens << '\t' << static_cast<int>(p.sharing) << '\t'
              << p.version << '\t' << esc(p.summary) << '\n';
        }
    }
    if (!write_private_file(tmp, f.str())) return false;
    const bool ok = std::rename(tmp.c_str(), manifest_path.c_str()) == 0;
    if (ok) m.dirty = false;
    return ok;
}

bool VirtualContext::save_if_dirty(const std::string& manifest_path) {
    { std::lock_guard<std::mutex> lock(impl_->mu); if (!impl_->dirty) return true; }
    return save(manifest_path);
}

std::shared_ptr<VirtualContext> VirtualContext::load(const std::string& manifest_path, VirtualContextConfig cfg, std::shared_ptr<IPageBackend> backend, std::shared_ptr<EmbeddingCache> embeddings) {
    std::ifstream f(manifest_path, std::ios::binary);
    if (!f) return nullptr;
    std::string line;
    std::shared_ptr<VirtualContext> vc;
    while (std::getline(f, line)) {
        auto fl = split_tabs(line);
        if (fl.empty()) continue;
        if (fl[0] == "H" && fl.size() >= 7) {
            context::SecurityScope o;
            o.tenant_id = fl[2]; o.user_id = fl[3]; o.workspace_id = fl[4]; o.agent_id = fl[5]; o.session_id = fl[6];
            vc.reset(new VirtualContext(fl[1], o, cfg, backend, embeddings));
        } else if (!vc) {
            return nullptr;
        } else if (fl[0] == "I" && fl.size() >= 10) {
            Impl::Item it;
            it.id = std::stoull(fl[1]); it.kind = static_cast<ItemKind>(std::stoi(fl[2])); it.role = fl[3]; it.source = fl[4];
            it.provenance = static_cast<Provenance>(std::stoi(fl[5])); it.pinned = fl[6] == "1"; it.priority = std::stoi(fl[7]);
            it.sharing = static_cast<context::SharingScope>(std::stoi(fl[8])); it.tokens = std::stoull(fl[9]);
            vc->impl_->items.push_back(std::move(it));
        } else if (fl[0] == "P" && fl.size() >= 21) {
            ContextPage p;
            p.page_id = std::stoull(fl[1]); p.item_id = std::stoull(fl[2]); p.sequence_position = p.page_id;
            p.token_begin = std::stoull(fl[3]); p.token_end = std::stoull(fl[4]); p.content_hash = fl[5];
            std::stringstream ds(fl[6]); std::string d;
            while (std::getline(ds, d, ',')) if (!d.empty()) p.dependency.push_back(std::stoull(d));
            p.source = fl[7]; p.access_frequency = static_cast<uint32_t>(std::stoul(fl[8])); p.verification_refs = static_cast<uint32_t>(std::stoul(fl[9]));
            p.provenance = static_cast<Provenance>(std::stoi(fl[10])); p.kind = static_cast<ItemKind>(std::stoi(fl[11])); p.role = fl[12];
            p.pinned = fl[13] == "1"; p.priority = std::stoi(fl[14]); p.chunk_index = static_cast<uint32_t>(std::stoul(fl[15]));
            p.chunk_count = static_cast<uint32_t>(std::stoul(fl[16])); p.tokens = std::stoull(fl[17]);
            p.sharing = static_cast<context::SharingScope>(std::stoi(fl[18])); p.version = std::stoull(fl[19]); p.summary = fl[20];
            p.owner = vc->owner_;
            p.tier = ContextTier::kArchive;
            vc->impl_->pages.push_back(std::move(p));
        }
    }
    if (!vc) return nullptr;
    // rebuild the derived state: index, entities, dependents, message keys. Content comes from the backend.
    Impl& m = *vc->impl_;
    m.dependents.assign(m.pages.size(), 0);
    for (auto& p : m.pages) {
        m.logical_tokens = std::max<uint64_t>(m.logical_tokens, p.token_end);
        std::string c;
        if (backend && backend->get(p.content_hash, c)) {
            m.spilled.insert(p.content_hash);
            m.persisted.insert(p.content_hash);
            m.content_bytes += c.size();
            m.index_page(p, c);
            for (const auto& e : entities_of(c, 12)) m.entity_first.emplace(e, p.page_id);
        } else {
            ++m.st.backend_failures;   // stays in the manifest, cannot be searched or materialized, and is reported as unavailable
            m.index->reserve_gap(static_cast<uint32_t>(p.page_id - 1));
        }
        for (uint64_t d : p.dependency) if (d >= 1 && d <= m.dependents.size()) ++m.dependents[d - 1];
    }
    // item -> pages and message keys
    for (const auto& p : m.pages) if (p.item_id >= 1 && p.item_id <= m.items.size()) m.items[p.item_id - 1].pages.push_back(p.page_id);
    for (auto& it : m.items) {
        std::string joined, c;
        bool complete = true;
        for (uint64_t pid : it.pages) { if (backend && backend->get(m.pg(pid).content_hash, c)) joined += c; else complete = false; }
        if (complete) { it.key = m.msg_key(it.role, joined); m.msg_index[it.key].push_back(it.id); }
    }
    return vc;
}

// ---------------- store ----------------
std::string VirtualContextStore::context_id(const context::SecurityScope& s) {
    return hash_parts({s.tenant_id, s.user_id, s.workspace_id, s.agent_id, s.session_id});
}
VirtualContextStore::VirtualContextStore(VirtualContextConfig cfg, std::shared_ptr<IPageBackend> backend, std::string storage_dir)
    : cfg_(std::move(cfg)), backend_(std::move(backend)), dir_(std::move(storage_dir)) {
    if (!dir_.empty() && !make_private_dir(dir_)) dir_.clear();   // an unusable or foreign-owned directory disables persistence
    if (!dir_.empty() && !backend_) backend_ = std::make_shared<FilePageBackend>(dir_ + "/blobs");
}
std::shared_ptr<VirtualContext> VirtualContextStore::open(const context::SecurityScope& scope) {
    const std::string id = context_id(scope);
    std::lock_guard<std::mutex> lock(mu_);
    auto it = contexts_.find(id);
    if (it != contexts_.end()) return it->second;
    std::shared_ptr<VirtualContext> vc;
    if (!dir_.empty()) vc = VirtualContext::load(dir_ + "/" + id + ".manifest", cfg_, backend_, embeddings_);
    // a manifest is only trusted for the scope it was written for
    if (vc && (vc->id() != id || VirtualContextStore::context_id(vc->owner()) != id)) vc.reset();
    if (!vc) vc = std::make_shared<VirtualContext>(id, scope, cfg_, backend_, embeddings_);
    contexts_[id] = vc;
    return vc;
}
std::shared_ptr<VirtualContext> VirtualContextStore::open_ephemeral(const context::SecurityScope& scope) {
    const std::string id = context_id(scope);
    std::lock_guard<std::mutex> lock(mu_);
    auto vc = std::make_shared<VirtualContext>(id, scope, cfg_, nullptr, embeddings_);
    contexts_[id] = vc;
    return vc;
}
void VirtualContextStore::erase(const context::SecurityScope& scope) {
    std::lock_guard<std::mutex> lock(mu_);
    contexts_.erase(context_id(scope));
}
bool VirtualContextStore::persist(const std::shared_ptr<VirtualContext>& vc) {
    if (dir_.empty() || !vc) return true;
    return vc->save_if_dirty(dir_ + "/" + vc->id() + ".manifest");
}
RetrievalMetrics VirtualContextStore::retrieval_metrics() const {
    std::vector<std::shared_ptr<VirtualContext>> open;
    { std::lock_guard<std::mutex> lock(mu_); for (auto& kv : contexts_) open.push_back(kv.second); }
    RetrievalMetrics t;
    for (auto& vc : open) {
        const RetrievalMetrics m = vc->retrieval_metrics();
        t.queries += m.queries; t.l0_hits += m.l0_hits; t.l1_hits += m.l1_hits; t.l2_hits += m.l2_hits; t.l3_hits += m.l3_hits; t.l5_hits += m.l5_hits;
        t.l6_hits += m.l6_hits; t.negative_hits += m.negative_hits;
        t.exact_runs += m.exact_runs; t.lexical_runs += m.lexical_runs; t.vector_runs += m.vector_runs; t.structural_runs += m.structural_runs;
        t.exact_hits += m.exact_hits; t.lexical_hits += m.lexical_hits; t.vector_hits += m.vector_hits; t.structural_hits += m.structural_hits;
        t.exact_short_circuits += m.exact_short_circuits; t.widened += m.widened; t.duplicate_candidates += m.duplicate_candidates; t.loop_throttled += m.loop_throttled;
        t.embedding_hits += m.embedding_hits; t.embedding_misses += m.embedding_misses;
        t.retrieval_ms += m.retrieval_ms; t.rerank_ms += m.rerank_ms; t.embedding_ms += m.embedding_ms;
        for (int i = 0; i < 10; ++i) t.class_counts[i] += m.class_counts[i];
    }
    return t;
}
size_t VirtualContextStore::size() const { std::lock_guard<std::mutex> lock(mu_); return contexts_.size(); }

} // namespace strata::rt
