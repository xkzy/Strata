// src/rt/context_manager.cpp
#include "strata/rt/context_manager.hpp"

#include "strata/rt/claim.hpp"

#include <algorithm>
#include <cctype>
#include <set>

namespace strata::rt {

namespace {
std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
size_t est_tokens(const std::string& s) { return (s.size() + 3) / 4; }
std::string clip(const std::string& s, size_t n) { return s.size() <= n ? s : s.substr(0, n) + "..."; }
} // namespace

ContextManager::ContextManager(ContextConfig cfg, std::vector<std::shared_ptr<IEvidenceProvider>> providers, std::shared_ptr<MemoryStore> memory)
    : cfg_(cfg), providers_(std::move(providers)), memory_(std::move(memory)) {}

bool ContextManager::needs_retrieval(const std::string& user_text) {
    const std::string t = lower(user_text);
    static const char* cues[] = {"configured", "configuration", "config", "setting", "settings", "which port", "what port", "port is", "what version",
                                 "which version", "where is", "where are", "what is the value", "current value", "in this repo", "in the repo",
                                 "this project", "our project", "the codebase", "in our", "the file", "which file", "what file", "path to", "environment variable",
                                 "env var", "according to", "per the docs", "documentation says", "does the server", "does it use", "what does the"};
    for (const char* c : cues) if (t.find(c) != std::string::npos) return true;
    // identifiers or paths in the question
    if (t.find('`') != std::string::npos || t.find('/') != std::string::npos) return true;
    return false;
}

EvidenceSet ContextManager::retrieve(const RequestScope& scope, const std::string& query, size_t top_k) const {
    EvidenceSet set;
    std::set<std::string> seen;
    for (const auto& p : providers_) {
        for (auto& ev : p->retrieve(scope, query, top_k)) {
            if (seen.insert(ev.id).second) set.items.push_back(std::move(ev));
        }
    }
    if (memory_ && cfg_.memory_enabled) {
        for (auto& ev : memory_->retrieve(scope, query, top_k)) if (seen.insert(ev.id).second) set.items.push_back(std::move(ev));
    }
    return set;
}

PreparedContext ContextManager::prepare(const RequestScope& scope, const std::vector<Message>& conversation) {
    PreparedContext out;
    std::string last_user;
    for (auto it = conversation.rbegin(); it != conversation.rend(); ++it)
        if (it->role == "user") { last_user = it->content; break; }

    // 1. evidence: tool results that arrived with the request are always in play; the rest only when the question needs it
    const bool want = cfg_.rag_enabled && needs_retrieval(last_user);
    if (want) out.evidence = retrieve(scope, last_user, cfg_.rag_top_k);
    else {
        // cheap, authoritative and request-local sources are still available to the verifier
        for (const auto& p : providers_) {
            if (p->name() == "request" || p->name() == "exact_state" || p->name() == "virtual_context")
                for (auto& ev : p->retrieve(scope, last_user, cfg_.rag_top_k)) out.evidence.items.push_back(std::move(ev));
        }
    }
    out.retrieval_used = want && !out.evidence.empty();

    // 2. memory facts that bear on the question
    std::vector<MemoryEntry> mem;
    if (memory_ && cfg_.memory_enabled) mem = memory_->search(scope.security, last_user, 3);
    out.memory_items_used = mem.size();

    // 3. the internal context block: snippets only, never whole documents
    std::string block;
    if (!mem.empty()) {
        block += "Known about this user:\n";
        for (const auto& m : mem) block += "- " + clip(m.value, cfg_.snippet_chars) + "\n";
    }
    if (out.retrieval_used) {
        block += "Relevant information:\n";
        for (const auto& ev : out.evidence.items) {
            if (ev.kind == EvidenceKind::kMemory) continue;
            block += "- [" + ev.source_id + "] " + clip(ev.text, cfg_.snippet_chars) + "\n";
        }
    }

    // 4. assemble within the token budget: system messages first, then the newest conversation turns
    std::vector<Message> system_msgs, convo;
    for (const auto& m : conversation) (m.role == "system" ? system_msgs : convo).push_back(m);
    size_t budget = cfg_.max_prompt_tokens;
    size_t used = 0;
    for (const auto& m : system_msgs) used += est_tokens(m.content);
    if (!block.empty()) used += est_tokens(block);

    std::vector<Message> kept;
    size_t dropped = 0;
    for (auto it = convo.rbegin(); it != convo.rend(); ++it) {
        size_t t = est_tokens(it->content) + 4;
        if (used + t > budget && !kept.empty()) { ++dropped; continue; }   // always keep the newest turn
        used += t;
        kept.push_back(*it);
    }
    std::reverse(kept.begin(), kept.end());
    out.dropped_messages = dropped;

    out.messages = system_msgs;
    if (!block.empty()) {
        Message ctx;
        ctx.role = "system";
        ctx.content = block;
        out.messages.push_back(std::move(ctx));
    }
    for (auto& m : kept) out.messages.push_back(std::move(m));
    out.estimated_tokens = used;
    return out;
}

} // namespace strata::rt
