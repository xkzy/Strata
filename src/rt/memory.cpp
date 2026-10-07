// src/rt/memory.cpp - persistent memory (see memory.hpp)
#include "strata/rt/memory.hpp"

#include <algorithm>
#include <cctype>
#include <set>
#include <sstream>

namespace strata::rt {

uint64_t MemoryStore::put(const context::SecurityScope& owner, context::SharingScope sharing, const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& e : entries_) {
        if (e.key == key && e.owner.to_string() == owner.to_string()) {
            if (e.value != value || !e.valid) { e.value = value; e.valid = true; ++e.version; }
            e.sharing = sharing;
            return e.version;
        }
    }
    MemoryEntry e;
    e.key = key; e.value = value; e.owner = owner; e.sharing = sharing;
    entries_.push_back(std::move(e));
    return 1;
}

bool MemoryStore::invalidate(const context::SecurityScope& owner, const std::string& key) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& e : entries_) {
        if (e.key == key && e.owner.to_string() == owner.to_string() && e.valid) { e.valid = false; ++e.version; return true; }
    }
    return false;
}

std::vector<MemoryEntry> MemoryStore::search(const context::SecurityScope& requester, const std::string& query, size_t top_k) const {
    std::set<std::string> qwords;
    {
        std::istringstream in(normalize_key(query));
        std::string w;
        while (in >> w) if (w.size() > 2) qwords.insert(w);
    }
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::pair<int, MemoryEntry>> scored;
    for (const auto& e : entries_) {
        if (!e.valid || !requester.can_access(e.owner, e.sharing)) continue;
        int hits = 0;
        std::istringstream in(normalize_key(e.key + " " + e.value));
        std::string w;
        while (in >> w) if (qwords.count(w)) ++hits;
        if (hits > 0) scored.emplace_back(hits, e);
    }
    std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<MemoryEntry> out;
    for (auto& p : scored) { if (out.size() >= top_k) break; out.push_back(std::move(p.second)); }
    return out;
}

size_t MemoryStore::observe_turn(const context::SecurityScope& owner, const std::string& user_text) {
    size_t stored = 0;
    std::string t = user_text;
    std::string low = t;
    std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c) { return std::tolower(c); });
    auto cut = [&](size_t from) {
        size_t end = t.find_first_of(".!?\n", from);
        std::string s = t.substr(from, end == std::string::npos ? std::string::npos : end - from);
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
        return s;
    };
    size_t p = low.find("remember that ");
    if (p != std::string::npos) {
        std::string fact = cut(p + 14);
        if (!fact.empty()) { put(owner, context::SharingScope::PRIVATE, "note:" + fact.substr(0, 40), fact); ++stored; }
    }
    for (const char* lead : {"my name is ", "i live in ", "i work at ", "i prefer "}) {
        size_t q = low.find(lead);
        if (q != std::string::npos) {
            std::string rest = cut(q + std::string(lead).size());
            if (!rest.empty()) { put(owner, context::SharingScope::PRIVATE, std::string("user:") + std::string(lead).substr(0, std::string(lead).size() - 1), rest); ++stored; }
        }
    }
    return stored;
}

size_t MemoryStore::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    size_t n = 0;
    for (const auto& e : entries_) if (e.valid) ++n;
    return n;
}

std::vector<Evidence> MemoryStore::retrieve(const RequestScope& scope, const std::string& query, size_t top_k) {
    std::vector<Evidence> out;
    for (const auto& m : search(scope.security, query, top_k)) {
        Evidence ev;
        ev.kind = EvidenceKind::kMemory;
        ev.source_id = "memory:" + m.key;
        ev.source_version = std::to_string(m.version);
        ev.text = m.value;
        ev.hash = hash128(ev.text);
        ev.id = "mem:" + m.key;
        out.push_back(std::move(ev));
    }
    return out;
}

} // namespace strata::rt
