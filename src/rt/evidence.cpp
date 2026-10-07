// src/rt/evidence.cpp - evidence sets, fact extraction and the stock providers
#include "strata/rt/evidence.hpp"

#include "strata/rt/claim.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <set>
#include <sstream>

namespace strata::rt {

std::string normalize_key(const std::string& key) {
    std::string r;
    bool last_space = true;
    for (unsigned char c : key) {
        if (std::isalnum(c)) { r.push_back(static_cast<char>(std::tolower(c))); last_space = false; }
        else if (!last_space) { r.push_back(' '); last_space = true; }
    }
    while (!r.empty() && r.back() == ' ') r.pop_back();
    return r;
}

namespace {

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

std::string unquote(std::string s) {
    s = trim(s);
    while (!s.empty() && (s.back() == ',' || s.back() == ';')) s.pop_back();
    s = trim(s);
    if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\'') || (s.front() == '`' && s.back() == '`')))
        s = s.substr(1, s.size() - 2);
    return trim(s);
}

bool is_number_text(const std::string& s) {
    if (s.empty()) return false;
    size_t i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
    bool digit = false;
    for (; i < s.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (std::isdigit(c)) digit = true;
        else if (c != ',' && c != '.' && c != '_') return false;
    }
    return digit;
}

std::string number_text(const std::string& s) {
    std::string r;
    for (char c : s) if (c != ',' && c != '_') r.push_back(c);
    return r;
}

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

} // namespace

bool values_equal(const std::string& a0, const std::string& b0) {
    std::string a = unquote(a0), b = unquote(b0);
    // split a trailing unit: "64 gb" / "30 seconds" / "50%"
    auto split = [](const std::string& s, std::string& num, std::string& unit) {
        size_t i = 0;
        if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
        while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == ',' || s[i] == '.' || s[i] == '_')) ++i;
        num = s.substr(0, i);
        size_t j = i;
        while (j < s.size() && std::isspace(static_cast<unsigned char>(s[j]))) ++j;
        unit = lower(s.substr(j));
        return is_number_text(num);
    };
    std::string an, au, bn, bu;
    if (split(a, an, au) && split(b, bn, bu)) {
        an = number_text(an); bn = number_text(bn);
        // units: "s"/"seconds", "ms"/"milliseconds" are not folded: different spellings of the same unit are rare
        // enough that treating a missing unit as a wildcard is the safer reading
        bool unit_ok = au == bu || au.empty() || bu.empty();
        if (!unit_ok) return false;
        if (an == bn) return true;
        char* e1 = nullptr; char* e2 = nullptr;
        double x = std::strtod(an.c_str(), &e1), y = std::strtod(bn.c_str(), &e2);
        if (*e1 || *e2) return false;
        return x == y;
    }
    return lower(trim(a)) == lower(trim(b));
}

bool subjects_match(const std::string& a0, const std::string& b0) {
    std::string a = normalize_key(a0), b = normalize_key(b0);
    if (a.empty() || b.empty()) return false;
    if (a == b) return true;
    auto suffix = [](const std::string& longer, const std::string& shorter) {
        return longer.size() > shorter.size() && longer.compare(longer.size() - shorter.size(), shorter.size(), shorter) == 0 &&
               longer[longer.size() - shorter.size() - 1] == ' ';
    };
    return suffix(a, b) || suffix(b, a);
}

std::vector<Fact> extract_facts(const Evidence& ev) {
    std::vector<Fact> facts;
    std::set<std::string> seen;
    auto add = [&](const std::string& key, const std::string& value) {
        std::string k = normalize_key(key);
        std::string v = unquote(value);
        if (k.empty() || v.empty() || v.size() > 200) return;
        if (!seen.insert(k + "\x1f" + v).second) return;
        Fact f;
        f.key = k;
        f.value = v;
        f.evidence_id = ev.id;
        f.authoritative = ev.authoritative;
        facts.push_back(std::move(f));
    };
    // line oriented: key: value | key = value | "key": value
    std::istringstream in(ev.text);
    std::string line;
    while (std::getline(in, line)) {
        std::string t = trim(line);
        if (t.empty() || t[0] == '#' || t[0] == '/' || t.rfind("//", 0) == 0) continue;
        size_t sep = std::string::npos;
        size_t colon = t.find(':'), eq = t.find('=');
        sep = std::min(colon, eq);
        if (sep == std::string::npos || sep == 0) continue;
        std::string key = unquote(t.substr(0, sep));
        std::string val = t.substr(sep + 1);
        if (!val.empty() && val[0] == '=') val = val.substr(1);   // ==
        if (key.find(' ') != std::string::npos && key.size() > 40) continue;   // prose with a colon
        if (key.size() > 60) continue;
        add(key, val);
    }
    // prose: reuse the claim detector on the evidence text ("The server port is 8080.")
    ClaimDetector det;
    RequestScope scope;
    for (const auto& c : det.scan(ev.text, 0, ev.text.size(), scope)) {
        if ((c.kind == ClaimKind::kConfigValue || c.kind == ClaimKind::kNumber || c.kind == ClaimKind::kSpecification || c.kind == ClaimKind::kDate) && !c.subject.empty())
            add(c.subject, c.value);
    }
    return facts;
}

std::string EvidenceSet::hash() const {
    std::vector<std::string> parts;
    for (const auto& e : items) parts.push_back(e.id + "@" + e.source_version);
    std::sort(parts.begin(), parts.end());
    return hash_parts(parts);
}

std::vector<std::string> EvidenceSet::source_ids() const {
    std::set<std::string> ids;
    for (const auto& e : items) ids.insert(e.source_id);
    return std::vector<std::string>(ids.begin(), ids.end());
}

// ---- ExactStateStore ----
void ExactStateStore::put(const context::SecurityScope& owner, context::SharingScope sharing, const std::string& source_id,
                          const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& e : entries_) {
        if (e.source_id == source_id && e.key == key && e.owner.to_string() == owner.to_string()) {
            if (e.value != value) { e.value = value; ++e.version; }
            e.sharing = sharing;
            return;
        }
    }
    Entry e;
    e.owner = owner; e.sharing = sharing; e.source_id = source_id; e.key = key; e.value = value;
    entries_.push_back(std::move(e));
}

bool ExactStateStore::erase(const context::SecurityScope& owner, const std::string& source_id, const std::string& key) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->source_id == source_id && it->key == key && it->owner.to_string() == owner.to_string()) { entries_.erase(it); return true; }
    }
    return false;
}

std::vector<Evidence> ExactStateStore::retrieve(const RequestScope& scope, const std::string& query, size_t top_k) {
    std::lock_guard<std::mutex> lock(mu_);
    std::set<std::string> qwords;
    {
        std::istringstream in(normalize_key(query));
        std::string w;
        while (in >> w) if (w.size() > 2) qwords.insert(w);
    }
    std::vector<std::pair<int, Evidence>> scored;
    for (const auto& e : entries_) {
        if (!scope.security.can_access(e.owner, e.sharing)) continue;
        int hits = 0;
        std::istringstream in(normalize_key(e.key + " " + e.source_id));
        std::string w;
        while (in >> w) if (qwords.count(w)) ++hits;
        if (hits == 0 && !qwords.empty()) continue;
        Evidence ev;
        ev.kind = EvidenceKind::kExactState;
        ev.source_id = e.source_id;
        ev.text = e.key + ": " + e.value;
        ev.hash = hash128(ev.text);
        // content addressed (plus the counter): two owners with the same counter but different values never collide
        ev.source_version = std::to_string(e.version) + ":" + ev.hash;
        ev.id = "state:" + e.source_id + ":" + e.key;
        ev.authoritative = true;
        ev.score = hits;
        scored.emplace_back(hits, std::move(ev));
    }
    std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<Evidence> out;
    for (auto& p : scored) { if (out.size() >= top_k) break; out.push_back(std::move(p.second)); }
    return out;
}

// ---- DocumentEvidenceProvider ----
std::vector<Evidence> DocumentEvidenceProvider::retrieve(const RequestScope& scope, const std::string& query, size_t top_k) {
    std::vector<Evidence> out;
    if (!mtc_) return out;
    for (const auto& r : mtc_->retrieve(query, scope.security, top_k)) {
        Evidence ev;
        ev.kind = EvidenceKind::kRag;
        ev.source_id = r.filename.empty() ? "item:" + std::to_string(r.item_id) : r.filename;
        ev.text = r.matched_content;
        ev.hash = hash128(ev.text);
        ev.source_version = ev.hash;   // content addressed: a changed document is a new version
        ev.id = "doc:" + ev.source_id + ":" + ev.hash.substr(0, 8);
        ev.score = r.relevance_score;
        out.push_back(std::move(ev));
    }
    return out;
}

// ---- RequestEvidenceProvider ----
void RequestEvidenceProvider::add(const std::string& request_key, Evidence ev) {
    std::lock_guard<std::mutex> lock(mu_);
    if (ev.hash.empty()) ev.hash = hash128(ev.text);
    if (ev.source_version.empty()) ev.source_version = ev.hash;
    by_request_[request_key].push_back(std::move(ev));
}

void RequestEvidenceProvider::drop(const std::string& request_key) {
    std::lock_guard<std::mutex> lock(mu_);
    by_request_.erase(request_key);
}

std::vector<Evidence> RequestEvidenceProvider::retrieve(const RequestScope& scope, const std::string&, size_t top_k) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = by_request_.find(scope.key());
    if (it == by_request_.end()) return {};
    std::vector<Evidence> out = it->second;
    if (out.size() > top_k) out.resize(top_k);
    return out;
}

} // namespace strata::rt
