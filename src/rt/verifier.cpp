// src/rt/verifier.cpp - verifiers, dependency-aware cache and the progressive pipeline
#include "strata/rt/verifier.hpp"

#include "strata/math/cas/engine.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace strata::rt {

namespace {

bool is_word_char(unsigned char c) { return std::isalnum(c) || c == '_'; }

// literal appears in text, bounded by non-word characters (so "80" does not match inside "8080")
bool contains_bounded(const std::string& text, const std::string& lit) {
    if (lit.empty()) return false;
    size_t pos = 0;
    while ((pos = text.find(lit, pos)) != std::string::npos) {
        bool left_ok = pos == 0 || !is_word_char(static_cast<unsigned char>(text[pos - 1])) || !is_word_char(static_cast<unsigned char>(lit.front()));
        size_t end = pos + lit.size();
        bool right_ok = end >= text.size() || !is_word_char(static_cast<unsigned char>(text[end])) || !is_word_char(static_cast<unsigned char>(lit.back()));
        if (left_ok && right_ok) return true;
        ++pos;
    }
    return false;
}

std::string decimal_string(const strata::math::cas::Rational& q, int digits) {
    using strata::math::cas::BigInt;
    if (q.is_integer()) return q.num.to_string();
    BigInt scale = BigInt::pow(BigInt(10), static_cast<uint64_t>(digits));
    BigInt scaled = (q.num.abs() * scale) / q.den;     // truncated
    std::string s = scaled.to_string();
    while (s.size() <= static_cast<size_t>(digits)) s = "0" + s;
    std::string ip = s.substr(0, s.size() - digits), fp = s.substr(s.size() - digits);
    while (!fp.empty() && fp.back() == '0') fp.pop_back();
    return std::string(q.is_negative() ? "-" : "") + ip + (fp.empty() ? "" : "." + fp);
}

bool parse_rational(const std::string& text, strata::math::cas::Rational& out) {
    using namespace strata::math::cas;
    std::string s;
    for (char c : text) if (c != ',' && c != '_' && !std::isspace(static_cast<unsigned char>(c))) s.push_back(c);
    if (s.empty()) return false;
    bool neg = false;
    size_t i = 0;
    if (s[0] == '-') { neg = true; i = 1; } else if (s[0] == '+') i = 1;
    std::string ip, fp;
    bool seen_point = false;
    for (; i < s.size(); ++i) {
        if (std::isdigit(static_cast<unsigned char>(s[i]))) (seen_point ? fp : ip).push_back(s[i]);
        else if (s[i] == '.' && !seen_point) seen_point = true;
        else return false;
    }
    if (ip.empty() && fp.empty()) return false;
    BigInt n;
    if (!BigInt::from_string(ip + fp, n)) return false;
    BigInt d = BigInt::pow(BigInt(10), static_cast<uint64_t>(fp.size()));
    if (neg) n = -n;
    out = Rational(n, d);
    return true;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------
// ExactEvidenceVerifier
// ---------------------------------------------------------------------------------------------------------------
bool ExactEvidenceVerifier::handles(const Claim& c) const {
    switch (c.kind) {
        case ClaimKind::kConfigValue: case ClaimKind::kNumber: case ClaimKind::kSpecification: case ClaimKind::kDate:
        case ClaimKind::kFilePath: case ClaimKind::kIdentifier: case ClaimKind::kApiSignature: case ClaimKind::kCitation:
            return true;
        default:
            return false;
    }
}

VerifyOutcome ExactEvidenceVerifier::verify(const Claim& claim, const EvidenceSet& evidence, Clock::time_point) {
    VerifyOutcome out;
    out.verifier = name();
    if (evidence.empty()) {
        out.state = VerificationState::kUnknown;
        out.explanation = "no evidence available";
        return out;
    }
    auto text_hit = [&](const std::string& lit, std::vector<std::string>& ids, bool& authoritative) {
        bool hit = false;
        for (const auto& ev : evidence.items) {
            if (contains_bounded(ev.text, lit)) { hit = true; ids.push_back(ev.id); authoritative = authoritative || ev.authoritative; }
        }
        return hit;
    };

    switch (claim.kind) {
        case ClaimKind::kConfigValue: case ClaimKind::kNumber: case ClaimKind::kSpecification: {
            std::vector<Fact> matching;
            for (const auto& ev : evidence.items)
                for (const auto& f : extract_facts(ev))
                    if (subjects_match(claim.subject, f.key)) matching.push_back(f);
            if (!matching.empty()) {
                const Fact* agree = nullptr;
                for (const auto& f : matching) if (values_equal(f.value, claim.value)) { if (!agree || f.authoritative) agree = &f; }
                // conflicting evidence: authoritative facts that disagree with each other
                bool conflict = false;
                for (size_t i = 0; i < matching.size() && !conflict; ++i)
                    for (size_t j = i + 1; j < matching.size(); ++j)
                        if (matching[i].authoritative && matching[j].authoritative && !values_equal(matching[i].value, matching[j].value)) { conflict = true; break; }
                if (conflict) {
                    out.state = VerificationState::kUnknown;
                    out.conflicting_evidence = true;
                    out.explanation = "authoritative evidence disagrees with itself about '" + claim.subject + "'";
                    for (const auto& f : matching) out.evidence_ids.push_back(f.evidence_id);
                    return out;
                }
                if (agree) {
                    out.state = agree->authoritative ? VerificationState::kVerified : VerificationState::kSupported;
                    out.authoritative = agree->authoritative;
                    out.evidence_ids.push_back(agree->evidence_id);
                    out.explanation = "evidence states " + claim.subject + " = " + agree->value;
                    return out;
                }
                // a different value: prefer the authoritative fact
                const Fact* best = &matching.front();
                for (const auto& f : matching) if (f.authoritative && !best->authoritative) best = &f;
                out.state = VerificationState::kContradicted;
                out.authoritative = best->authoritative;
                out.corrected_value = best->value;
                out.evidence_ids.push_back(best->evidence_id);
                out.explanation = "evidence states " + claim.subject + " = " + best->value + ", not " + claim.value;
                return out;
            }
            // no fact about this subject: does the evidence at least mention the value?
            std::vector<std::string> ids;
            bool auth = false;
            if (text_hit(claim.value, ids, auth)) {
                out.state = VerificationState::kSupported;
                out.evidence_ids = ids;
                out.explanation = "value appears in the evidence";
                return out;
            }
            out.state = VerificationState::kUnsupported;
            out.explanation = "evidence does not mention " + claim.subject + " = " + claim.value;
            return out;
        }
        case ClaimKind::kDate: case ClaimKind::kFilePath: case ClaimKind::kIdentifier: case ClaimKind::kCitation: {
            std::vector<std::string> ids;
            bool auth = false;
            if (text_hit(claim.value, ids, auth)) {
                out.state = VerificationState::kSupported;
                out.evidence_ids = ids;
                out.explanation = "appears in the evidence";
                return out;
            }
            out.state = VerificationState::kUnsupported;
            out.explanation = "not found in the evidence";
            return out;
        }
        case ClaimKind::kApiSignature: {
            // exact signature present -> supported; same function with other parameters -> contradicted
            std::vector<std::string> ids;
            bool auth = false;
            if (text_hit(claim.value, ids, auth)) {
                out.state = VerificationState::kSupported;
                out.evidence_ids = ids;
                out.explanation = "signature appears in the evidence";
                return out;
            }
            const std::string head = claim.subject + "(";
            for (const auto& ev : evidence.items) {
                size_t pos = ev.text.find(head);
                while (pos != std::string::npos) {
                    bool left_ok = pos == 0 || !is_word_char(static_cast<unsigned char>(ev.text[pos - 1]));
                    size_t close = ev.text.find(')', pos);
                    if (left_ok && close != std::string::npos) {
                        out.state = VerificationState::kContradicted;
                        out.corrected_value = ev.text.substr(pos, close - pos + 1);
                        out.evidence_ids.push_back(ev.id);
                        out.authoritative = ev.authoritative;
                        out.explanation = "evidence gives " + out.corrected_value + ", not " + claim.value;
                        return out;
                    }
                    pos = ev.text.find(head, pos + 1);
                }
            }
            out.state = VerificationState::kUnsupported;
            out.explanation = "function not found in the evidence";
            return out;
        }
        default:
            out.state = VerificationState::kUnknown;
            return out;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// ArithmeticVerifier
// ---------------------------------------------------------------------------------------------------------------
VerifyOutcome ArithmeticVerifier::verify(const Claim& claim, const EvidenceSet&, Clock::time_point deadline) {
    using namespace strata::math::cas;
    VerifyOutcome out;
    out.verifier = name();
    out.authoritative = true;
    try {
        Budget budget;
        double remaining = std::chrono::duration<double, std::milli>(deadline - Clock::now()).count();
        budget.timeout_ms = std::max(5.0, std::min(budget.timeout_ms, remaining));
        Engine engine(budget);
        Expr result = engine.eval(parse(claim.formal.expression));
        if (result->is_symbol_named("True") || result->is_symbol_named("False")) {   // a yes/no property (primality, parity)
            const std::string truth = result->name;
            if (truth == claim.formal.claimed_value) {
                out.state = VerificationState::kVerified;
                out.explanation = claim.subject + " (exact)";
            } else {
                out.state = VerificationState::kContradicted;
                out.explanation = "the statement \"" + claim.subject + "\" is false (computed exactly)";
            }
            return out;
        }
        if (!result->is_number()) {
            out.state = VerificationState::kUnknown;
            out.explanation = "expression did not reduce to a number";
            return out;
        }
        Rational claimed;
        if (!parse_rational(claim.formal.claimed_value, claimed)) {
            out.state = VerificationState::kUnknown;
            out.explanation = "claimed value is not a number";
            return out;
        }
        const Rational& truth = result->q;
        if (truth == claimed) {
            out.state = VerificationState::kVerified;
            out.explanation = claim.subject + " = " + decimal_string(truth, 12) + " (exact)";
            return out;
        }
        // a claimed decimal with d digits is a statement about the rounded value (10/3 = 3.33)
        const std::string& cv = claim.formal.claimed_value;
        size_t dot = cv.find('.');
        if (dot != std::string::npos) {
            const size_t d = cv.size() - dot - 1;
            BigInt scale = BigInt::pow(BigInt(10), d);
            // round half away from zero
            BigInt scaled = truth.num.abs() * scale * BigInt(2) + truth.den;
            BigInt rounded = scaled / (truth.den * BigInt(2));
            if (truth.is_negative()) rounded = -rounded;
            if (Rational(rounded, scale) == claimed) {
                out.state = VerificationState::kSupported;
                out.explanation = claim.subject + " = " + decimal_string(truth, static_cast<int>(d) + 6) + ", rounds to the claimed value";
                return out;
            }
        }
        out.state = VerificationState::kContradicted;
        out.corrected_value = decimal_string(truth, 10);
        out.explanation = claim.subject + " = " + out.corrected_value + (truth.is_integer() ? "" : " (exactly " + truth.to_string() + ")") + ", not " + claim.formal.claimed_value;
        return out;
    } catch (const CasLimitError& e) {
        out.state = VerificationState::kUnknown;
        out.failed = true;
        out.explanation = std::string("resource limit: ") + e.what();
    } catch (const CasError& e) {
        out.state = VerificationState::kUnknown;
        out.explanation = e.what();
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Consistency inside one response
// ---------------------------------------------------------------------------------------------------------------
std::vector<std::pair<size_t, size_t>> find_inconsistent_claims(const std::vector<Claim>& claims) {
    std::vector<std::pair<size_t, size_t>> out;
    for (size_t i = 0; i < claims.size(); ++i) {
        for (size_t j = i + 1; j < claims.size(); ++j) {
            const Claim &a = claims[i], &b = claims[j];
            bool kv = (a.kind == ClaimKind::kConfigValue || a.kind == ClaimKind::kSpecification || a.kind == ClaimKind::kNumber) &&
                      (b.kind == ClaimKind::kConfigValue || b.kind == ClaimKind::kSpecification || b.kind == ClaimKind::kNumber);
            if (kv && subjects_match(a.subject, b.subject) && !values_equal(a.value, b.value)) out.emplace_back(i, j);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// dependency-aware cache
// ---------------------------------------------------------------------------------------------------------------
std::string CacheKeyParts::key() const {
    return hash_parts({normalized_claim, evidence_hash, source_versions, verifier, verifier_version, backend_version, assumptions, scope});
}

void VerificationCache::erase_locked(std::list<Entry>::iterator it) {
    for (const auto& s : it->sources) {
        auto b = by_source_.find(s);
        if (b != by_source_.end()) { b->second.erase(it->key); if (b->second.empty()) by_source_.erase(b); }
    }
    table_.erase(it->key);
    lru_.erase(it);
}

bool VerificationCache::get(const std::string& key, VerifyOutcome& out) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = table_.find(key);
    if (it == table_.end()) { ++stats_.misses; return false; }
    lru_.splice(lru_.begin(), lru_, it->second);
    out = it->second->outcome;
    ++stats_.hits;
    return true;
}

void VerificationCache::put(const std::string& key, const VerifyOutcome& outcome, const std::vector<std::string>& source_ids) {
    if (outcome.failed) return;   // a failure is transient: never remember it as an answer
    std::lock_guard<std::mutex> lock(mu_);
    auto it = table_.find(key);
    if (it != table_.end()) erase_locked(it->second);
    lru_.push_front(Entry{key, outcome, source_ids});
    table_[key] = lru_.begin();
    for (const auto& s : source_ids) by_source_[s].insert(key);
    ++stats_.puts;
    while (lru_.size() > max_entries_) { erase_locked(std::prev(lru_.end())); ++stats_.evictions; }
}

size_t VerificationCache::invalidate_source(const std::string& source_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto b = by_source_.find(source_id);
    if (b == by_source_.end()) return 0;
    std::vector<std::string> keys(b->second.begin(), b->second.end());
    size_t n = 0;
    for (const auto& k : keys) {
        auto it = table_.find(k);
        if (it != table_.end()) { erase_locked(it->second); ++n; }
    }
    stats_.invalidations += n;
    return n;
}

size_t VerificationCache::invalidate_verifier(const std::string& verifier_name) {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::list<Entry>::iterator> victims;
    for (auto it = lru_.begin(); it != lru_.end(); ++it)
        if (it->outcome.verifier == verifier_name) victims.push_back(it);
    for (auto it : victims) erase_locked(it);
    stats_.invalidations += victims.size();
    return victims.size();
}

void VerificationCache::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    lru_.clear(); table_.clear(); by_source_.clear();
}

CacheStats VerificationCache::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    CacheStats s = stats_;
    s.size = lru_.size();
    return s;
}

// ---------------------------------------------------------------------------------------------------------------
// progressive verification
// ---------------------------------------------------------------------------------------------------------------
ProgressiveVerifier::ProgressiveVerifier(std::shared_ptr<VerificationCache> cache, ProgressiveConfig cfg)
    : cache_(cache ? std::move(cache) : std::make_shared<VerificationCache>()), cfg_(cfg) {}

void ProgressiveVerifier::add_verifier(std::shared_ptr<IClaimVerifier> v) {
    std::lock_guard<std::mutex> lock(mu_);
    verifiers_.push_back(std::move(v));
    std::stable_sort(verifiers_.begin(), verifiers_.end(), [](const auto& a, const auto& b) { return a->tier() < b->tier(); });
}

VerifierMetrics ProgressiveVerifier::metrics() const {
    std::lock_guard<std::mutex> lock(mu_);
    return metrics_;
}

EvidenceSet ProgressiveVerifier::relevant(const Claim&, const EvidenceSet& all) { return all; }

ClaimVerdict ProgressiveVerifier::verify(const Claim& claim, const EvidenceSet& evidence_all, const std::string& assumptions,
                                       const std::string& scope) {
    const auto t0 = Clock::now();
    const auto deadline = t0 + std::chrono::microseconds(static_cast<int64_t>(cfg_.claim_deadline_ms * 1000.0));
    ClaimVerdict v;
    v.claim = claim;
    EvidenceSet evidence = relevant(claim, evidence_all);
    std::vector<std::shared_ptr<IClaimVerifier>> chain;
    {
        std::lock_guard<std::mutex> lock(mu_);
        chain = verifiers_;
        ++metrics_.claims;
    }

    bool decided = false;
    VerifyOutcome best;
    best.state = VerificationState::kUnknown;
    bool any_failed = false, any_conflict = false;
    bool first_cache_hit = false;

    for (const auto& ver : chain) {
        if (!ver->available() || !ver->handles(claim)) continue;
        // cache: keyed by everything the answer depends on
        CacheKeyParts kp;
        kp.normalized_claim = claim.normalized;
        const bool uses_evidence = ver->tier() <= Tier::kExact;
        kp.evidence_hash = uses_evidence ? evidence.hash() : "";
        kp.source_versions = "";
        for (const auto& e : evidence.items) if (uses_evidence) kp.source_versions += e.source_id + "@" + e.source_version + ";";
        kp.verifier = ver->name();
        kp.verifier_version = ver->version();
        kp.backend_version = ver->backend_version();
        kp.assumptions = assumptions;
        kp.scope = uses_evidence ? scope : "";
        const std::string key = kp.key();

        VerifyOutcome oc;
        bool cached = cache_->get(key, oc);
        if (cached) {
            std::lock_guard<std::mutex> lock(mu_);
            ++metrics_.cache_hits;
        } else {
            if (Clock::now() >= deadline) { any_failed = true; continue; }
            {
                std::lock_guard<std::mutex> lock(mu_);
                ++metrics_.tier_calls[static_cast<int>(ver->tier())];
            }
            try {
                oc = ver->verify(claim, evidence, deadline);
            } catch (const std::exception& e) {
                oc = VerifyOutcome();
                oc.state = VerificationState::kUnknown;
                oc.failed = true;
                oc.verifier = ver->name();
                oc.explanation = std::string("verifier error: ") + e.what();
            } catch (...) {
                oc = VerifyOutcome();
                oc.state = VerificationState::kUnknown;
                oc.failed = true;
                oc.verifier = ver->name();
                oc.explanation = "verifier error";
            }
            if (oc.verifier.empty()) oc.verifier = ver->name();
            std::vector<std::string> deps = uses_evidence ? evidence.source_ids() : std::vector<std::string>{};
            cache_->put(key, oc, deps);
        }
        ++v.tiers_run;
        if (cached && v.tiers_run == 1) first_cache_hit = true;
        if (oc.failed) any_failed = true;
        if (oc.conflicting_evidence) any_conflict = true;

        // keep the strongest answer seen so far
        auto rank = [](VerificationState s) {
            switch (s) {
                case VerificationState::kVerified: return 5;
                case VerificationState::kContradicted: return 5;
                case VerificationState::kSupported: return 4;
                case VerificationState::kPartiallySupported: return 3;
                case VerificationState::kUnsupported: return 2;
                case VerificationState::kUnknown: return 1;
            }
            return 0;
        };
        // a deterministic VERIFIED and a CONTRADICTED from another verifier cannot both be right: UNKNOWN + conflict
        if ((best.state == VerificationState::kVerified && oc.state == VerificationState::kContradicted && oc.authoritative && best.authoritative) ||
            (best.state == VerificationState::kContradicted && oc.state == VerificationState::kVerified && oc.authoritative && best.authoritative)) {
            best = VerifyOutcome();
            best.state = VerificationState::kUnknown;
            best.conflicting_evidence = true;
            best.explanation = "verifiers disagree";
            any_conflict = true;
            decided = true;
            break;
        }
        if (rank(oc.state) > rank(best.state) || (rank(oc.state) == rank(best.state) && oc.authoritative && !best.authoritative)) best = oc;
        if (oc.state == VerificationState::kVerified || oc.state == VerificationState::kContradicted) { decided = true; break; }
        if (cfg_.stop_on_supported && oc.state == VerificationState::kSupported) { decided = true; break; }
    }
    (void)decided;

    v.state = best.state;
    v.corrected_value = best.corrected_value;
    v.explanation = best.explanation;
    v.evidence_ids = best.evidence_ids;
    v.deciding_verifier = best.verifier;
    v.from_cache = first_cache_hit && v.tiers_run == 1;
    v.verifier_failed = any_failed && best.state == VerificationState::kUnknown;
    v.conflicting_evidence = any_conflict;
    v.elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    {
        std::lock_guard<std::mutex> lock(mu_);
        switch (v.state) {
            case VerificationState::kVerified: case VerificationState::kSupported: case VerificationState::kPartiallySupported: ++metrics_.verified; break;
            case VerificationState::kContradicted: ++metrics_.contradicted; break;
            case VerificationState::kUnsupported: ++metrics_.unsupported; break;
            case VerificationState::kUnknown: ++metrics_.unknown; break;
        }
        if (v.verifier_failed) ++metrics_.verifier_failures;
    }
    return v;
}

} // namespace strata::rt
