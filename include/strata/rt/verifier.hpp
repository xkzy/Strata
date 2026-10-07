// include/strata/rt/verifier.hpp - deterministic verifiers, the dependency-aware cache and progressive verification
#pragma once

#include "strata/rt/claim.hpp"
#include "strata/rt/evidence.hpp"

#include <chrono>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

namespace strata::rt {

using Clock = std::chrono::steady_clock;

enum class Tier { kCheap = 0, kExact = 1, kSpecialized = 2, kExpensive = 3 };

struct VerifyOutcome {
    VerificationState state = VerificationState::kUnknown;
    std::string corrected_value;               // what the evidence / computation says, when it disagrees
    std::string explanation;
    std::vector<std::string> evidence_ids;
    std::string verifier;
    bool authoritative = false;                // a deterministic computation or authoritative exact state
    bool failed = false;                       // the verifier itself failed (timeout, crash): UNKNOWN, never "false"
    bool conflicting_evidence = false;
};

class IClaimVerifier {
public:
    virtual ~IClaimVerifier() = default;
    virtual std::string name() const = 0;
    virtual std::string version() const = 0;
    virtual std::string backend_version() const { return ""; }
    virtual Tier tier() const = 0;
    virtual bool available() const { return true; }
    virtual bool handles(const Claim& claim) const = 0;
    virtual VerifyOutcome verify(const Claim& claim, const EvidenceSet& evidence, Clock::time_point deadline) = 0;
};

// Checks claims about configuration values, numbers, dates, paths, identifiers and citations against evidence.
class ExactEvidenceVerifier : public IClaimVerifier {
public:
    std::string name() const override { return "exact_evidence"; }
    std::string version() const override { return "1"; }
    Tier tier() const override { return Tier::kExact; }
    bool handles(const Claim& c) const override;
    VerifyOutcome verify(const Claim& claim, const EvidenceSet& evidence, Clock::time_point deadline) override;
};

// Exact arithmetic on the native CAS (arbitrary precision, no overflow, no floating point unless the claim is decimal).
class ArithmeticVerifier : public IClaimVerifier {
public:
    std::string name() const override { return "arithmetic_cas"; }
    std::string version() const override { return "1"; }
    std::string backend_version() const override { return "cas-0.2.0"; }
    Tier tier() const override { return Tier::kSpecialized; }
    bool handles(const Claim& c) const override { return c.kind == ClaimKind::kArithmetic && c.has_formal; }
    VerifyOutcome verify(const Claim& claim, const EvidenceSet& evidence, Clock::time_point deadline) override;
};

// Contradictions between claims of one response ("port 8080 ... port 9090").
std::vector<std::pair<size_t, size_t>> find_inconsistent_claims(const std::vector<Claim>& claims);

// ---- dependency-aware verification cache ----
// key = hash(normalised claim, evidence hash, source versions, verifier + version, backend version, assumptions).
// Each entry remembers the evidence sources it depends on, so a changed source invalidates exactly its dependents.
struct CacheKeyParts {
    std::string normalized_claim;
    std::string evidence_hash;
    std::string source_versions;
    std::string verifier;
    std::string verifier_version;
    std::string backend_version;
    std::string assumptions;
    std::string scope;   // security scope of the evidence the answer depends on (empty for evidence-free verifiers)
    std::string key() const;
};

struct CacheStats {
    uint64_t hits = 0, misses = 0, puts = 0, invalidations = 0, evictions = 0;
    size_t size = 0;
};

class VerificationCache {
public:
    explicit VerificationCache(size_t max_entries = 20000) : max_entries_(max_entries) {}
    bool get(const std::string& key, VerifyOutcome& out);
    void put(const std::string& key, const VerifyOutcome& outcome, const std::vector<std::string>& source_ids);
    size_t invalidate_source(const std::string& source_id);          // only entries that depended on it
    size_t invalidate_verifier(const std::string& verifier_name);
    void clear();
    CacheStats stats() const;

private:
    struct Entry { std::string key; VerifyOutcome outcome; std::vector<std::string> sources; };
    mutable std::mutex mu_;
    size_t max_entries_;
    std::list<Entry> lru_;
    std::unordered_map<std::string, std::list<Entry>::iterator> table_;
    std::unordered_map<std::string, std::set<std::string>> by_source_;   // source -> keys
    mutable CacheStats stats_;
    void erase_locked(std::list<Entry>::iterator it);
};

// ---- progressive verification ----
struct ClaimVerdict {
    Claim claim;
    VerificationState state = VerificationState::kUnknown;
    std::string corrected_value;
    std::string explanation;
    std::vector<std::string> evidence_ids;
    std::string deciding_verifier;
    int tiers_run = 0;
    bool from_cache = false;
    bool verifier_failed = false;
    bool conflicting_evidence = false;
    double elapsed_ms = 0.0;
};

struct VerifierMetrics {
    uint64_t claims = 0, cache_hits = 0, verified = 0, contradicted = 0, unsupported = 0, unknown = 0, verifier_failures = 0;
    uint64_t tier_calls[4] = {0, 0, 0, 0};
};

struct ProgressiveConfig {
    double claim_deadline_ms = 100.0;
    bool stop_on_supported = true;     // SUPPORTED by exact evidence ends the search; no need for the expensive tier
};

// cheap check -> cache -> exact evidence -> specialized verifier -> expensive verifier; stops as soon as the
// answer is decisive. A failing verifier yields UNKNOWN for that tier, never CONTRADICTED.
class ProgressiveVerifier {
public:
    explicit ProgressiveVerifier(std::shared_ptr<VerificationCache> cache = nullptr, ProgressiveConfig cfg = ProgressiveConfig());
    void add_verifier(std::shared_ptr<IClaimVerifier> v);
    ClaimVerdict verify(const Claim& claim, const EvidenceSet& evidence, const std::string& assumptions = "",
                        const std::string& scope = "");
    VerifierMetrics metrics() const;
    VerificationCache& cache() { return *cache_; }

private:
    std::shared_ptr<VerificationCache> cache_;
    ProgressiveConfig cfg_;
    std::vector<std::shared_ptr<IClaimVerifier>> verifiers_;   // sorted by tier
    mutable std::mutex mu_;
    VerifierMetrics metrics_;
    static EvidenceSet relevant(const Claim& claim, const EvidenceSet& all);
};

} // namespace strata::rt
