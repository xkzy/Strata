// include/strata/rt/memory.hpp - persistent memory: scoped, permission-aware, versioned, independently invalidatable
#pragma once

#include "strata/rt/evidence.hpp"

#include <mutex>
#include <string>
#include <vector>

namespace strata::rt {

struct MemoryEntry {
    std::string key;
    std::string value;
    context::SecurityScope owner;
    context::SharingScope sharing = context::SharingScope::PRIVATE;
    uint64_t version = 1;
    bool valid = true;     // invalidated entries are kept (for audit) but never returned
};

// The caller never touches this: the runtime reads it while assembling context and may write to it after a turn
// when configured. It is also an evidence provider, so remembered facts can back (or contradict) a claim.
class MemoryStore : public IEvidenceProvider {
public:
    uint64_t put(const context::SecurityScope& owner, context::SharingScope sharing, const std::string& key, const std::string& value);
    bool invalidate(const context::SecurityScope& owner, const std::string& key);
    std::vector<MemoryEntry> search(const context::SecurityScope& requester, const std::string& query, size_t top_k) const;
    // "remember that X" / "my name is Y": extracts durable statements from a user turn and stores them privately.
    size_t observe_turn(const context::SecurityScope& owner, const std::string& user_text);
    size_t size() const;

    std::string name() const override { return "memory"; }
    std::vector<Evidence> retrieve(const RequestScope& scope, const std::string& query, size_t top_k) override;

private:
    mutable std::mutex mu_;
    std::vector<MemoryEntry> entries_;
};

} // namespace strata::rt
