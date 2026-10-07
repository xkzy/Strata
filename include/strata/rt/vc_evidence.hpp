// include/strata/rt/vc_evidence.hpp - the virtual context as an evidence space for verification
#pragma once

#include "strata/rt/evidence.hpp"
#include "strata/rt/virtual_context.hpp"

namespace strata::rt {

// Lets the verifier and the recovery logic look things up in the session's logical context (earlier turns, files, tool
// results) without any retrieval operation being visible to the agent. Provenance is kept apart: "ctx:" ids come from
// earlier session state, "ext:" ids from files/documents/knowledge.
class VirtualContextEvidenceProvider : public IEvidenceProvider {
public:
    explicit VirtualContextEvidenceProvider(std::shared_ptr<VirtualContextStore> store) : store_(std::move(store)) {}
    std::string name() const override { return "virtual_context"; }
    std::vector<Evidence> retrieve(const RequestScope& scope, const std::string& query, size_t top_k) override {
        std::vector<Evidence> out;
        if (!store_) return out;
        auto vc = store_->open(scope.security);
        for (const auto& h : vc->search(scope.security, query, top_k)) {
            Evidence ev;
            const char* tag = h.provenance == Provenance::kInternalContext ? "ctx:" : "ext:";
            ev.kind = EvidenceKind::kRag;
            ev.source_id = std::string(tag) + h.source;
            ev.text = h.text;
            ev.hash = h.content_hash;
            ev.source_version = h.content_hash;     // content addressed
            ev.id = "vc:" + std::to_string(h.page_id) + ":" + h.content_hash.substr(0, 8);
            ev.score = h.score;
            out.push_back(std::move(ev));
        }
        return out;
    }
private:
    std::shared_ptr<VirtualContextStore> store_;
};

} // namespace strata::rt
