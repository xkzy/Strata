// include/strata/rt/context_manager.hpp - virtual context: decides what the model sees
//
// Physical context (what is in the prompt) is a small window onto a much larger virtual context (history, memory,
// indexed documents, tool state). Raw documents stay outside; only relevant snippets are materialised.
#pragma once

#include "strata/rt/evidence.hpp"
#include "strata/rt/memory.hpp"

#include <memory>
#include <vector>

namespace strata::rt {

struct ContextConfig {
    size_t max_prompt_tokens = 6000;     // budget for the whole prompt (chars / 4)
    size_t snippet_chars = 500;          // per retrieved snippet
    size_t rag_top_k = 4;
    bool rag_enabled = true;
    bool memory_enabled = true;
};

struct PreparedContext {
    std::vector<Message> messages;       // what the template renders
    EvidenceSet evidence;                // everything retrieved for this request (also used to verify the answer)
    bool retrieval_used = false;
    size_t memory_items_used = 0;
    size_t dropped_messages = 0;
    size_t estimated_tokens = 0;
};

class ContextManager {
public:
    ContextManager(ContextConfig cfg, std::vector<std::shared_ptr<IEvidenceProvider>> providers, std::shared_ptr<MemoryStore> memory);

    // Heuristic: does answering need external or project knowledge (configuration, files, exact state)?
    static bool needs_retrieval(const std::string& user_text);

    PreparedContext prepare(const RequestScope& scope, const std::vector<Message>& conversation);
    // Retrieval on demand (used when a high-risk claim has no support): same providers, query = the claim.
    EvidenceSet retrieve(const RequestScope& scope, const std::string& query, size_t top_k) const;

    const ContextConfig& config() const { return cfg_; }

private:
    ContextConfig cfg_;
    std::vector<std::shared_ptr<IEvidenceProvider>> providers_;
    std::shared_ptr<MemoryStore> memory_;
};

} // namespace strata::rt
