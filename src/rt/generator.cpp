// src/rt/generator.cpp - ChatML template and the exact-prefix KV reuse rule
#include "strata/rt/generator.hpp"

namespace strata::rt {

std::string ChatMLTemplate::render(const std::vector<Message>& messages, const std::string& assistant_prefix) const {
    std::string out;
    for (const auto& m : messages) {
        const std::string role = m.role == "tool" ? "user" : m.role;   // tool output is shown to the model as a user-side observation
        out += "<|im_start|>" + role + "\n";
        if (m.role == "tool") out += "[tool result" + (m.name.empty() ? "" : ": " + m.name) + "]\n";
        out += m.content + "<|im_end|>\n";
    }
    out += "<|im_start|>assistant\n" + assistant_prefix;
    return out;
}

KvReusePlan plan_kv_reuse(const KvIdentity& cached_identity, const std::vector<int32_t>& cached_tokens,
                          const KvIdentity& new_identity, const std::vector<int32_t>& new_tokens) {
    KvReusePlan plan;
    plan.identity_matches = cached_identity == new_identity;
    plan.to_prefill = new_tokens.size();
    if (!plan.identity_matches) return plan;
    size_t n = 0;
    const size_t lim = std::min(cached_tokens.size(), new_tokens.size());
    while (n < lim && cached_tokens[n] == new_tokens[n]) ++n;
    // The last token of a prompt must always be read fresh (it produces the first output logits).
    if (n >= new_tokens.size() && n > 0) n = new_tokens.size() - 1;
    plan.reusable_prefix = n;
    plan.to_prefill = new_tokens.size() - n;
    return plan;
}

} // namespace strata::rt
