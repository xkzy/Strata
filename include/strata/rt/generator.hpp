// include/strata/rt/generator.hpp - the narrow interfaces between the runtime and an inference engine
//
// The runtime talks to the model only through these. The real adapter (the engine's `--serve` loop) and the scripted
// test engine both implement them; nothing above this line knows which one is in use.
#pragma once

#include "strata/rt/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::rt {

struct GenToken {
    int32_t id = 0;
    std::string text;
};

// One running generation. Pull style: next() blocks until a token is ready or the generation ends.
class IGenSession {
public:
    virtual ~IGenSession() = default;
    virtual bool next(GenToken& out) = 0;          // false: the model ended (or the session was cancelled)
    virtual void set_temperature(double t) = 0;    // takes effect from the next token
    virtual void cancel() = 0;
};

class IGenerator {
public:
    virtual ~IGenerator() = default;
    virtual std::unique_ptr<IGenSession> start(const std::vector<int32_t>& prompt_ids, const Sampling& sampling,
                                               const std::string& request_key) = 0;
};

class ITokenizer {
public:
    virtual ~ITokenizer() = default;
    virtual std::vector<int32_t> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<int32_t>& ids) const = 0;
    virtual std::string id() const = 0;   // model + tokenizer identity: part of every KV-compatibility key
};

class IPromptTemplate {
public:
    virtual ~IPromptTemplate() = default;
    // Renders the conversation and opens the assistant turn. `assistant_prefix` is text the assistant has already
    // written in this turn (released output kept across a regeneration); generation continues after it.
    virtual std::string render(const std::vector<Message>& messages, const std::string& assistant_prefix) const = 0;
    virtual std::string version() const = 0;
};

// ChatML (<|im_start|>role ... <|im_end|>), the format of the Qwen family.
class ChatMLTemplate : public IPromptTemplate {
public:
    std::string render(const std::vector<Message>& messages, const std::string& assistant_prefix) const override;
    std::string version() const override { return "chatml-1"; }
};

// ---- KV cache correctness is independent of everything else ----
// A cached KV state may be reused for a prompt only if the cached token sequence is an exact prefix of the new
// one AND it was produced by the same model, tokenizer and template version. Same conversation does not imply
// same KV: retrieval, memory, compaction and regeneration all change the token sequence.
struct KvIdentity {
    std::string model_tokenizer_id;
    std::string template_version;
    bool operator==(const KvIdentity& o) const { return model_tokenizer_id == o.model_tokenizer_id && template_version == o.template_version; }
};

struct KvReusePlan {
    size_t reusable_prefix = 0;   // tokens whose KV can be kept
    size_t to_prefill = 0;        // tokens that must be read again
    bool identity_matches = false;
};

KvReusePlan plan_kv_reuse(const KvIdentity& cached_identity, const std::vector<int32_t>& cached_tokens,
                          const KvIdentity& new_identity, const std::vector<int32_t>& new_tokens);

} // namespace strata::rt
