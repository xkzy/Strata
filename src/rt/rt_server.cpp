// src/rt/rt_server.cpp - strata_rt_server: the transparent runtime as a sidecar process
//
// The Go server owns the HTTP API, the tokenizer, the chat template and the engine pipe. This process owns the runtime:
// the virtual context window, retrieval and memory, claim detection and verification (CAS), loop protection and
// recovery. They talk JSON lines over stdin/stdout:
//
//   Go -> rt   config | request | cancel | metrics | quit, and the answers to the rt's own calls:
//              prompt_result / prompt_error (a prompt built from messages), gen_token / gen_end / gen_error
//   rt -> Go   ready | emit (text released to the caller) | done | error | metrics_result, and its calls:
//              prompt (render + tokenize these messages) and gen / gen_cancel (run the model on these ids)
//
// Nothing here is visible to the API client: it sends an ordinary chat request and receives ordinary text.
#include "strata/rt/json.hpp"
#include "strata/rt/runtime.hpp"

#include <atomic>
#include <cstdio>
#include <deque>
#include <iostream>
#include <stdexcept>

using namespace strata::rt;

namespace {

struct Gone : std::runtime_error { Gone() : std::runtime_error("the host closed the connection") {} };
struct HostError : std::runtime_error {
    int status;
    std::string type;
    HostError(int s, std::string t, const std::string& m) : std::runtime_error(m), status(s), type(std::move(t)) {}
};

class Host {
public:
    void send(const Json& j) {
        const std::string line = j.dump() + "\n";
        std::fwrite(line.data(), 1, line.size(), stdout);
        std::fflush(stdout);
    }

    // Next message from the host. Messages that are not for the caller are handled here or queued.
    Json read() {
        while (true) {
            if (!pending_.empty()) { Json m = std::move(pending_.front()); pending_.pop_front(); return m; }
            Json m = read_raw();
            if (!route(m)) return m;
        }
    }
    // Reads until `accept(m)` is true; everything else is routed (cancel, metrics) or queued (requests).
    template <class F> Json wait(F accept) {
        while (true) {
            Json m = read_raw();
            if (accept(m)) return m;
            route(m);
        }
    }

    bool cancelled() const { return cancel_.load(); }
    void begin_request(int64_t id) { current_ = id; cancel_ = false; }
    void set_metrics(std::function<Json()> f) { metrics_ = std::move(f); }
    bool has_pending() const { return !pending_.empty(); }
    Json take_pending() { Json m = std::move(pending_.front()); pending_.pop_front(); return m; }
    int64_t next_id() { return ++counter_; }

private:
    std::deque<Json> pending_;
    std::atomic<bool> cancel_{false};
    int64_t current_ = -1, counter_ = 0;
    std::function<Json()> metrics_;

    Json read_raw() {
        std::string line;
        while (true) {
            if (!std::getline(std::cin, line)) throw Gone();
            if (line.empty()) continue;
            Json m;
            std::string err;
            if (Json::parse(line, m, &err)) return m;
            std::fprintf(stderr, "strata_rt_server: bad message (%s)\n", err.c_str());
        }
    }
    // true when the message was consumed here
    bool route(Json& m) {
        const std::string& op = m["op"].str();
        if (op == "cancel") { if (m["id"].i64(-2) == current_) cancel_ = true; return true; }
        if (op == "metrics") {
            Json r = Json::object();
            r.set("op", Json::string("metrics_result")).set("rid", m["rid"]).set("metrics", metrics_ ? metrics_() : Json::object());
            send(r);
            return true;
        }
        if (op == "request" || op == "config" || op == "quit") { pending_.push_back(std::move(m)); return true; }
        return false;
    }
};

Message message_from(const Json& j) {
    Message m;
    m.role = j["role"].str();
    m.content = j["content"].str();
    m.name = j["name"].str();
    return m;
}

// The chat template and the tokenizer live in the host: it renders and tokenizes the messages and returns the ids.
class HostPrompt : public IPromptBuilder {
public:
    HostPrompt(Host& h, std::string identity) : host_(h), identity_(std::move(identity)) {}
    void set_request(int64_t id, bool reasoning_prefix) { id_ = id; reasoning_prefix_ = reasoning_prefix; }
    std::vector<int32_t> build(const std::vector<Message>& messages, const std::string& prefix) override {
        const int64_t rid = host_.next_id();
        Json req = Json::object();
        req.set("op", Json::string("prompt")).set("rid", Json::integer(rid)).set("id", Json::integer(id_));
        Json arr = Json::array();
        for (const auto& m : messages) {
            Json o = Json::object();
            o.set("role", Json::string(m.role)).set("content", Json::string(m.content));
            if (!m.name.empty()) o.set("name", Json::string(m.name));
            arr.push(std::move(o));
        }
        req.set("messages", std::move(arr)).set("prefix", Json::string(prefix));
        host_.send(req);
        Json r = host_.wait([&](const Json& m) {
            const std::string& op = m["op"].str();
            return (op == "prompt_result" || op == "prompt_error") && m["rid"].i64(-1) == rid;
        });
        if (r["op"].str() == "prompt_error")
            throw HostError(static_cast<int>(r["status"].i64(500)), r["type"].str("server_error"), r["message"].str());
        std::vector<int32_t> ids;
        ids.reserve(r["ids"].size());
        for (const auto& v : r["ids"].items()) ids.push_back(static_cast<int32_t>(v.i64()));
        return ids;
    }
    std::string identity() const override { return identity_; }
private:
    Host& host_;
    std::string identity_;
    int64_t id_ = 0;
    bool reasoning_prefix_ = false;
};

class HostSession : public IGenSession {
public:
    HostSession(Host& h, int64_t gid) : host_(h), gid_(gid) {}
    ~HostSession() override { finish(); }
    bool next(GenToken& out) override {
        while (!ended_) {
            if (host_.cancelled() && !cancel_sent_) cancel();
            Json m = host_.wait([&](const Json& x) {
                const std::string& op = x["op"].str();
                return (op == "gen_token" || op == "gen_end" || op == "gen_error") && x["gid"].i64(-1) == gid_;
            });
            const std::string& op = m["op"].str();
            if (op == "gen_token") {
                if (cancel_sent_) continue;   // draining after a cancel
                out.id = static_cast<int32_t>(m["id"].i64());
                out.text = m["text"].str();
                return true;
            }
            ended_ = true;
            if (op == "gen_error") throw HostError(502, "engine_error", m["message"].str());
            finish_reason_ = m["finish"].str();
        }
        return false;
    }
    void set_temperature(double) override {}   // the engine takes the temperature per request: a change applies to the next attempt
    void cancel() override {
        if (ended_ || cancel_sent_) return;
        cancel_sent_ = true;
        Json j = Json::object();
        j.set("op", Json::string("gen_cancel")).set("gid", Json::integer(gid_));
        host_.send(j);
    }
private:
    Host& host_;
    int64_t gid_;
    bool ended_ = false, cancel_sent_ = false;
    std::string finish_reason_;
    // leave the engine idle: cancel and read until its end
    void finish() {
        try {
            if (!ended_) cancel();
            GenToken t;
            while (!ended_) { if (!next(t)) break; }
        } catch (...) {}
    }
};

class HostGenerator : public IGenerator {
public:
    explicit HostGenerator(Host& h) : host_(h) {}
    void set_request(int64_t id, const Json& extra) { id_ = id; extra_ = extra; }
    std::unique_ptr<IGenSession> start(const std::vector<int32_t>& ids, const Sampling& s, const std::string&) override {
        const int64_t gid = host_.next_id();
        Json j = Json::object();
        j.set("op", Json::string("gen")).set("gid", Json::integer(gid)).set("id", Json::integer(id_));
        Json arr = Json::array();
        for (int32_t v : ids) arr.push(Json::integer(v));
        Json samp = extra_;
        if (!samp.is_object()) samp = Json::object();
        samp.set("temperature", Json::number(s.temperature)).set("top_p", Json::number(s.top_p)).set("top_k", Json::integer(s.top_k)).set("seed", Json::integer(s.seed));
        j.set("ids", std::move(arr)).set("sampling", std::move(samp)).set("max_tokens", Json::integer(s.max_tokens));
        host_.send(j);
        return std::make_unique<HostSession>(host_, gid);
    }
private:
    Host& host_;
    int64_t id_ = 0;
    Json extra_;
};

Json metrics_json(const RuntimeMetrics& m) {
    Json j = Json::object();
    auto put = [&](const char* k, uint64_t v) { j.set(k, Json::integer(static_cast<int64_t>(v))); };
    put("requests", m.requests); put("attempts", m.attempts); put("regenerations", m.regenerations); put("loop_recoveries", m.loop_recoveries);
    put("retrievals", m.retrievals); put("claims_checked", m.claims_checked); put("contradictions", m.contradictions);
    put("late_contradictions", m.late_contradictions); put("hedged", m.hedged); put("unknown_delivered", m.unknown_delivered);
    put("temperature_changes", m.temperature_changes); put("safe_stops", m.safe_stops); put("vc_selections", m.vc_selections);
    put("vc_page_ins", m.vc_page_ins); put("vc_page_outs", m.vc_page_outs); put("vc_unavailable", m.vc_unavailable); put("vc_rejected", m.vc_rejected);
    put("kv_reused_tokens", m.kv_reused_tokens); put("kv_rebuilt_tokens", m.kv_rebuilt_tokens);
    return j;
}

} // namespace

int main() {
    std::ios::sync_with_stdio(false);
    Host host;
    std::unique_ptr<InferenceRuntime> runtime;
    std::shared_ptr<HostPrompt> prompt;
    std::shared_ptr<HostGenerator> generator;

    auto configure = [&](const Json& c) {
        RuntimeConfig cfg;
        cfg.virtual_context.physical_tokens = std::max<int64_t>(2048, c["physical_tokens"].i64(32768));
        if (c.has("virtual_tokens")) cfg.virtual_context.advertised_virtual_tokens = c["virtual_tokens"].i64();
        cfg.persist_contexts = c["persist"].boolean_or(true);
        cfg.context_storage_dir = c["storage_dir"].str();
        cfg.verification_enabled = c["verification"].boolean_or(true);
        cfg.recovery.strict = c["strict"].boolean_or(false);
        cfg.virtual_context.enabled = c["virtual_context"].boolean_or(true);
        RuntimeDeps d;
        prompt = std::make_shared<HostPrompt>(host, c["identity"].str("host"));
        generator = std::make_shared<HostGenerator>(host);
        d.prompt_builder = prompt;
        d.generator = generator;
        runtime = std::make_unique<InferenceRuntime>(d, cfg);
        host.set_metrics([&]() {
            Json j = metrics_json(runtime->metrics());
            Json r;
            if (Json::parse(runtime->contexts().retrieval_metrics().to_json(), r)) j.set("retrieval", r);   // hybrid retrieval and cache counters
            return j;
        });
        Json r = Json::object();
        r.set("op", Json::string("ready"));
        host.send(r);
    };

    auto handle_request = [&](const Json& m) {
        const int64_t id = m["id"].i64();
        host.begin_request(id);
        auto fail = [&](int status, const std::string& type, const std::string& msg) {
            Json e = Json::object();
            e.set("op", Json::string("error")).set("id", Json::integer(id)).set("status", Json::integer(status)).set("type", Json::string(type)).set("message", Json::string(msg));
            host.send(e);
        };
        if (!runtime) { fail(503, "engine_unavailable", "the runtime has not been configured"); return; }
        InferenceRequest req;
        const Json& sc = m["scope"];
        auto pick = [](const Json& v, std::string& dst) { if (v.is_string() && !v.str().empty()) dst = v.str(); };
        pick(sc["tenant"], req.scope.security.tenant_id);
        pick(sc["user"], req.scope.security.user_id);
        pick(sc["workspace"], req.scope.security.workspace_id);
        pick(sc["agent"], req.scope.security.agent_id);
        pick(sc["session"], req.scope.security.session_id);
        req.scope.request_id = std::to_string(id);
        for (const auto& j : m["messages"].items()) req.messages.push_back(message_from(j));
        const Json& s = m["sampling"];
        req.sampling.temperature = s["temperature"].num(0.0);
        req.sampling.top_p = s["top_p"].num(1.0);
        req.sampling.top_k = static_cast<int>(s["top_k"].i64(0));
        req.sampling.seed = s["seed"].i64(0);
        req.sampling.max_tokens = static_cast<int>(std::max<int64_t>(1, m["max_tokens"].i64(1024)));
        for (const auto& st : s["stop"].items()) if (st.is_string() && !st.str().empty()) req.sampling.stop.push_back(st.str());
        req.reasoning_prefix = m["reasoning_prefix"].boolean_or(false);
        req.debug = m["debug"].boolean_or(false);
        req.ephemeral = m["ephemeral"].boolean_or(false);
        Json extra = Json::object();
        for (const char* k : {"min_p", "repetition_penalty", "frequency_penalty", "presence_penalty"}) if (s.has(k)) extra.set(k, s[k]);
        generator->set_request(id, extra);
        prompt->set_request(id, req.reasoning_prefix);
        try {
            InferenceResponse out = runtime->generate(req, [&](const std::string& text) {
                Json e = Json::object();
                e.set("op", Json::string("emit")).set("id", Json::integer(id)).set("text", Json::string(text));
                host.send(e);
            });
            Json d = Json::object();
            d.set("op", Json::string("done")).set("id", Json::integer(id));
            d.set("finish", Json::string(host.cancelled() ? "cancel" : to_string(out.finish)));
            d.set("prompt_tokens", Json::integer(out.prompt_tokens)).set("completion_tokens", Json::integer(out.completion_tokens));
            if (req.debug) {
                Json tr = Json::array();
                for (const auto& t : out.trace) {
                    Json o = Json::object();
                    o.set("ms", Json::number(t.ms)).set("kind", Json::string(t.kind)).set("detail", Json::string(t.detail));
                    tr.push(std::move(o));
                }
                d.set("trace", std::move(tr));
            }
            host.send(d);
        } catch (const HostError& e) {
            fail(e.status, e.type, e.what());
        } catch (const Gone&) {
            throw;
        } catch (const std::exception& e) {
            fail(500, "server_error", e.what());
        }
    };

    try {
        while (true) {
            Json m = host.read();
            const std::string& op = m["op"].str();
            if (op == "quit") break;
            if (op == "config") configure(m);
            else if (op == "request") handle_request(m);
        }
    } catch (const Gone&) {
    }
    return 0;
}
