// tests/rt/test_transparent_runtime.cpp - the acceptance scenario: a normal request goes in, internal context,
// verification and regeneration happen, a normal response comes out. No tool call, no special prompt.
#undef NDEBUG
#include "strata/rt/runtime.hpp"

#include <cstdio>
#include <functional>

using namespace strata::rt;

static int g_fail = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  FAILED: %s (line %d)\n", m, __LINE__); ++g_fail; } } while (0)

struct ByteTokenizer : ITokenizer {
    std::vector<int32_t> encode(const std::string& t) const override { return std::vector<int32_t>(t.begin(), t.end()); }
    std::string decode(const std::vector<int32_t>& ids) const override { return std::string(ids.begin(), ids.end()); }
    std::string id() const override { return "byte-test"; }
};

// scripted model: reply(prompt, attempt) -> text, delivered 3 bytes per token
struct ScriptedGenerator : IGenerator {
    using Reply = std::function<std::string(const std::string& prompt, int attempt)>;
    Reply reply;
    int attempts = 0;
    explicit ScriptedGenerator(Reply r) : reply(std::move(r)) {}
    struct Session : IGenSession {
        std::string text; size_t pos = 0; bool cancelled = false;
        bool next(GenToken& o) override {
            if (cancelled || pos >= text.size()) return false;
            size_t n = std::min<size_t>(3, text.size() - pos);
            o.text = text.substr(pos, n); o.id = static_cast<int32_t>(std::hash<std::string>()(o.text) & 0x7fff); pos += n;
            return true;
        }
        void set_temperature(double) override {}
        void cancel() override { cancelled = true; }
    };
    std::unique_ptr<IGenSession> start(const std::vector<int32_t>& ids, const Sampling&, const std::string&) override {
        auto s = std::make_unique<Session>();
        s->text = reply(std::string(ids.begin(), ids.end()), attempts++);
        return s;
    }
};

static InferenceRuntime make(ScriptedGenerator::Reply r, std::shared_ptr<ExactStateStore> st = nullptr, RuntimeConfig cfg = RuntimeConfig()) {
    RuntimeDeps d;
    d.generator = std::make_shared<ScriptedGenerator>(std::move(r));
    d.tokenizer = std::make_shared<ByteTokenizer>();
    d.exact_state = st;
    cfg.persist_contexts = false;
    return InferenceRuntime(d, cfg);
}

static InferenceRequest ask(const std::string& q, const std::string& user = "u1", const std::string& tenant = "t1") {
    InferenceRequest r;
    r.scope.security.user_id = user; r.scope.security.tenant_id = tenant; r.scope.request_id = "r";
    r.messages = {{"user", q, ""}};
    return r;
}

int main() {
    {   std::printf("[1] wrong arithmetic is regenerated; the caller never sees the wrong value\n");
        auto rt = make([](const std::string& p, int) {
            return p.find("703") != std::string::npos ? std::string("37 x 19 = 703.") : std::string("The product is 37 * 19 = 713.");
        });
        std::string streamed;
        auto q1 = ask("What is 37 * 19?"); q1.debug = true;
        auto out = rt.generate(q1, [&](const std::string& s) { streamed += s; });
        if (getenv("RT_TRACE")) { std::printf("text=[%s]\n", out.text.c_str()); for (auto& t : out.trace) std::printf("  %s %s\n", t.kind.c_str(), t.detail.c_str()); }
        CHECK(out.text.find("703") != std::string::npos, "corrected answer delivered");
        CHECK(out.text.find("713") == std::string::npos && streamed.find("713") == std::string::npos, "wrong value never streamed");
        CHECK(streamed == out.text, "stream equals final text");
        CHECK(rt.metrics().regenerations == 1, "one internal regeneration");
    }
    {   std::printf("[2] authoritative state beats the model, scoped per tenant\n");
        auto st = std::make_shared<ExactStateStore>();
        strata::context::SecurityScope a, b; a.tenant_id = "ta"; b.tenant_id = "tb";
        st->put(a, strata::context::SharingScope::PRIVATE, "server.conf", "port", "8080");
        st->put(b, strata::context::SharingScope::PRIVATE, "server.conf", "port", "9090");
        auto rt = make([](const std::string&, int) { return std::string("The port is 8080."); }, st);
        auto ra = rt.generate(ask("What port is configured?", "default_user", "ta"));
        auto rb = rt.generate(ask("What port is configured?", "default_user", "tb"));
        CHECK(ra.text.find("8080") != std::string::npos, "tenant A confirmed");
        CHECK(rb.text.find("8080") == std::string::npos, "tenant B's contradicted claim is not delivered");
        CHECK(rt.metrics().contradictions >= 1, "contradiction counted for B only");
    }
    {   std::printf("[3] a correct answer passes through untouched\n");
        auto rt = make([](const std::string&, int) { return std::string("Sure. 2 + 2 = 4."); });
        auto out = rt.generate(ask("2+2?"));
        CHECK(out.text == "Sure. 2 + 2 = 4.", "text unchanged");
        CHECK(rt.metrics().regenerations == 0, "no regeneration");
    }
    {   std::printf("[4] a generation loop is recovered internally\n");
        auto rt = make([](const std::string&, int a) {
            if (a == 0) { std::string s = "Here is the plan. "; for (int i = 0; i < 80; ++i) s += "and then we repeat the step. "; return s; }
            return std::string("Here is the plan. Done.");
        });
        auto out = rt.generate(ask("plan?"));
        CHECK(rt.metrics().loop_recoveries >= 1, "loop recovered");
        CHECK(out.text.find("Done.") != std::string::npos, "final text complete");
    }
    {   std::printf("[5] debug trace only on request\n");
        auto rt = make([](const std::string&, int) { return std::string("ok 3 + 4 = 7."); });
        auto q = ask("x"); auto plain = rt.generate(q); q.debug = true; auto dbg = rt.generate(q);
        CHECK(plain.trace.empty() && !dbg.trace.empty(), "trace gated by debug flag");
    }
    {   std::printf("[6] huge virtual context behind an ordinary endpoint (agent resends only the last message)\n");
        std::string last_prompt; size_t max_prompt = 0; bool leaked_marker = false;
        RuntimeConfig cfg; cfg.virtual_context.physical_tokens = 2048;
        auto rt = make([&](const std::string& p, int) {
            last_prompt = p; max_prompt = std::max(max_prompt, p.size());
            for (const char* m : {"paged", "unloaded", "summary inserted", "[retrieved"}) if (p.find(m) != std::string::npos) leaked_marker = true;
            // an honest model: answers from what it can see
            size_t q = p.rfind("chunk_size is ");
            if (p.find("chunk_size 4096") != std::string::npos) return std::string("It was 4096.");
            if (p.find("not available to you") != std::string::npos) return std::string("I do not have that earlier context.");
            (void)q; return std::string("Noted.");
        }, nullptr, cfg);
        auto turn = [&](const std::string& text, const std::string& session = "s1") {
            InferenceRequest r = ask(text); r.scope.security.session_id = session; return rt.generate(r);
        };
        turn("Decision: use algorithm Blake3 with chunk_size 4096 for hashing.");
        for (int i = 0; i < 300; ++i) turn("Status note " + std::to_string(i) + ": build finished, nothing unusual, moving on to the next small task today.");
        auto out = turn("Which chunk_size did we discuss earlier?");
        CHECK(out.text == "It was 4096.", "old decision available without the agent resending or retrieving it");
        CHECK(max_prompt / 4 <= 2200, "every physical prompt stayed within the model window");
        CHECK(!leaked_marker, "paging never shows up in the prompt");
        auto other = turn("Which chunk_size did we discuss earlier?", "s2");
        CHECK(other.text == "I do not have that earlier context.", "another session cannot see it; honest unavailable state instead of a guess");
        std::string info = rt.model_info_json(ask("x").scope);
        CHECK(info.find("\"physical\":2048") != std::string::npos && info.find("\"virtual\":10000000") != std::string::npos, "window metadata distinguishes physical from virtual");
        CHECK(rt.metrics().vc_page_ins > 0 && rt.metrics().kv_reused_tokens > 0, "paging and KV reuse happened internally");
    }
    {   std::printf("[7] capacity exhausted: the current question is still delivered\n");
        std::string seen;
        RuntimeConfig cfg; cfg.virtual_context.advertised_virtual_tokens = 400; cfg.virtual_context.physical_tokens = 2048;
        auto rt = make([&](const std::string& p, int) { seen = p; return std::string("ok"); }, nullptr, cfg);
        for (int i = 0; i < 30; ++i) rt.generate(ask("filler message number " + std::to_string(i) + " with enough words to use up the tiny capacity quickly, again and again"));
        rt.generate(ask("What is the zebra-question?"));
        CHECK(seen.find("What is the zebra-question?") != std::string::npos, "question reaches the model even past capacity");
        CHECK(rt.metrics().vc_rejected > 0, "rejections are counted, not silent");
    }
    {   std::printf("[8] tool output of this request is exact evidence even when the window pages it\n");
        RuntimeConfig cfg;
        auto rt = make([](const std::string& p, int) {
            return p.find("8080") != std::string::npos && p.find("Correct value") != std::string::npos ? std::string("Connecting on port 8080.") : std::string("Connecting. The port is 9090.");
        }, nullptr, cfg);
        InferenceRequest r = ask("ok, connect to it");
        r.messages = {{"user", "check the config", ""}, {"tool", "port: 8080\nhost: localhost\n", "read_config"}, {"user", "ok, connect to it", ""}};
        auto out = rt.generate(r);
        CHECK(out.text.find("9090") == std::string::npos && out.text.find("8080") != std::string::npos, "contradicted by the tool result and corrected");
    }
    {   std::printf("[9] sessions survive a restart (no resend, nothing in RAM)\n");
        char tmpl[] = "/tmp/strata_rtXXXXXX"; std::string dir = mkdtemp(tmpl);
        RuntimeConfig cfg; cfg.persist_contexts = true; cfg.context_storage_dir = dir; cfg.virtual_context.physical_tokens = 2048;
        auto reply = [](const std::string& p, int) { return p.find("chunk_size 4096") != std::string::npos ? std::string("It was 4096.") : std::string("Noted."); };
        {
            RuntimeDeps d; d.generator = std::make_shared<ScriptedGenerator>(reply); d.tokenizer = std::make_shared<ByteTokenizer>();
            InferenceRuntime rt(d, cfg);
            rt.generate(ask("Decision: use algorithm Blake3 with chunk_size 4096 for hashing."));
            for (int i = 0; i < 100; ++i) rt.generate(ask("Status note " + std::to_string(i) + ": build finished, nothing unusual, moving on to the next task today."));
        }
        RuntimeDeps d2; d2.generator = std::make_shared<ScriptedGenerator>(reply); d2.tokenizer = std::make_shared<ByteTokenizer>();
        InferenceRuntime rt2(d2, cfg);
        auto out = rt2.generate(ask("Which chunk_size did we discuss earlier?"));
        CHECK(out.text == "It was 4096.", "context restored from disk after restart");
        (void)std::system(("rm -rf '" + dir + "'").c_str());
    }
    {   std::printf("[10] ordinary 'before' wording does not trigger the unavailable-context state\n");
        std::string seen;
        RuntimeConfig cfg; cfg.persist_contexts = false;
        auto rt = make([&](const std::string& p, int) { seen = p; return std::string("ok"); }, nullptr, cfg);
        rt.generate(ask("The weather is nice and the build is green today."));
        rt.generate(ask("Run the tests before you commit, and originally name it widget_main."));
        CHECK(seen.find("not available to you") == std::string::npos, "no false 'context unavailable' note");
    }
    {   std::printf("[11] number-theory claims in prose are checked exactly\n");
        RuntimeConfig cfg; cfg.persist_contexts = false;
        auto rt = make([](const std::string& p, int) {
            const bool fixed = p.find("false (computed exactly)") != std::string::npos;
            return fixed ? std::string("91 is not prime, it is 7 * 13.") : std::string("Yes, 91 is prime.");
        }, nullptr, cfg);
        auto out = rt.generate(ask("Is 91 prime?"));
        CHECK(out.text.find("not prime") != std::string::npos && out.text.find("Yes") == std::string::npos, "false primality claim corrected");
        auto rt2 = make([](const std::string&, int) { return std::string("The 20th Fibonacci number is 6765, and 97 is prime."); }, nullptr, cfg);
        auto ok = rt2.generate(ask("fib 20 and 97?"));
        CHECK(ok.text.find("6765") != std::string::npos && rt2.metrics().regenerations == 0, "true claims pass untouched");
        auto rt3 = make([](const std::string& p, int) { return p.find("Correct value") != std::string::npos ? std::string("The 20th Fibonacci number is 6765.") : std::string("The 20th Fibonacci number is 6764."); }, nullptr, cfg);
        auto bad = rt3.generate(ask("fib 20?"));
        CHECK(bad.text.find("6765") != std::string::npos && bad.text.find("6764") == std::string::npos, "wrong Fibonacci value corrected");
    }
    std::printf(g_fail ? "%d FAILED\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
