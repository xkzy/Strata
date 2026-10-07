// tests/rt/test_symbolic_claims.cpp - equation and calculus statements in prose: what is recognized, what is not, and the verdicts
#undef NDEBUG
#include "strata/rt/claim.hpp"
#include "strata/rt/runtime.hpp"
#include "strata/rt/symbolic.hpp"

#include <cstdio>
#include <functional>

using namespace strata::rt;

static int g_fail = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  FAILED: %s (line %d)\n", m, __LINE__); ++g_fail; } } while (0)

static ClaimVerdict check(const std::string& sentence) {
    ClaimDetector det;
    RequestScope scope;
    auto claims = det.scan(sentence, 0, sentence.size(), scope);
    for (const auto& c : claims) {
        if (c.kind != ClaimKind::kSymbolic) continue;
        ProgressiveVerifier pv(std::make_shared<VerificationCache>());
        pv.add_verifier(std::make_shared<SymbolicVerifier>());
        return pv.verify(c, EvidenceSet());
    }
    ClaimVerdict none;
    none.explanation = "NO CLAIM";
    return none;
}
static bool no_claim(const std::string& s) { return check(s).explanation == "NO CLAIM"; }

struct ByteTokenizer : ITokenizer {
    std::vector<int32_t> encode(const std::string& t) const override { return std::vector<int32_t>(t.begin(), t.end()); }
    std::string decode(const std::vector<int32_t>& ids) const override { return std::string(ids.begin(), ids.end()); }
    std::string id() const override { return "byte-test"; }
};
struct ScriptedGenerator : IGenerator {
    std::function<std::string(const std::string&)> reply;
    struct Session : IGenSession {
        std::string text; size_t pos = 0;
        bool next(GenToken& o) override { if (pos >= text.size()) return false; size_t n = std::min<size_t>(3, text.size() - pos); o.text = text.substr(pos, n); o.id = 1000 + static_cast<int32_t>(pos); pos += n; return true; }
        void set_temperature(double) override {}
        void cancel() override {}
    };
    std::unique_ptr<IGenSession> start(const std::vector<int32_t>& ids, const Sampling&, const std::string&) override {
        auto s = std::make_unique<Session>(); s->text = reply(std::string(ids.begin(), ids.end())); return s;
    }
};

int main() {
    std::printf("[1] calculus statements\n");
    CHECK(check("The derivative of x^2 is 2x.").state == VerificationState::kVerified, "x^2 -> 2x");
    CHECK(check("The derivative of x^3 + 2x is 3x^2 + 2.").state == VerificationState::kVerified, "polynomial derivative");
    CHECK(check("So d/dx sin(x) = cos(x).").state == VerificationState::kVerified, "d/dx sin x = cos x");
    CHECK(check("The derivative of x^2 with respect to x is 2x.").state == VerificationState::kVerified, "explicit variable");
    auto bad = check("The derivative of x^3 is 2x^2.");
    CHECK(bad.state == VerificationState::kContradicted && bad.corrected_value.find("3") != std::string::npos, "wrong derivative contradicted with the right one");
    CHECK(check("The integral of 2x is x^2 + C.").state == VerificationState::kVerified, "integral with + C");
    CHECK(check("The antiderivative of cos(x) is sin(x).").state == VerificationState::kVerified, "antiderivative");
    CHECK(check("The integral of 2x is x^3.").state == VerificationState::kContradicted, "wrong integral");

    std::printf("[2] identities\n");
    CHECK(check("(x+1)^2 expands to x^2 + 2x + 1.").state == VerificationState::kVerified, "expansion");
    auto e2 = check("(x+1)^2 expands to x^2 + 1.");
    CHECK(e2.state == VerificationState::kContradicted && !e2.corrected_value.empty(), "wrong expansion has the correct one");
    CHECK(check("x^2 - 1 factors as (x - 1)(x + 1).").state == VerificationState::kVerified, "factorization");
    CHECK(check("x^2 - 1 factors into (x - 1)(x - 1).").state == VerificationState::kContradicted, "wrong factorization");
    CHECK(check("Note that sin(x)^2 + cos(x)^2 equals 1.").state == VerificationState::kVerified, "trig identity");
    CHECK(check("Then (x+1)(x-1) equals x^2 - 1.").state == VerificationState::kVerified, "equals");
    CHECK(check("Then (x+1)(x-1) equals x^2 + 1.").state == VerificationState::kContradicted, "false equals");

    std::printf("[3] equations\n");
    CHECK(check("The solution of 2x + 5 = 15 is x = 5.").state == VerificationState::kVerified, "linear");
    auto s2 = check("The solution of 2x + 5 = 15 is x = 6.");
    CHECK(s2.state == VerificationState::kContradicted && s2.corrected_value.find("5") != std::string::npos, "wrong solution corrected");
    CHECK(check("The solutions of x^2 - 5x + 6 = 0 are x = 2 and x = 3.").state == VerificationState::kVerified, "two roots");
    CHECK(check("The solutions of x^2 - 5x + 6 = 0 are x = 2 and x = 4.").state == VerificationState::kContradicted, "a wrong root");
    CHECK(check("The solution of x^2 - 5x + 6 = 0 is x = 2.").state == VerificationState::kContradicted, "an incomplete solution set is wrong");

    std::printf("[3b] limits\n");
    CHECK(check("The limit of sin(x)/x as x approaches 0 is 1.").state == VerificationState::kVerified, "sin x / x");
    CHECK(check("The limit of 1/x as x approaches infinity is 0.").state == VerificationState::kVerified, "1/x at infinity");
    auto l2 = check("The limit of (x^2 - 1)/(x - 1) as x approaches 1 is 3.");
    CHECK(l2.state == VerificationState::kContradicted && l2.corrected_value == "2", "wrong limit corrected");

    std::printf("[4] things that are not claims\n");
    CHECK(no_claim("The line is y = 2x + 1."), "a definition is not an identity");
    CHECK(no_claim("The solution of the problem is x = 5."), "no equation, no claim");
    CHECK(no_claim("Alice equals Bob in height."), "prose");
    CHECK(no_claim("The derivative of this is hard to compute."), "no expression");
    CHECK(no_claim("2 + 2 equals 4."), "pure arithmetic belongs to the arithmetic verifier");
    CHECK(no_claim("I think a equals b sometimes."), "single-letter words are not variables here");
    CHECK(no_claim("The integral of the function over the interval is large."), "prose integral");

    std::printf("[5] end to end: a wrong derivative is corrected internally\n");
    auto gen = std::make_shared<ScriptedGenerator>();
    gen->reply = [](const std::string& p) {
        return p.find("Verified facts for this answer") != std::string::npos ? std::string("The derivative of x^3 is 3x^2.") : std::string("The derivative of x^3 is 2x^2.");
    };
    RuntimeDeps d;
    d.generator = gen; d.tokenizer = std::make_shared<ByteTokenizer>();
    RuntimeConfig cfg; cfg.persist_contexts = false;
    InferenceRuntime rt(d, cfg);
    InferenceRequest r;
    r.scope.request_id = "q"; r.messages = {{"user", "What is the derivative of x^3?", ""}};
    std::string streamed;
    auto out = rt.generate(r, [&](const std::string& s) { streamed += s; });
    CHECK(out.text.find("3x^2") != std::string::npos && out.text.find("2x^2") == std::string::npos && streamed == out.text, "the caller only ever sees the correct derivative");
    CHECK(rt.metrics().regenerations == 1, "one internal regeneration");

    std::printf(g_fail ? "%d FAILED\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
