// tests/rt/test_json.cpp - the runtime's JSON layer: round trips, escapes, and hostile input
#undef NDEBUG
#include "strata/rt/json.hpp"

#include <cstdio>

using strata::rt::Json;
static int g_fail = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  FAILED: %s (line %d)\n", m, __LINE__); ++g_fail; } } while (0)

static bool parses(const std::string& s, Json& out) { return Json::parse(s, out); }

int main() {
    Json j;
    CHECK(parses(R"({"a":1,"b":[true,false,null,"x\ny"],"c":{"d":-2.5e1},"e":"\u00e9\ud83d\ude00"})", j), "valid document");
    CHECK(j["a"].i64() == 1 && j["b"].size() == 4 && j["b"].at(3).str() == "x\ny" && j["c"]["d"].num() == -25.0, "values");
    CHECK(j["e"].str() == "\xC3\xA9\xF0\x9F\x98\x80", "\\u escapes and surrogate pairs become UTF-8");
    Json round;
    CHECK(parses(j.dump(), round) && round.dump() == j.dump(), "dump/parse round trip");
    CHECK(j["missing"]["deeper"].is_null() && j["a"].str().empty(), "lookups on absent or wrong-typed values are safe");

    Json o = Json::object();
    o.set("t", Json::string("tab\there \"quoted\" \\ and \x01 control"));
    Json back;
    CHECK(parses(o.dump(), back) && back["t"].str() == o["t"].str(), "control characters survive");
    CHECK(o.dump().find('\n') == std::string::npos, "a message is one line");

    for (const char* bad : {"", "{", "{\"a\":}", "[1,]", "{\"a\" 1}", "nul", "\"abc", "{\"a\":1} x", "\"\\q\"", "\"\\u12\"", "[1 2]", "01x", "\"raw\ncontrol\""}) {
        Json x;
        CHECK(!Json::parse(bad, x), bad);
    }
    std::string deep(1000, '['), deep_end(1000, ']');
    Json x;
    std::string err;
    CHECK(!Json::parse(deep + deep_end, x, &err) && err.find("deep") != std::string::npos, "nesting depth is bounded");
    CHECK(!Json::parse(std::string(100, 'x'), x, &err, 64, 50) && err.find("large") != std::string::npos, "size is bounded");
    CHECK(parses("\"\\ud800 lone\"", x) && x.str().find("\xEF\xBF\xBD") == 0, "a lone surrogate becomes U+FFFD, not invalid UTF-8");
    CHECK(parses("12345678901234567890", x) && x.num() > 1e19, "an integer too large for int64 is kept as a double");
    std::printf(g_fail ? "%d FAILED\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
