// tests/rt/test_hybrid_retrieval.cpp - hybrid retrieval: routing, fusion, multi-level caches, isolation, persistence
#undef NDEBUG
#include "strata/rt/retrieval.hpp"
#include "strata/rt/virtual_context.hpp"

#include <cstdio>
#include <atomic>
#include <filesystem>
#include <thread>

using namespace strata::rt;

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  FAIL: %s (line %d)\n", msg, __LINE__); ++failures; } } while (0)

static RetrievalConfig test_cfg() { RetrievalConfig c; c.l2_admit_min_docs = 0; return c; }
static bool has(const std::vector<Candidate>& v, uint32_t doc) { for (auto& c : v) if (c.doc == doc) return true; return false; }
static void add(HybridIndex& ix, uint32_t i, const std::string& text, const std::string& source = "conversation") {
    ix.add(i, text, hash128(text), DocInfo{source, 0.0});
}
static HybridQuery q(const std::string& text, const std::string& scope = "A") {
    HybridQuery h; h.text = text; h.scope_key = scope; return h;
}

static void corpus(HybridIndex& ix) {
    add(ix, 0, "int process_packet(struct pkt *p) {\n    if (validate_checksum(p) != 0) return -1;\n    return forward(p);\n}\n", "src/net/packet.c");
    add(ix, 1, "static int validate_checksum(const struct pkt *p) {\n    return crc32(p->data, p->len) == p->crc ? 0 : -1;\n}\n", "src/net/checksum.c");
    add(ix, 2, "The retry policy uses exponential backoff with a TCP connection timeout of 30 seconds.");
    add(ix, 3, "Connection pool exhaustion happens when workers hold sockets while waiting for the database.");
    add(ix, 4, "def handleRequest(req):\n    return process_packet(req.body)\n", "app/server.py");
    add(ix, 5, "import socket\nfrom net import transport\n", "app/net_glue.py");
    add(ix, 6, "We decided earlier to keep the cache write-through and revisit it after the benchmark.");
}

int main() {
    std::printf("[1] query classification and routing\n");
    CHECK(plan_query("process_packet()").cls == QueryClass::kExact, "identifier -> EXACT");
    CHECK(plan_query("foo.c").cls == QueryClass::kExact, "file name -> EXACT");
    CHECK(plan_query("ERR_CONN_RESET").cls == QueryClass::kExact, "error code -> EXACT");
    CHECK(plan_query("9fceb02d0ae598e9").cls == QueryClass::kExact, "commit hash -> EXACT");
    CHECK(plan_query("who calls validate_checksum").cls == QueryClass::kStructural, "callers -> STRUCTURAL");
    CHECK(plan_query("where is process_packet defined").structural_op == "defs", "definition question");
    CHECK(plan_query("how does packet processing work in the server?").cls == QueryClass::kSemantic, "natural language -> SEMANTIC");
    CHECK(plan_query("what did we decide earlier about the cache").cls == QueryClass::kConversational, "earlier -> CONVERSATIONAL");
    CHECK(plan_query("tcp connection timeout configuration").cls == QueryClass::kLexical, "keywords -> LEXICAL");
    CHECK(plan_query("fix process_packet timeout handling").cls == QueryClass::kCode, "identifier + words -> CODE");
    CHECK((plan_query("process_packet()").retrievers & kRetrieverVector) == 0, "exact plan never includes the vector retriever");

    std::printf("[2] exact queries run exact lookup only\n");
    {
        HybridIndex ix(test_cfg(), nullptr, nullptr);
        corpus(ix);
        auto r = ix.search(q("process_packet()"));
        auto m = ix.metrics();
        CHECK(has(r, 0) && has(r, 4), "definition and caller both contain the identifier");
        CHECK(m.exact_runs == 1 && m.vector_runs == 0 && m.lexical_runs == 0, "no vector or lexical search for an identifier");
        CHECK(m.exact_short_circuits == 1, "exact answered alone");
        auto r2 = ix.search(q("src/net/checksum.c"));
        CHECK(has(r2, 1), "a path finds the pages of that file");
    }

    std::printf("[3] snake_case and camelCase are the same name\n");
    {
        HybridIndex ix(test_cfg(), nullptr, nullptr);
        corpus(ix);
        CHECK(has(ix.search(q("processPacket")), 0), "camelCase query finds snake_case definition");
        CHECK(has(ix.search(q("handle_request")), 4), "snake_case query finds camelCase definition");
        CHECK(has(ix.search(q("how does validateChecksum verify the crc")), 1), "variants reach lexical retrieval too");
    }

    std::printf("[4] structural retrieval\n");
    {
        HybridIndex ix(test_cfg(), nullptr, nullptr);
        corpus(ix);
        auto callers = ix.search(q("who calls validate_checksum"));
        CHECK(has(callers, 0), "the caller is found");
        auto defs = ix.search(q("where is validate_checksum defined"));
        CHECK(!defs.empty() && defs.front().doc == 1, "the definition ranks first");
        auto imp = ix.search(q("which files imports socket"));
        CHECK(has(imp, 5), "importer found");
    }

    std::printf("[5] hybrid fusion counts a page once\n");
    {
        HybridIndex ix(test_cfg(), nullptr, nullptr);
        corpus(ix);
        auto r = ix.search(q("where is process_packet defined"));
        int n = 0;
        for (auto& c : r) if (c.doc == 0) ++n;
        CHECK(n == 1, "one result for a page returned by several retrievers");
        bool multi = false;
        for (auto& c : r) if (c.doc == 0 && (c.methods & (c.methods - 1))) multi = true;
        CHECK(multi, "its provenance records every retriever that found it");
        CHECK(ix.metrics().duplicate_candidates >= 1, "duplicate retrieval counted");
        for (auto& c : r) CHECK(c.score >= 0.0 && c.score <= 1.0, "fused score stays in 0..1");
    }

    std::printf("[6] vector retrieval (hashed n-grams: near-spellings, not paraphrases)\n");
    {
        HybridIndex ix(test_cfg(), nullptr, nullptr);
        corpus(ix);
        auto r = ix.search(q("connections pooling exhausting"));   // shares no whole word with the page: lexical finds nothing
        auto m = ix.metrics();
        CHECK(has(r, 3), "near-spellings (connection/connections, pool/pooling, exhaustion/exhausting) reach the page by character n-grams");
        CHECK(m.widened >= 1 && m.vector_runs >= 1, "the cheap plan found nothing, so the vector retriever ran");
        auto none = ix.search(q("quarterly marketing budget spreadsheet"));
        CHECK(!has(none, 3), "unrelated text does not match");
    }

    std::printf("[7] caches: request-local, query, negative, invalidation\n");
    {
        HybridIndex ix(test_cfg(), nullptr, nullptr);
        corpus(ix);
        RequestCache rc;
        ix.search(q("tcp connection timeout configuration"), &rc);
        ix.search(q("tcp connection timeout configuration"), &rc);
        CHECK(ix.metrics().l0_hits == 1, "L0: the same query in one request is searched once");
        ix.search(q("tcp connection timeout configuration"));
        CHECK(ix.metrics().l2_hits == 1, "L2: a repeated query across requests hits the query cache");
        auto none = ix.search(q("nonexistent_symbol_xyz"));
        CHECK(none.empty(), "nothing found");
        const auto runs = ix.metrics().exact_runs + ix.metrics().lexical_runs + ix.metrics().vector_runs;
        ix.search(q("nonexistent_symbol_xyz"));
        const auto runs2 = ix.metrics().exact_runs + ix.metrics().lexical_runs + ix.metrics().vector_runs;
        CHECK(ix.metrics().negative_hits == 1 && runs2 == runs, "negative cache: the repeated miss costs no retrieval");
        add(ix, 8, "an unrelated page about lunch menus");
        const auto r3 = ix.metrics().exact_runs + ix.metrics().lexical_runs + ix.metrics().vector_runs;
        ix.search(q("nonexistent_symbol_xyz"));
        CHECK(ix.metrics().negative_hits == 2 && ix.metrics().exact_runs + ix.metrics().lexical_runs + ix.metrics().vector_runs == r3, "an append that adds none of its terms keeps the negative entry");
        add(ix, 7, "the nonexistent_symbol_xyz function was added today");
        CHECK(has(ix.search(q("nonexistent_symbol_xyz")), 7), "an append invalidates the negative entry: the new page is found");
    }

    std::printf("[8] a shared hash never grants access\n");
    {
        auto shared = std::make_shared<EmbeddingCache>();
        HybridIndex ix(test_cfg(), nullptr, shared);
        const std::string same = "the deploy key rotation procedure for the payments cluster";
        add(ix, 0, same);   // scope A
        add(ix, 1, same);   // scope B, identical content and hash
        HybridQuery qa = q("deploy key rotation procedure payments", "A");
        qa.allowed = [](uint32_t d) { return d == 0; };
        HybridQuery qb = q("deploy key rotation procedure payments", "B");
        qb.allowed = [](uint32_t d) { return d == 1; };
        auto ra = ix.search(qa);
        auto rb = ix.search(qb);   // same text, same index version: must not be served A's cached result
        CHECK(ra.size() == 1 && ra[0].doc == 0, "A sees only its page");
        CHECK(rb.size() == 1 && rb[0].doc == 1, "B sees only its page, not A's cached result");
        HybridQuery qc = q("deploy key rotation procedure payments", "C");
        qc.allowed = [](uint32_t) { return false; };
        CHECK(ix.search(qc).empty(), "a scope with no access gets nothing, cached or not");
        // a result cached under an authorization that is later withdrawn is re-checked on return
        HybridQuery qd = q("deploy key rotation procedure payments", "A");
        bool open = true;
        qd.allowed = [&](uint32_t d) { return open && d == 0; };
        CHECK(!ix.search(qd).empty(), "authorized: found");
        open = false;
        CHECK(ix.search(qd).empty(), "authorization withdrawn: the cached entry is not returned");
        // embeddings are shared by content, not results
        ix.search(q("a long natural language question about rotation of keys for payments clusters?", "A"));
        CHECK(shared->hits() >= 1, "L4: the identical content was embedded once and reused");
    }

    std::printf("[9] retrieval loop guard\n");
    {
        HybridIndex ix(test_cfg(), nullptr, nullptr);
        CHECK(ix.observe_state("q", "c", false) == 1 && ix.observe_state("q", "c", false) == 2, "repeats without progress are counted");
        CHECK(!ix.loop_throttled("q", "c"), "below the limit");
        ix.observe_state("q", "c", false);
        CHECK(ix.loop_throttled("q", "c") && ix.metrics().loop_throttled >= 1, "throttled at the limit");
        ix.observe_state("q", "c", true);
        CHECK(!ix.loop_throttled("q", "c"), "new evidence resets it");
    }

    std::printf("[10] materialization cache (formatted text; token ids live in the Go server)\n");
    {
        HybridIndex ix(test_cfg(), nullptr, nullptr);
        std::string out;
        CHECK(!ix.materialized_get("k", out), "miss");
        ix.materialized_put("k", "block");
        CHECK(ix.materialized_get("k", out) && out == "block" && ix.metrics().l6_hits == 1, "hit");
    }

    std::printf("[11] the virtual context uses the hybrid index and survives a restart\n");
    {
        const std::string dir = (std::filesystem::temp_directory_path() / "strata_hybrid_test").string();
        std::filesystem::remove_all(dir);
        VirtualContextConfig cfg;
        cfg.retrieval.l2_admit_min_docs = 0;
        strata::context::SecurityScope scope;
        scope.session_id = "s1";
        {
            VirtualContextStore store(cfg, nullptr, dir);
            auto vc = store.open(scope);
            NewItem f;
            f.kind = ItemKind::kFile; f.role = "tool"; f.source = "src/net/packet.c"; f.provenance = Provenance::kExternalKnowledge;
            f.content = "int process_packet(struct pkt *p) {\n    return forward(p);\n}\n";
            vc->append(f);
            for (int i = 0; i < 20; ++i) { NewItem n; n.content = "unrelated chatter number " + std::to_string(i) + " about lunch plans"; vc->append(n); }
            auto hits = vc->search(scope, "process_packet", 5);
            CHECK(!hits.empty() && hits[0].source == "src/net/packet.c", "exact lookup through the virtual context");
            CHECK(vc->retrieval_metrics().exact_runs >= 1, "the context's metrics show it");
            store.persist(vc);
        }
        {
            VirtualContextStore store(cfg, nullptr, dir);
            auto vc = store.open(scope);
            auto hits = vc->search(scope, "processPacket", 5);
            CHECK(!hits.empty() && hits[0].source == "src/net/packet.c", "after a restart the index is rebuilt and finds the symbol");
            strata::context::SecurityScope other = scope;
            other.session_id = "s2";
            other.user_id = "someone_else";
            CHECK(vc->search(other, "process_packet", 5).empty(), "another scope cannot read it");
        }
        std::filesystem::remove_all(dir);
    }

    std::printf("[12] concurrent appends, searches and scopes\n");
    {
        VirtualContextConfig cfg;
        cfg.retrieval.l2_admit_min_docs = 0;
        VirtualContextStore store(cfg);
        std::atomic<int> wrong{0};
        std::vector<std::thread> th;
        for (int t = 0; t < 6; ++t)
            th.emplace_back([&, t]() {
                strata::context::SecurityScope sc;
                sc.session_id = "sess" + std::to_string(t % 3);   // two threads share each session
                sc.user_id = "user" + std::to_string(t % 3);
                auto vc = store.open(sc);
                for (int i = 0; i < 150; ++i) {
                    NewItem n;
                    n.content = "note " + std::to_string(t) + "_" + std::to_string(i) + " about handler_" + std::to_string(t % 3) + "_" + std::to_string(i) + " and caches";
                    vc->append(n);
                    for (auto& h : vc->search(sc, "handler_" + std::to_string(t % 3) + "_" + std::to_string(i), 5))
                        if (h.text.find("handler_" + std::to_string(t % 3) + "_") == std::string::npos) ++wrong;   // a page of another session leaked
                    vc->search(sc, "how do the caches work in this handler?", 5);
                }
            });
        for (auto& x : th) x.join();
        CHECK(wrong == 0, "no result from another session under concurrency");
        strata::context::SecurityScope sc; sc.session_id = "sess0"; sc.user_id = "user0";
        CHECK(!store.open(sc)->search(sc, "handler_0_149", 3).empty(), "the last concurrent append is searchable");
    }

    std::printf(failures ? "FAILED (%d)\n" : "ALL PASSED\n", failures);
    return failures ? 1 : 0;
}
