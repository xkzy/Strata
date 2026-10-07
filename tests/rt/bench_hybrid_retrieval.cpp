// tests/rt/bench_hybrid_retrieval.cpp - synthetic benchmark: BM25-only vs hybrid, cold vs warm (not a correctness test)
#include "strata/rt/retrieval.hpp"
#include "strata/rt/types.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <sstream>

using namespace strata::rt;
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 20000;
    std::mt19937 rng(7);
    auto alpha = [](uint32_t k) { std::string w = "zq"; do { w += static_cast<char>('a' + k % 26); k /= 26; } while (k); return w; };   // plain words, not identifier-like
    // a 3000-word vocabulary with a skewed distribution (a few common words, a long tail), like prose
    auto word = [&]() { const uint32_t r = rng() % 100; const uint32_t k = r < 40 ? rng() % 30 : r < 80 ? rng() % 400 : rng() % 3000; return alpha(k); };
    auto filler = [&](int n) { std::string s; for (int i = 0; i < n; ++i) s += word() + " "; return s; };

    std::vector<std::string> pages;
    std::vector<int> def_page;   // symbol k is defined on page def_page[k]
    const int symbols = N / 10;
    for (int i = 0; i < N; ++i) {
        std::string t;
        if (i % 10 == 0) {
            const int k = i / 10;
            t = "int handle_event_" + std::to_string(k) + "(struct ctx *c) {\n    // " + filler(20) + "\n    return forward_" + std::to_string(k % 97) + "(c);\n}\n";
            def_page.push_back(i);
        } else t = filler(60);
        pages.push_back(std::move(t));
    }
    auto build = [&](bool lexical_only) {
        RetrievalConfig c; c.lexical_only = lexical_only; c.l2_admit_min_docs = 0;
        auto ix = std::make_unique<HybridIndex>(c, nullptr, nullptr);
        for (int i = 0; i < N; ++i) ix->add(static_cast<uint32_t>(i), pages[i], hash128(pages[i]), DocInfo{"src/m" + std::to_string(i % 50) + ".c", 0.0});
        return ix;
    };

    struct Q { std::string text; int expect; };   // expect: page that must be in the top 5 (-1: none)
    std::vector<Q> qs;
    for (int k = 0; k < 200; ++k) qs.push_back({"handle_event_" + std::to_string((k * 7) % symbols) + "()", def_page[(k * 7) % symbols]});   // exact
    for (int k = 0; k < 100; ++k) qs.push_back({"where is handle_event_" + std::to_string((k * 11) % symbols) + " defined", def_page[(k * 11) % symbols]});   // structural
    for (int k = 0; k < 100; ++k) {   // keyword queries: three rare words taken from one page
        const int pg = (k * 131 + 7) % N;
        if (pg % 10 == 0) { qs.push_back({"timeout buffer", -1}); continue; }
        std::string q, w; int got = 0; std::istringstream is(pages[pg]);
        while (is >> w && got < 3) if (w.size() >= 5) { q += w + " "; ++got; }
        qs.push_back({q, got == 3 ? pg : -1});
    }

    std::printf("hardware: %s; %d pages (%d symbol definitions); %zu queries (200 exact, 100 structural, 100 keyword)\n",
                "this machine (single thread)", N, symbols, qs.size());
    for (int mode = 0; mode < 2; ++mode) {
        auto t0 = Clock::now();
        auto ix = build(mode == 0);
        const double build_ms = ms_since(t0);
        const char* names[3] = {"exact", "structural", "keyword"};
        double cold[3] = {0, 0, 0}, warm[3] = {0, 0, 0};
        int cnt[3] = {0, 0, 0}, hit[3] = {0, 0, 0}, asked[3] = {0, 0, 0};
        auto cls = [&](size_t i) { return i < 200 ? 0 : i < 300 ? 1 : 2; };
        for (size_t i = 0; i < qs.size(); ++i) {
            HybridQuery h; h.text = qs[i].text; h.scope_key = "bench";
            auto t = Clock::now();
            auto r = ix->search(h);
            cold[cls(i)] += ms_since(t); ++cnt[cls(i)];
            if (qs[i].expect >= 0) { ++asked[cls(i)]; for (size_t k = 0; k < r.size() && k < 5; ++k) if (r[k].doc == static_cast<uint32_t>(qs[i].expect)) { ++hit[cls(i)]; break; } }
        }
        for (size_t i = 0; i < qs.size(); ++i) {
            HybridQuery h; h.text = qs[i].text; h.scope_key = "bench";
            auto t = Clock::now();
            ix->search(h);
            warm[cls(i)] += ms_since(t);
        }
        auto m = ix->metrics();
        std::printf("%s (index build %.0f ms; one-time lazy embedding %.0f ms)\n", mode == 0 ? "BM25-only" : "hybrid", build_ms, m.embedding_ms);
        for (int c = 0; c < 3; ++c)
            std::printf("  %-10s cold %8.3f ms/query   warm %8.4f ms/query   recall@5 %3d/%d\n", names[c], cold[c] / cnt[c], warm[c] / cnt[c], hit[c], asked[c]);
        std::printf("  cache hit rate %.2f | runs: exact %llu lexical %llu vector %llu structural %llu | tokens: not measured here\n",
                    static_cast<double>(m.l0_hits + m.l2_hits + m.l3_hits + m.negative_hits) / static_cast<double>(std::max<uint64_t>(1, m.queries)),
                    (unsigned long long)m.exact_runs, (unsigned long long)m.lexical_runs, (unsigned long long)m.vector_runs, (unsigned long long)m.structural_runs);
    }
    return 0;
}
