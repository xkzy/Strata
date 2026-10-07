// tests/rt/test_virtual_window.cpp - virtual context window: paging, retrieval, reversibility, isolation
#undef NDEBUG
#include "strata/rt/virtual_context.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

using namespace strata::rt;
using strata::context::SecurityScope;
using strata::context::SharingScope;

static int g_fail = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  FAILED: %s (line %d)\n", m, __LINE__); ++g_fail; } } while (0)

static VirtualContextConfig small_cfg() {
    VirtualContextConfig c;
    c.physical_tokens = 2048;
    c.advertised_virtual_tokens = 10000000;
    return c;
}
static bool has(const WorkingSet& ws, const std::string& needle) {
    for (const auto& m : ws.messages) if (m.content.find(needle) != std::string::npos) return true;
    return false;
}
static std::string filler(int i) {
    return "Status note " + std::to_string(i) + ": the build finished, nothing unusual happened today, moving on to the next small task in the queue.";
}
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
    SecurityScope me;
    {   std::printf("[1] old decision is recovered by 'discussed earlier' after 3000 messages\n");
        VirtualContext vc("c1", me, small_cfg());
        std::vector<Message> hist;
        for (int i = 0; i < 3000; ++i) {
            if (i == 100) hist.push_back({"user", "Decision: use algorithm Blake3 with chunk_size 4096 for the hashing stage.", ""});
            else hist.push_back({i % 2 ? "assistant" : "user", filler(i), ""});
        }
        hist.push_back({"user", "Change the chunk_size parameter we discussed earlier to 8192.", ""});
        std::set<uint64_t> fresh;
        auto ids = vc.ingest(hist, &fresh);
        SelectionRequest r; r.query = hist.back().content; r.budget_tokens = 1500; r.current_items = {ids.back()}; r.requester = me;
        auto ws = vc.select(r);
        CHECK(has(ws, "Blake3 with chunk_size 4096"), "decision from message 100 paged in");
        CHECK(has(ws, "Change the chunk_size parameter"), "current request present");
        CHECK(ws.tokens <= 1500 + 64, "working set respects the physical budget");
        CHECK(ws.unavailable.empty(), "nothing reported unavailable");
        auto wi = vc.window_info();
        CHECK(wi.virtual_used > wi.physical * 3, "logical context is far larger than the physical window");
        CHECK(wi.virtual_capacity <= wi.virtual_advertised, "capacity never exceeds the advertised size");
        std::string all;
        for (const auto& m : ws.messages) all += m.content;
        CHECK(all.find("[") == std::string::npos || all.find("paged") == std::string::npos, "no paging markers in the prompt");
    }
    {   std::printf("[2] resent history is deduplicated; only the new message is appended\n");
        VirtualContext vc("c2", me, small_cfg());
        std::vector<Message> h = {{"system", "You are helpful.", ""}, {"user", "hello there", ""}, {"assistant", "hi", ""}, {"user", "what is 2+2?", ""}};
        vc.ingest(h);
        const size_t n1 = vc.stats().items;
        std::set<uint64_t> fresh;
        h.push_back({"assistant", "4", ""}); h.push_back({"user", "thanks", ""});
        vc.ingest(h, &fresh);
        CHECK(vc.stats().items == n1 + 2 && fresh.size() == 2, "exactly two new items");
        std::set<uint64_t> again;
        vc.ingest(h, &again);
        CHECK(vc.stats().items == n1 + 2 && again.empty(), "an identical retry adds nothing");
        // a truncated history (agent dropped old turns) still ends with the new message appended
        std::vector<Message> cut = {{"user", "thanks", ""}, {"user", "thanks", ""}};
        std::set<uint64_t> f3; vc.ingest(cut, &f3);
        CHECK(!f3.empty(), "a repeated user text that is not a retry is a new message");
    }
    {   std::printf("[3] nothing is deleted: spill to storage, restore byte for byte, detect corruption\n");
        char tmpl[] = "/tmp/strata_vwXXXXXX"; std::string dir = mkdtemp(tmpl);
        auto be = std::make_shared<FilePageBackend>(dir);
        auto cfg = small_cfg(); cfg.resident_bytes = 20000;
        VirtualContext vc("c3", me, cfg, be);
        std::vector<std::string> originals; std::vector<uint64_t> its;
        for (int i = 0; i < 400; ++i) { originals.push_back(filler(i) + " unique-token-" + std::to_string(i * 7919)); NewItem n; n.content = originals.back(); its.push_back(vc.append(n)); }
        size_t sp = vc.compact();
        auto s = vc.stats();
        CHECK(sp > 0 && s.spilled_pages > 0 && s.resident_bytes <= cfg.resident_bytes + 4096, "cold pages spilled, memory bounded");
        bool same = true;
        for (size_t i = 0; i < originals.size(); ++i) { std::string c; auto pg = vc.item_pages(its[i]); if (pg.empty() || !vc.page_content(pg[0], c) || c != originals[i]) same = false; }
        CHECK(same, "every original is recoverable byte for byte");
        auto hits = vc.search(me, "unique-token-" + std::to_string(5 * 7919), 3);
        CHECK(!hits.empty() && hits[0].text == originals[5], "archived page found and restored by search");
        // corrupt one file: the page must be reported unavailable, never returned as wrong text
        auto pg = vc.item_pages(its[7]);
        ContextPage info; vc.page_info(pg[0], info);
        std::string cmd = "echo garbage > '" + dir + "/" + info.content_hash + ".page'"; (void)std::system(cmd.c_str());
        std::string c; bool ok = vc.page_content(pg[0], c);
        CHECK(!ok || c == originals[7], "corrupted storage never yields wrong text");
        std::string rm = "rm -rf '" + dir + "'"; (void)std::system(rm.c_str());
    }
    {   std::printf("[4] a 200,000-line tool result stays logical; only the relevant region is materialized\n");
        VirtualContext vc("c4", me, small_cfg());
        std::string big;
        for (int i = 0; i < 200000; ++i) { big += "+ line " + std::to_string(i) + " of the diff changes nothing important\n"; if (i == 123456) big += "+ int compute_checksum(const char* buf) { return crc32(buf); }\n"; }
        NewItem t; t.kind = ItemKind::kToolResult; t.role = "tool"; t.source = "git diff"; t.content = big;
        NewItem u; u.content = "Where is compute_checksum changed in the diff?";
        uint64_t tid = vc.append(t), uid = vc.append(u);
        SelectionRequest r; r.query = u.content; r.budget_tokens = 2000; r.current_items = {tid, uid}; r.requester = me;
        auto ws = vc.select(r);
        CHECK(has(ws, "compute_checksum(const char* buf)"), "relevant region of the tool result materialized");
        CHECK(ws.tokens <= 2000 + 64, "huge result did not blow the window");
        CHECK(vc.window_info().virtual_used > 1000000, "the whole result is part of the logical context");
    }
    {   std::printf("[5] dependencies travel together (definition + later reference)\n");
        VirtualContext vc("c5", me, small_cfg());
        NewItem d; d.content = "class FrameScheduler owns the retry policy. It is defined in scheduler_core.cpp and retries three times.";
        vc.append(d);
        for (int i = 0; i < 1500; ++i) { NewItem n; n.content = filler(i); vc.append(n); }
        NewItem ref; ref.content = "Later we raised FrameScheduler limits when the queue backed up.";
        uint64_t rid = vc.append(ref);
        for (int i = 0; i < 1500; ++i) { NewItem n; n.content = filler(i + 5000); vc.append(n); }
        NewItem q; q.content = "What did we raise for the queue backup?";
        uint64_t qid = vc.append(q);
        SelectionRequest r; r.query = "raise limits queue backed up"; r.budget_tokens = 1200; r.current_items = {qid}; r.requester = me;
        auto ws = vc.select(r);
        CHECK(has(ws, "Later we raised FrameScheduler limits"), "reference page found");
        CHECK(has(ws, "class FrameScheduler owns the retry policy"), "its definition came with it");
        (void)rid;
    }
    {   std::printf("[6] isolation: authorization happens before retrieval\n");
        SecurityScope a, b, c;
        a.user_id = "alice"; b.user_id = "bob"; c.user_id = "carol"; a.workspace_id = b.workspace_id = "proj"; c.workspace_id = "other";
        VirtualContext vc("c6", a, small_cfg());
        NewItem priv; priv.content = "alice private note: the vault passphrase is kiwi-lamp-77"; priv.sharing = SharingScope::PRIVATE;
        NewItem proj; proj.content = "project note: deployment uses region eu-west-3"; proj.sharing = SharingScope::PROJECT;
        vc.append(priv); vc.append(proj);
        CHECK(!vc.search(a, "vault passphrase", 3).empty(), "owner finds private note");
        CHECK(vc.search(b, "vault passphrase", 3).empty(), "same-project user cannot see a PRIVATE page, however similar the query");
        CHECK(!vc.search(b, "deployment region", 3).empty(), "PROJECT page visible inside the project");
        CHECK(vc.search(c, "deployment region", 3).empty(), "PROJECT page invisible outside the project");
        SelectionRequest r; r.query = "vault passphrase"; r.budget_tokens = 800; r.requester = c;
        CHECK(!has(vc.select(r), "kiwi-lamp-77"), "select() never materializes unauthorized pages");
        {
            VirtualContextStore st(small_cfg());
            SecurityScope e1; e1.session_id = "ephemeral-x";
            auto vc1 = st.open_ephemeral(e1);
            NewItem n; n.content = "private to this request"; vc1->append(n);
            CHECK(st.size() == 1 && st.open(e1) == vc1, "an ephemeral context is found by its scope while the request runs");
            st.erase(e1);
            CHECK(st.size() == 0 && st.open(e1)->stats().items == 0, "erase drops it: nothing is left to recall");
        }
        auto s1 = VirtualContextStore(small_cfg()); SecurityScope s_a, s_b; s_b.session_id = "other_session";
        CHECK(s1.open(s_a) != s1.open(s_b) && s1.size() == 2, "different sessions get different contexts");
    }
    {   std::printf("[7] unavailable context is reported, not fabricated\n");
        VirtualContext vc("c7", me, small_cfg());
        for (int i = 0; i < 50; ++i) { NewItem n; n.content = filler(i); vc.append(n); }
        NewItem q; q.content = "Use the zebra_quux setting we discussed earlier."; uint64_t qid = vc.append(q);
        SelectionRequest r; r.query = q.content; r.budget_tokens = 800; r.current_items = {qid}; r.requester = me;
        auto ws = vc.select(r);
        CHECK(ws.references_earlier && !ws.unavailable.empty(), "required_context_unavailable raised");
    }
    {   std::printf("[8] pinned context stays resident; recent tail always present\n");
        VirtualContext vc("c8", me, small_cfg());
        NewItem p; p.content = "Coding rule: all public functions must return Result<T>, never throw."; uint64_t pid = vc.append(p); vc.pin(pid);
        for (int i = 0; i < 2000; ++i) { NewItem n; n.content = filler(i); vc.append(n); }
        NewItem q; q.content = "Write the parse function."; uint64_t qid = vc.append(q);
        SelectionRequest r; r.query = q.content; r.budget_tokens = 900; r.current_items = {qid}; r.requester = me;
        auto ws = vc.select(r);
        CHECK(has(ws, "must return Result<T>"), "pinned rule resident without any query match");
        CHECK(has(ws, filler(1999)), "recent turn kept");
    }
    {   std::printf("[9] persistence: save, reload, search\n");
        char tmpl[] = "/tmp/strata_vwXXXXXX"; std::string dir = mkdtemp(tmpl);
        auto be = std::make_shared<FilePageBackend>(dir);
        std::string manifest = dir + "/ctx.manifest";
        {
            VirtualContext vc("c9", me, small_cfg(), be);
            NewItem a; a.content = "The retry budget is 5 attempts with exponential backoff.\tTabs\nand newlines survive."; vc.append(a);
            for (int i = 0; i < 300; ++i) { NewItem n; n.content = filler(i); vc.append(n); }
            CHECK(vc.save(manifest), "saved");
        }
        auto vc2 = VirtualContext::load(manifest, small_cfg(), be);
        CHECK(vc2 != nullptr && vc2->stats().items == 301, "restored all items");
        auto h = vc2 ? vc2->search(me, "retry budget backoff", 2) : std::vector<SearchHit>{};
        CHECK(!h.empty() && h[0].text.find("Tabs\nand newlines survive.") != std::string::npos, "restored content is exact");
        std::string rm = "rm -rf '" + dir + "'"; (void)std::system(rm.c_str());
    }
    {   std::printf("[10] KV: reuse only the exactly matching prefix, only for the same model/template\n");
        VirtualContext vc("c10", me, small_cfg());
        KvIdentity id{"m1", "t1"};
        std::vector<int32_t> a = {1, 2, 3, 4, 5, 6}, b = {1, 2, 3, 9, 9};
        auto p0 = vc.plan_kv(id, a);
        CHECK(p0.reusable_prefix == 0, "nothing cached at first");
        auto p1 = vc.plan_kv(id, b);
        CHECK(p1.reusable_prefix == 3 && p1.to_prefill == 2, "shared prefix kept, the rest rebuilt");
        auto p2 = vc.plan_kv(KvIdentity{"m2", "t1"}, b);
        CHECK(p2.reusable_prefix == 0, "different model never reuses KV");
    }
    {   std::printf("[11] stable working set between similar requests; scale/latency\n");
        auto cfg = small_cfg(); cfg.physical_tokens = 8192;
        VirtualContext vc("c11", me, cfg);
        double t0 = now_ms();
        for (int i = 0; i < 40000; ++i) { NewItem n; n.content = filler(i) + " " + std::string(200, 'x'); vc.append(n); }   // ~2.2M tokens
        double ing = now_ms() - t0;
        NewItem q; q.content = "status note 31337 what happened"; uint64_t qid = vc.append(q);
        SelectionRequest r; r.query = q.content; r.budget_tokens = 6000; r.current_items = {qid}; r.requester = me;
        double t1 = now_ms(); auto w1 = vc.select(r); double sel = now_ms() - t1;
        NewItem q2; q2.content = "status note 31337 and what else"; uint64_t q2id = vc.append(q2);
        r.query = q2.content; r.current_items = {q2id};
        auto w2 = vc.select(r);
        std::printf("  logical=%lld tokens, ingest %.0f ms, select %.1f ms, page_in(second)=%zu of %zu\n", (long long)vc.window_info().virtual_used, ing, sel, w2.paged_in, w2.page_ids.size());
        CHECK(sel < 250.0, "select latency stays bounded at ~2M tokens");
        CHECK(w2.paged_in * 4 <= w2.page_ids.size() + 8, "second request reuses most of the first working set");
        CHECK(has(w1, "Status note 31337"), "specific old note found at scale");
    }
    std::printf(g_fail ? "%d FAILED\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
