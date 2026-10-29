// src/kernels/flint_engine.cpp - the flint model as an engine process for `strata serve` (config: strata-flint.json).
//
//     flint_engine --native <flint.gguf> [--max-context N (0 = the model's limit, or what RAM holds)] [--chunk N] [--kv-fp32] [--kv-device-gib G]
//
// Speaks the same stdin/stdout line protocol as cmd/densengine (GEN / STOP / QUIT in; INFO, READY, PP, REUSED, T, DONE,
// ERR out), so the server's tokenizer, chat template and API layers need nothing engine-specific. One request at a time.
// The KV cache is fp32 and positional, so any common prefix of the previous prompt is reused. The weights are placed by
// flint_loader.hpp: experts live on the card while there is room and are otherwise read from pinned host memory.
#include "flint_loader.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <cstdarg>

using namespace flint_loader;

namespace {

// Free RAM in bytes (MemAvailable counts the page cache, which the kernel gives back).
size_t mem_available() {
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    char line[256];
    size_t kb = 0;
    while (f && std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
    if (f) std::fclose(f);
    return kb * 1024;
}

struct Sampling {
    double temp = 0, top_p = 1, min_p = 0, rep = 1, freq = 0, pres = 0;
    int top_k = 0;
    long long seed = 0;
};

void say(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void say(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    std::putchar('\n');
    std::fflush(stdout);
}

// "GEN <max_new> [k=v ...] <id,id,...>"
bool parse_gen(const std::string& line, int& max_new, Sampling& sp, std::vector<int32_t>& ids) {
    std::istringstream in(line);
    std::vector<std::string> f;
    for (std::string w; in >> w;) f.push_back(w);
    if (f.size() < 3) return false;
    max_new = std::atoi(f[1].c_str());
    for (size_t i = 2; i + 1 < f.size(); ++i) {
        const size_t eq = f[i].find('=');
        if (eq == std::string::npos) continue;
        const std::string k = f[i].substr(0, eq);
        const double v = std::atof(f[i].c_str() + eq + 1);
        if (k == "temperature") sp.temp = v;
        else if (k == "top_p") sp.top_p = v;
        else if (k == "top_k") sp.top_k = (int) v;
        else if (k == "min_p") sp.min_p = v;
        else if (k == "penalty_repeat") sp.rep = v;
        else if (k == "penalty_freq") sp.freq = v;
        else if (k == "penalty_present") sp.pres = v;
        else if (k == "seed") sp.seed = (long long) v;
    }
    std::istringstream idl(f.back());
    for (std::string s; std::getline(idl, s, ',');) {
        char* end = nullptr;
        const long v = std::strtol(s.c_str(), &end, 10);
        if (s.empty() || *end) return false;
        ids.push_back((int32_t) v);
    }
    return true;
}

// the same sampler as cmd/densengine: penalties on the tokens generated so far, then greedy or top-k / min-p / top-p
int pick(std::vector<float>& lg, const Sampling& sp, const std::unordered_map<int, int>& hist, std::mt19937_64& rng) {
    for (auto [id, n] : hist) {
        double l = lg[id];
        if (sp.rep != 1 && sp.rep > 0) l = l > 0 ? l / sp.rep : l * sp.rep;
        lg[id] = (float) (l - (sp.freq * n + sp.pres));
    }
    if (sp.temp <= 0) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    int k = sp.top_k;
    if (k < 1 || k > 64) k = 64;
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize(k);
    const double mx = lg[idx[0]];
    std::vector<double> p(k);
    double sum = 0;
    for (int i = 0; i < k; ++i) sum += p[i] = std::exp((lg[idx[i]] - mx) / sp.temp);
    int cut = k;
    double acc = 0;
    for (int i = 0; i < k; ++i) {
        if (sp.min_p > 0 && p[i] / p[0] < sp.min_p) { cut = i; break; }
        acc += p[i] / sum;
        if (sp.top_p > 0 && sp.top_p < 1 && acc >= sp.top_p) { cut = i + 1; break; }
    }
    double s2 = 0;
    for (int i = 0; i < cut; ++i) s2 += p[i];
    double r = std::uniform_real_distribution<double>(0, 1)(rng) * s2;
    for (int i = 0; i < cut; ++i)
        if ((r -= p[i]) <= 0) return idx[i];
    return idx[cut - 1];
}

std::mutex mu;
std::condition_variable cv;
std::deque<std::string> cmds;
std::atomic<bool> stop_req{false};

void die(const char* what, cudaError_t e) {
    say("ERR %s: %s", what, cudaGetErrorString(e));
    std::_Exit(1);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model;
    int ctx = 8192, chunk = 2048;  // a bigger chunk amortises the expert copies: 8000 tokens prefill in 34 s at 2048, 46 s at 512, 60 s at 256
    double kv_dev_gib = 1.5;  // device memory for the first positions of the full-attention layers; the rest of the context sits in host RAM
    bool kv_fp32 = false;     // full layers keep fp32 K/V instead of fp16
    double margin_gib = 1.0;  // left free after the expert pool: per-call temporaries
    for (int i = 1; i < argc; ++i) {  // the server adds flags of its own; the ones not named here are skipped
        const std::string a = argv[i];
        const bool val = i + 1 < argc;
        if (a == "--native" && val) model = argv[++i];
        else if (a == "--max-context" && val) ctx = std::atoi(argv[++i]);
        else if (a == "--chunk" && val) chunk = std::atoi(argv[++i]);
        else if (a == "--kv-fp32") kv_fp32 = true;
        else if (a == "--kv-device-gib" && val) kv_dev_gib = std::atof(argv[++i]);
        else if (a == "--margin-gib" && val) margin_gib = std::atof(argv[++i]);
    }
    if (model.empty()) { std::fprintf(stderr, "flint_engine: --native <model.gguf> is required\n"); return 2; }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { say("ERR no CUDA device"); return 1; }
    int dev = 0;  // the card with the most memory
    size_t best = 0;
    for (int d = 0; d < ndev; ++d) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);
        if (p.totalGlobalMem > best) best = p.totalGlobalMem, dev = d;
    }
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    GgufFile g(model);
    const int n_layer = (int) num(g, "flint.block_count"), E = (int) num(g, "flint.embedding_length");
    const int window = (int) num(g, "flint.attention.sliding_window");
    const int KD = (int) num(g, "flint.attention.head_count_kv") * (int) num(g, "flint.attention.key_length");
    const float eps = (float) num(g, "flint.attention.layer_norm_rms_epsilon");

    // The context-sized buffers go first, so the loader sees what is left. A sliding-window layer keeps a ring of window + chunk
    // rows. A full layer keeps its first positions on the card (kv_dev_gib across those layers) and the rest in pinned host RAM,
    // which the attention kernel reads over the bus; --max-context 0 (or past the model's limit) means as much as RAM holds.
    const MetaValue* pat = g.get("flint.attention.sliding_window_pattern");
    int n_full = 0;
    for (int l = 0; l < n_layer; ++l) n_full += !(pat && (size_t) l < pat->items.size() && pat->items[l].u != 0);
    const size_t row = (size_t) KD * (kv_fp32 ? 4 : 2) * 2;  // K and V of one full layer's position
    const int model_ctx = (int) num(g, "flint.context_length");
    if (ctx <= 0 || ctx > model_ctx) ctx = model_ctx;
    const size_t ram_free = mem_available(), file_bytes = std::filesystem::file_size(model);
    const size_t ram_kv = ram_free > file_bytes + (4ull << 30) ? ram_free - file_bytes - (4ull << 30) : 0;  // the experts are pinned too
    const int ram_ctx = (int) std::min<size_t>(ram_kv / (row * n_full), (size_t) model_ctx);
    if (ctx > ram_ctx) {
        std::fprintf(stderr, "flint_engine: context %d lowered to %d: the rest of the KV cache would not fit in RAM\n", ctx, ram_ctx);
        ctx = ram_ctx;
    }
    const int dev_cap = (int) std::min<size_t>(ctx, (size_t) (kv_dev_gib * (1ull << 30)) / (row * n_full));
    std::vector<FlintKV> kv(n_layer);
    for (int l = 0; l < n_layer; ++l) {
        FlintKV& c = kv[l];
        const bool local = pat && (size_t) l < pat->items.size() && pat->items[l].u != 0;
        c.cap = local ? std::min(ctx, window + chunk) : dev_cap;
        c.host_cap = local ? 0 : ctx - dev_cap;
        c.half = !local && !kv_fp32;
        const size_t esz = c.half ? 2 : 4, db = (size_t) c.cap * KD * esz, hb = (size_t) c.host_cap * KD * esz;
        if (cudaMalloc(&c.k, db) != cudaSuccess || cudaMalloc(&c.v, db) != cudaSuccess) {
            say("ERR the KV cache does not fit on %s; lower --kv-device-gib", prop.name);
            return 1;
        }
        if (hb && (cudaHostAlloc(&c.hk, hb, cudaHostAllocMapped) != cudaSuccess || cudaHostAlloc(&c.hv, hb, cudaHostAllocMapped) != cudaSuccess)) {
            say("ERR %.1f GiB of pinned host memory for the KV cache is not available; lower --max-context", n_full * 2.0 * hb / 1073741824.0);
            return 1;
        }
    }
    std::fprintf(stderr, "flint_engine: context %d: %d full layers keep %d positions on the card (%.2f GiB) and %d in host RAM (%.2f GiB)\n", ctx,
                 n_full, dev_cap, n_full * (double) dev_cap * row / 1073741824.0, ctx - dev_cap, n_full * (double) (ctx - dev_cap) * row / 1073741824.0);
    Loader ld(g);
    ld.cache_mode = true;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<FlintLayer> layers;
    for (int l = 0; l < n_layer; ++l) layers.push_back(load_layer(ld, l, window));
    const FlintMat emb = ld.mat("token_embd.weight"), head = ld.mat("output.weight");
    const float* out_norm = ld.f32("output_norm.weight");
    const size_t emb_row = iq_row_bytes(emb.type, emb.n_in);
    float *ha = nullptr, *hb = nullptr, *d_logits = nullptr;
    int32_t* d_tok = nullptr;
    cudaMalloc(&ha, (size_t) chunk * E * 4);
    cudaMalloc(&hb, (size_t) chunk * E * 4);
    cudaMalloc(&d_logits, (size_t) head.n_out * 4);
    cudaMalloc(&d_tok, (size_t) chunk * 4);
    std::fprintf(stderr, "flint_engine: %s, %d layers loaded in %.1f s; experts on the card %.2f GiB, read from host %.2f GiB\n",
                 prop.name, n_layer, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                 ld.resident_big / 1073741824.0, ld.streamed_big / 1073741824.0);

    const int n_slot = make_cache(ld, layers, (size_t) (margin_gib * (1ull << 30)));
    std::fprintf(stderr, "flint_engine: expert pool of %d slots (%.2f GiB) for %d experts\n", n_slot, n_slot * (double) ld.cache.slot_bytes / 1073741824.0,
                 (n_layer - 2) * ld.cache.n_exp);
    std::vector<float> logits(head.n_out);
    // runs ids[0..n) at positions pos.. and leaves the last row's logits in `logits`
    auto step = [&](const int32_t* ids, int n, int pos) {
        cudaMemcpy(d_tok, ids, (size_t) n * 4, cudaMemcpyHostToDevice);
        iq_embed_rows(emb.type, emb.w, emb_row, d_tok, n, E, ha, nullptr);
        for (int l = 0; l < n_layer; ++l) {
            flint_layer_forward(layers[l], kv[l], pos, ha, hb, n, nullptr);
            std::swap(ha, hb);
        }
        flint_head(ha + (size_t) (n - 1) * E, out_norm, head, eps, d_logits, nullptr);
        const cudaError_t e = cudaMemcpy(logits.data(), d_logits, logits.size() * 4, cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) die("CUDA", e);
    };

    say("INFO engine=flint model=flint device=%s context=%d", prop.name, ctx);
    say("READY %d stop", ctx);

    std::thread([] {
        for (std::string l; std::getline(std::cin, l);) {
            if (l == "STOP") stop_req = true;
            else if (l == "QUIT") std::_Exit(0);
            else {
                std::lock_guard<std::mutex> lk(mu);
                cmds.push_back(std::move(l));
                cv.notify_one();
            }
        }
        std::_Exit(0);  // the server went away
    }).detach();

    std::vector<int32_t> seen;  // the tokens whose K/V are in the cache, in order
    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [] { return !cmds.empty(); });
            line = std::move(cmds.front());
            cmds.pop_front();
        }
        stop_req = false;
        if (line.compare(0, 4, "GEN ") != 0) continue;
        int max_new = 0;
        Sampling sp;
        std::vector<int32_t> ids;
        if (!parse_gen(line, max_new, sp, ids) || ids.empty() || (int) ids.size() + 1 > ctx) {
            say("ERR bad request (prompt %zu tokens, context %d)", ids.size(), ctx);
            continue;
        }
        const auto ts = std::chrono::steady_clock::now();
        const auto ms_since = [&] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count(); };
        size_t common = 0;
        while (common < seen.size() && common < ids.size() && seen[common] == ids[common]) ++common;
        common = std::min(common, ids.size() - 1);  // the last prompt token is always run: it yields the first logits
        seen.resize(common);
        if (common > 0) say("REUSED %zu", common);
        for (size_t p = common; p < ids.size() && !stop_req;) {
            const int n = (int) std::min<size_t>(chunk, ids.size() - p);
            step(ids.data() + p, n, (int) p);
            seen.insert(seen.end(), ids.begin() + p, ids.begin() + p + n);
            p += n;
            say("PP %zu %zu %.3f %.1f", p, ids.size(), (double) p / ids.size(), (p - common) / (ms_since() / 1000));
        }
        const double prefill = ms_since();
        std::mt19937_64 rng(sp.seed ? (uint64_t) sp.seed : std::random_device{}());
        std::unordered_map<int, int> hist;
        int n = 0;
        const char* reason = "length";
        if (seen.size() < ids.size()) reason = "stop";  // stopped during the prefill; the cache holds what was run
        else {
            for (; n < max_new; ++n) {
                if (stop_req) { reason = "stop"; break; }
                if ((int) seen.size() + 1 >= ctx) break;
                const int id = pick(logits, sp, hist, rng);
                ++hist[id];
                say("T %d", id);
                const int32_t t = id;
                step(&t, 1, (int) seen.size());
                seen.push_back(t);
            }
        }
        { const FlintExpertCache& c = ld.cache; std::fprintf(stderr, "flint_engine: expert cache: %lld hits, %lld copied in, %lld read from host\n", c.hits, c.copies, c.direct); }
        say("DONE %d %zu %lld %lld %s", n, ids.size(), (long long) prefill, (long long) (ms_since() - prefill), reason);
    }
}
