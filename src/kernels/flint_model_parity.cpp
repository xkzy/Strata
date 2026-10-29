// src/kernels/flint_model_parity.cpp - the whole flint model on the GPU against the Go reference (pkg/densecpu/flint.go).
//
//     FLINT_FIXTURE_DIR=<dir> go test -run TestFlintDumpReference ./pkg/densecpu     (writes flint_reference.bin)
//     build/flint_model_parity <flint.gguf> <dir>/flint_reference.bin [reserve_gib]
//
// The prompt is prefilled, then the Go reference's greedy tokens are fed one at a time (teacher forcing, so a near-tie
// that flips the GPU's argmax cannot derail the comparison).  After every step the GPU's logits at the reference's top-10
// ids are compared with the reference's.  Pass, at every step: the top-1 token agrees (or the two candidates are within
// 0.25 logits in both), and the total variation between the two softmaxes over the reference's top-10 is under 0.5.
// Why not tighter: the activations go through q8_1, and a top-6 router near-tie flips an expert under that noise, after
// which the hidden state moves by a few percent (flint_layer_parity shows each block within ~1% on the reference's own
// inputs).  `go test -run TestFlintQ8Noise ./pkg/densecpu` measures that floor by rounding the CPU reference's activations
// to q8_1 alone: its TV against the exact run is 0.100 0.000 0.339 0.006 0.012 0.000 0.004 0.005 0.035 on the nine
// steps; this program's is 0.090 0.001 0.343 0.005 0.004 0.000 0.010 0.004 0.022.
#include "flint_loader.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace flint_loader;

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <flint.gguf> <flint_reference.bin> [reserve_gib]\n", argv[0]); return 2; }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::printf("no CUDA device\n"); return 77; }
    int dev = 0;  // the card with the most memory (CUDA's device order is not nvidia-smi's)
    size_t best = 0;
    for (int d = 0; d < ndev; ++d) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);
        if (p.totalGlobalMem > best) best = p.totalGlobalMem, dev = d;
    }
    cudaSetDevice(dev);
    { cudaDeviceProp p; cudaGetDeviceProperties(&p, dev); std::printf("device %d: %s, %.1f GiB\n", dev, p.name, p.totalGlobalMem / 1073741824.0); }
    std::FILE* f = std::fopen(argv[2], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
    int32_t hdr[4];
    if (std::fread(hdr, 4, 4, f) != 4 || hdr[0] != 0x464c4e33) { std::fprintf(stderr, "not a flint reference\n"); return 2; }
    const int n_prompt = hdr[1], n_gen = hdr[2], topk = hdr[3];
    std::vector<int32_t> prompt(n_prompt);
    if (std::fread(prompt.data(), 4, n_prompt, f) != (size_t) n_prompt) return 2;
    struct Ref { std::vector<int32_t> id; std::vector<float> lg; };
    std::vector<Ref> ref(n_gen + 1);
    for (Ref& r : ref) {
        r.id.resize(topk);
        r.lg.resize(topk);
        if (std::fread(r.id.data(), 4, topk, f) != (size_t) topk || std::fread(r.lg.data(), 4, topk, f) != (size_t) topk) return 2;
    }
    std::fclose(f);

    GgufFile g(argv[1]);
    const int n_layer = (int) num(g, "flint.block_count"), E = (int) num(g, "flint.embedding_length");
    const int window = (int) num(g, "flint.attention.sliding_window");
    Loader ld(g);
    ld.cache_mode = true;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<FlintLayer> layers;
    for (int l = 0; l < n_layer; ++l) layers.push_back(load_layer(ld, l, window));
    const FlintMat emb = ld.mat("token_embd.weight"), head = ld.mat("output.weight");
    const float* out_norm = ld.f32("output_norm.weight");
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("loaded %d layers in %.1f s; experts in pinned host memory %.2f GiB\n", n_layer, load_s,
                ld.streamed_big / 1073741824.0);

    const int total = n_prompt + n_gen;
    const int KD = (int) num(g, "flint.attention.head_count_kv") * (int) num(g, "flint.attention.key_length");
    std::vector<FlintKV> kv(n_layer);
    for (int l = 0; l < n_layer; ++l) {  // FLINT_PARITY_HALF=1: the full layers keep fp16 K/V, as the engine does
        FlintKV& c = kv[l];
        c.cap = total;
        c.half = std::getenv("FLINT_PARITY_HALF") && !layers[l].local;
        cudaMalloc(&c.k, (size_t) total * KD * (c.half ? 2 : 4));
        cudaMalloc(&c.v, (size_t) total * KD * (c.half ? 2 : 4));
    }
    float *ha = nullptr, *hb = nullptr, *d_logits = nullptr;
    int32_t* d_tok = nullptr;
    cudaMalloc(&ha, (size_t) n_prompt * E * 4);
    cudaMalloc(&hb, (size_t) n_prompt * E * 4);
    cudaMalloc(&d_logits, (size_t) head.n_out * 4);
    cudaMalloc(&d_tok, (size_t) n_prompt * 4);
    const float eps = (float) num(g, "flint.attention.layer_norm_rms_epsilon");
    const size_t emb_row = iq_row_bytes(emb.type, emb.n_in);

    const int n_slot = make_cache(ld, layers, (size_t) ((argc > 3 ? std::atof(argv[3]) : 1.0) * (1ull << 30)));
    std::printf("expert pool: %d slots of %.2f MiB\n", n_slot, ld.cache.slot_bytes / 1048576.0);
    // optional per-block trace of the prompt's last row (argv[4]): where does the GPU's error come from
    std::vector<float> trace;
    if (argc > 4) {
        std::FILE* tf = std::fopen(argv[4], "rb");
        int32_t th[3];
        if (!tf || std::fread(th, 4, 3, tf) != 3 || th[0] != 0x464c4e34 || th[1] != n_layer || th[2] != E) { std::fprintf(stderr, "bad trace file\n"); return 2; }
        trace.resize((size_t) n_layer * E);
        if (std::fread(trace.data(), 4, trace.size(), tf) != trace.size()) return 2;
        std::fclose(tf);
    }
    int failures = 0, pos = 0;
    double worst_tv = 0;
    constexpr double TV_LIMIT = 0.5;  // the exact Go run vs Go with q8_1-rounded activations (TestFlintQ8Noise) reaches 0.34
    std::vector<int32_t> feed = prompt;
    std::vector<float> logits(head.n_out);
    for (int step = 0; step <= n_gen; ++step) {
        const int n = (int) feed.size();
        auto ts = std::chrono::steady_clock::now();
        cudaMemcpy(d_tok, feed.data(), (size_t) n * 4, cudaMemcpyHostToDevice);
        iq_embed_rows(emb.type, emb.w, emb_row, d_tok, n, E, ha, nullptr);
        for (int l = 0; l < n_layer; ++l) {
            flint_layer_forward(layers[l], kv[l], pos, ha, hb, n, nullptr);
            std::swap(ha, hb);
            if (step == 0 && !trace.empty()) {
                std::vector<float> row(E);
                cudaMemcpy(row.data(), ha + (size_t) (n - 1) * E, (size_t) E * 4, cudaMemcpyDeviceToHost);
                double dd = 0, rr = 0;
                for (int i = 0; i < E; ++i) {
                    const double r = trace[(size_t) l * E + i];
                    dd += (row[i] - r) * (row[i] - r);
                    rr += r * r;
                }
                std::printf("  after block %2d: hidden rel err %.4f\n", l, std::sqrt(dd / rr));
            }
        }
        flint_head(ha + (size_t) (n - 1) * E, out_norm, head, eps, d_logits, nullptr);
        cudaMemcpy(logits.data(), d_logits, logits.size() * 4, cudaMemcpyDeviceToHost);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
        pos += n;

        const Ref& r = ref[step];
        int best = 0;
        for (int i = 1; i < head.n_out; ++i) if (logits[i] > logits[best]) best = i;
        double worst = 0;
        for (int k = 0; k < topk; ++k) worst = std::max(worst, (double) std::fabs(logits[r.id[k]] - r.lg[k]));
        const double gap = r.lg[0] - r.lg[1];
        const bool top1 = best == r.id[0];
        const bool tie = !top1 && std::fabs(logits[best] - logits[r.id[0]]) < 0.25 && gap < 0.25;
        // total variation between the two softmaxes over the reference's top-k ids (renormalised to those ids)
        double zr = 0, zg = 0;
        for (int k = 0; k < topk; ++k) zr += std::exp((double) r.lg[k] - r.lg[0]), zg += std::exp((double) logits[r.id[k]] - logits[best]);
        double tv = 0;
        for (int k = 0; k < topk; ++k) tv += std::fabs(std::exp((double) r.lg[k] - r.lg[0]) / zr - std::exp((double) logits[r.id[k]] - logits[best]) / zg);
        tv *= 0.5;
        worst_tv = std::max(worst_tv, tv);
        const bool ok = (top1 || tie) && tv < TV_LIMIT;
        std::printf("step %d (%2d tok, %7.0f ms): ref top1 %6d (%.3f, gap %.3f)  gpu top1 %6d (%.3f)  max |dlogit| %.3f, TV over ref top-%d %.3f  %s%s\n",
                    step, n, ms, r.id[0], r.lg[0], gap, best, logits[best], worst, topk, tv, ok ? "ok" : "FAIL", tie ? " (near tie)" : "");
        failures += !ok;
        feed = {r.id[0]};
    }
    std::printf("worst TV %.3f; expert cache: %lld hits, %lld copied in, %lld read from host\n", worst_tv, ld.cache.hits, ld.cache.copies, ld.cache.direct);
    for (FlintKV& c : kv) { cudaFree(c.k); cudaFree(c.v); }
    cudaFree(ha); cudaFree(hb); cudaFree(d_logits); cudaFree(d_tok);
    return failures ? 1 : 0;
}
