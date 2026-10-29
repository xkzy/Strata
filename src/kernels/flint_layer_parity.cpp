// src/kernels/flint_layer_parity.cpp - one flint block on the GPU against the Go reference (pkg/densecpu/flint.go).
//
//     FLINT_FIXTURE_DIR=<dir> go test -run TestFlintDumpLayers ./pkg/densecpu     (writes flint_layer_<L>_w<W>.bin)
//     build/flint_layer_parity <flint.gguf> <dir>/flint_layer_*.bin ...
//
// A fixture is int32 {magic, layer, n_embd, window, steps}, then per step int32 n_tok and the block's input and its Go
// output as fp32 rows: a prefill chunk, then single decode tokens, the block keeping its K/V cache between steps.  The
// weights are read from the GGUF.  Projections run through q8_1 activations, so the GPU is not bitwise equal; the check is the
// relative L2 error of what the block ADDED (out - in) - the residual stream alone would hide a wrong block.
#include "flint_loader.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace flint_loader;

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <flint.gguf> <fixture.bin>...\n", argv[0]); return 2; }
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) { std::printf("no CUDA device\n"); return 77; }
    GgufFile g(argv[1]);
    int failures = 0;
    for (int a = 2; a < argc; ++a) {
        std::FILE* f = std::fopen(argv[a], "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[a]); return 2; }
        int32_t hdr[5];
        if (std::fread(hdr, 4, 5, f) != 5 || hdr[0] != 0x464c4e32) { std::fprintf(stderr, "%s: not a flint fixture\n", argv[a]); return 2; }
        const int l = hdr[1], E = hdr[2], window = hdr[3], steps = hdr[4];
        struct Step { int n; std::vector<float> in, ref; };
        std::vector<Step> seq(steps);
        int total = 0;
        for (Step& st : seq) {
            int32_t n = 0;
            if (std::fread(&n, 4, 1, f) != 1) return 2;
            st.n = n;
            st.in.resize((size_t) n * E);
            st.ref.resize((size_t) n * E);
            if (std::fread(st.in.data(), 4, st.in.size(), f) != st.in.size() || std::fread(st.ref.data(), 4, st.ref.size(), f) != st.ref.size()) return 2;
            total += n;
        }
        std::fclose(f);

        Loader ld(g);
        FlintLayer L = load_layer(ld, l, window);
        // FLINT_PARITY_SPLIT=1 runs the tiers: a full layer keeps half its positions in pinned host memory, a local one a ring
        FlintKV kv;
        const bool split = std::getenv("FLINT_PARITY_SPLIT") != nullptr;
        int maxn = 0;
        for (const Step& st : seq) maxn = std::max(maxn, st.n);
        kv.cap = total;
        if (split && L.local) kv.cap = std::min(total, window + maxn);
        else if (split) kv.cap = (total + 1) / 2, kv.host_cap = total - kv.cap;
        kv.half = std::getenv("FLINT_PARITY_HALF") && !L.local;
        const size_t esz = kv.half ? 2 : 4;
        const size_t kvb = (size_t) kv.cap * L.n_kv * L.head_dim * esz, hkb = (size_t) kv.host_cap * L.n_kv * L.head_dim * esz;
        cudaMalloc(&kv.k, kvb);
        cudaMalloc(&kv.v, kvb);
        if (hkb) cudaHostAlloc(&kv.hk, hkb, cudaHostAllocMapped), cudaHostAlloc(&kv.hv, hkb, cudaHostAllocMapped);
        float *din = nullptr, *dout = nullptr;
        cudaMalloc(&din, (size_t) total * E * 4);
        cudaMalloc(&dout, (size_t) total * E * 4);
        int pos = 0;
        double worst = 0, worst_hidden = 0;
        std::string per_step;
        for (const Step& st : seq) {
            cudaMemcpy(din, st.in.data(), st.in.size() * 4, cudaMemcpyHostToDevice);
            flint_layer_forward(L, kv, pos, din, dout, st.n, nullptr);
            std::vector<float> got(st.in.size());
            cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
            pos += st.n;
            double dd = 0, rr = 0, full = 0, fullref = 0;
            for (size_t i = 0; i < got.size(); ++i) {
                const double dg = got[i] - st.in[i], dr = st.ref[i] - st.in[i];
                dd += (dg - dr) * (dg - dr);
                rr += dr * dr;
                full += ((double) got[i] - st.ref[i]) * ((double) got[i] - st.ref[i]);
                fullref += (double) st.ref[i] * st.ref[i];
            }
            const double rel = std::sqrt(dd / rr);
            worst = std::isfinite(rel) && rel > worst ? rel : (std::isfinite(rel) ? worst : 1e9);
            worst_hidden = std::max(worst_hidden, std::sqrt(full / fullref));
            char b[16];
            std::snprintf(b, sizeof b, " %.4f", rel);
            per_step += b;
        }
        cudaFree(kv.k);
        cudaFree(kv.v);
        if (kv.hk) cudaFreeHost(kv.hk), cudaFreeHost(kv.hv);
        cudaFree(din);
        cudaFree(dout);
        const bool ok = worst < 0.05;
        std::printf("layer %2d window %3d %s %s: %d steps (%d positions), worst delta rel err %.4f, hidden %.5f  %s\n  per step:%s\n",
                    l, window, L.local ? "swa " : "full", L.moe ? "moe  " : "dense", steps, total, worst, worst_hidden, ok ? "ok" : "FAIL", per_step.c_str());
        failures += !ok;
    }
    return failures ? 1 : 0;
}
