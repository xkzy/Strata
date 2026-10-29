// src/kernels/flint_loader.hpp - reads flint blocks from the GGUF onto the GPU, for flint_layer_parity and
// flint_model_parity (host code only).
//
// Placement: small tensors are always uploaded. The stacked expert tensors ("big") are uploaded while the card has
// `reserve` bytes to spare (and `stream_big` is set) and otherwise are copied to mapped pinned host memory, which the
// kernels read over PCIe. (cudaHostRegister of the file mapping itself is refused on this driver.) Layers are loaded in
// order, so the first layers' experts end up resident.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/flint_layer.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace flint_loader {

using namespace strata;
using namespace strata::kernels;

inline void* upload(const void* p, size_t n) {
    void* d = nullptr;
    if (cudaMalloc(&d, n) != cudaSuccess) { std::fprintf(stderr, "cudaMalloc %zu failed\n", n); std::exit(2); }
    cudaMemcpy(d, p, n, cudaMemcpyHostToDevice);
    return d;
}

inline double num(const GgufFile& g, const char* key) {
    const MetaValue* v = g.get(key);
    if (!v) { std::fprintf(stderr, "missing metadata %s\n", key); std::exit(2); }
    return v->num();
}

struct Loader {
    const GgufFile& g;
    size_t reserve = 0;          // bytes of VRAM left free before big tensors fall back to pinned host memory
    bool stream_big = false;
    bool cache_mode = false;     // every expert tensor goes to pinned host memory; make_cache() adds the device pool
    const void* last_host = nullptr;  // host address of the last tensor raw() put in pinned memory
    FlintExpertCache cache;
    size_t resident_big = 0, streamed_big = 0;
    std::vector<void*> owned, owned_host;
    explicit Loader(const GgufFile& gg) : g(gg) {}
    ~Loader() { for (void* p : owned) cudaFree(p); for (void* p : owned_host) cudaFreeHost(p); }
    Loader(const Loader&) = delete;

    const TensorInfo& need(const std::string& name) {
        const TensorInfo* t = g.find(name);
        if (!t) { std::fprintf(stderr, "missing tensor %s\n", name.c_str()); std::exit(2); }
        return *t;
    }
    size_t bytes(const TensorInfo& t) {
        size_t rows = 1;
        for (size_t i = 1; i < t.shape.size(); ++i) rows *= t.shape[i];
        if (t.type == 30) return rows * t.shape[0] * 2;
        if (t.type == 0) return rows * t.shape[0] * 4;
        if (t.type == 14) return rows * (t.shape[0] / 256 * 210);  // Q6_K: not in iq_row_bytes without STRATA_Q6K_EXPERTS
        return rows * iq_row_bytes((int) t.type, (int64_t) t.shape[0]);
    }
    const void* raw(const std::string& name, bool big = false) {
        const TensorInfo& t = need(name);
        const size_t n = bytes(t);
        if (big && (stream_big || cache_mode)) {
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            if (cache_mode || free_b < n + reserve) {
                void *hp = nullptr, *dp = nullptr;
                if (cudaHostAlloc(&hp, n, cudaHostAllocMapped) != cudaSuccess || cudaHostGetDevicePointer(&dp, hp, 0) != cudaSuccess) {
                    std::fprintf(stderr, "cannot pin %zu bytes of host memory for %s\n", n, name.c_str());
                    std::exit(2);
                }
                std::memcpy(hp, g.tensor_data(t), n);
                owned_host.push_back(hp);
                last_host = hp;
                streamed_big += n;
                return dp;
            }
            resident_big += n;
        }
        void* d = upload(g.tensor_data(t), n);
        owned.push_back(d);
        return d;
    }
    FlintMat mat(const std::string& name, bool big = false) {
        const TensorInfo& t = need(name);
        return {raw(name, big), (int) t.type, (int) t.shape[0], (int) t.shape[1]};
    }
    const float* f32(const std::string& name) {
        const TensorInfo& t = need(name);
        if (t.type != 0) { std::fprintf(stderr, "%s is not F32\n", name.c_str()); std::exit(2); }
        return (const float*) raw(name);
    }
};

/// Block l of the model, with its weights placed by `ld`.
inline FlintLayer load_layer(Loader& ld, int l, int window) {
    const GgufFile& g = ld.g;
    const int E = (int) num(g, "flint.embedding_length");
    const std::string p = "blk." + std::to_string(l) + ".";
    FlintLayer L;
    L.n_embd = E;
    L.head_dim = (int) g.get("flint.attention.key_length")->num();
    L.n_head = (int) num(g, "flint.attention.head_count");
    L.n_kv = (int) num(g, "flint.attention.head_count_kv");
    L.window = window;
    L.eps = (float) num(g, "flint.attention.layer_norm_rms_epsilon");
    L.rope_base = (float) num(g, "flint.rope.freq_base");
    const MetaValue* pat = g.get("flint.attention.sliding_window_pattern");
    L.local = pat && (size_t) l < pat->items.size() && pat->items[l].u != 0;
    L.attn_norm = ld.f32(p + "attn_norm.weight");
    L.q_norm = ld.f32(p + "attn_q_norm.weight");
    L.k_norm = ld.f32(p + "attn_k_norm.weight");
    L.sinks = ld.f32(p + "attn_sinks.weight");
    L.ffn_norm = ld.f32(p + "ffn_norm.weight");
    L.q = ld.mat(p + "attn_q.weight");
    L.k = ld.mat(p + "attn_k.weight");
    L.v = ld.mat(p + "attn_v.weight");
    L.o = ld.mat(p + "attn_output.weight");
    L.moe = l >= (int) num(g, "flint.leading_dense_block_count");
    if (!L.moe) {
        L.gate = ld.mat(p + "ffn_gate.weight");
        L.up = ld.mat(p + "ffn_up.weight");
        L.down = ld.mat(p + "ffn_down.weight");
    } else {
        L.n_exp = (int) num(g, "flint.expert_count");
        L.n_used = (int) num(g, "flint.expert_used_count");
        L.exp_scale = (float) num(g, "flint.expert_weights_scale");
        const MetaValue* nt = g.get("flint.expert_weights_norm");
        L.norm_topk = nt && nt->u != 0;
        L.router = ld.raw(p + "ffn_gate_inp.weight");
        auto stacked = [&](const char* nm) {  // [cols, rows, n_exp] -> per-expert matrix
            const TensorInfo& t = ld.need(p + nm);
            return FlintMat{ld.raw(p + nm, true), (int) t.type, (int) t.shape[0], (int) t.shape[1]};
        };
        L.index = l;
        ld.last_host = nullptr;
        L.gate_exps = stacked("ffn_gate_exps.weight");
        L.host_gate = ld.last_host;
        L.up_exps = stacked("ffn_up_exps.weight");
        L.host_up = ld.last_host;
        L.down_exps = stacked("ffn_down_exps.weight");
        L.host_down = ld.last_host;
        L.sh_gate = ld.mat(p + "ffn_gate_shexp.weight");
        L.sh_up = ld.mat(p + "ffn_up_shexp.weight");
        L.sh_down = ld.mat(p + "ffn_down_shexp.weight");
    }
    return L;
}

/// With ld.cache_mode set while the layers were loaded: gives the rest of the card (all but `margin` bytes) to a pool of
/// expert slots and attaches it to every MoE layer. Returns the number of slots.
inline int make_cache(Loader& ld, std::vector<FlintLayer>& layers, size_t margin) {
    FlintExpertCache& c = ld.cache;
    for (const FlintLayer& L : layers)
        if (L.moe) {
            c.n_exp = L.n_exp;
            const size_t per = (ld.bytes(ld.need("blk." + std::to_string(L.index) + ".ffn_gate_exps.weight")) +
                                ld.bytes(ld.need("blk." + std::to_string(L.index) + ".ffn_up_exps.weight")) +
                                ld.bytes(ld.need("blk." + std::to_string(L.index) + ".ffn_down_exps.weight"))) / L.n_exp;
            c.slot_bytes = std::max(c.slot_bytes, per);
        }
    c.slot_bytes = (c.slot_bytes + 255) & ~(size_t) 255;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    int total = 0;
    for (const FlintLayer& L : layers) total += L.moe ? L.n_exp : 0;
    c.n_slot = (int) std::min<size_t>((free_b > margin ? free_b - margin : 0) / c.slot_bytes, total);
    if (c.n_slot <= 0) return 0;
    if (cudaMalloc(&c.pool, (size_t) c.n_slot * c.slot_bytes) != cudaSuccess) { std::fprintf(stderr, "cannot allocate the expert pool\n"); std::exit(2); }
    ld.owned.push_back(c.pool);
    int max_layer = 0;
    for (const FlintLayer& L : layers) max_layer = std::max(max_layer, L.index);
    c.slot_of.assign((size_t) (max_layer + 1) * c.n_exp, -1);
    c.owner.assign(c.n_slot, -1);
    c.score.assign(c.n_slot, 0.0);
    c.locked_at.assign(c.n_slot, 0);
    c.layer.assign((size_t) max_layer + 2, nullptr);
    c.used_prev.assign((size_t) (max_layer + 1) * c.n_exp, 0);
    c.pend.assign(c.n_slot, 0);
    c.pf_ev.assign((size_t) max_layer + 2, nullptr);
    for (FlintLayer& L : layers) c.layer[L.index] = &L;
    for (FlintLayer& L : layers) if (L.moe) L.cache = &c;
    return c.n_slot;
}

}  // namespace flint_loader
