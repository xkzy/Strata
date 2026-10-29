// include/strata/kernels/flint_layer.hpp - one flint block on the GPU, for the parity harness (flint_layer_parity).
//
// The reference is pkg/densecpu/flint.go: windowed rope layers and full no-rope layers, both with per-head sinks, then a
// dense SwiGLU or shared + sigmoid-top-k routed experts. Weights stay in their GGUF types (the projections go through
// iq_mmvq with q8_1 activations); everything else is fp32. This is the correctness step before an engine integration:
// it runs n_tok new tokens at positions pos0.. (a prefill chunk or one decode token) against a fp32 K/V cache, and
// allocates its temporaries per call.
#pragma once

#include <cstdint>
#include <vector>

namespace strata::kernels {

/// One 2-D weight on the device: n_out rows of n_in values of ggml type `type`.
struct FlintMat {
    const void* w = nullptr;
    int type = 0, n_in = 0, n_out = 0;
};

/// A pool of device slots that keeps the experts used most, over all layers. The experts themselves live in host memory
/// (FlintLayer::host_*); a layer call copies the ones it routes to that are not in the pool into the least-used free
/// slots, and reads the rest straight from host memory. Slot = gate | up | down of one expert.
struct FlintLayer;

struct FlintExpertCache {
    char* pool = nullptr;
    size_t slot_bytes = 0;
    int n_slot = 0, n_exp = 0;
    std::vector<int> slot_of;          // layer * n_exp + expert -> slot, -1 when not in the pool
    std::vector<int> owner;            // slot -> layer * n_exp + expert, -1 when empty
    std::vector<double> score;         // decayed use count
    std::vector<uint64_t> locked_at;   // the call (epoch) that last used the slot: it cannot be evicted during that call
    uint64_t epoch = 0;
    double inc = 1;                    // what one use adds; grows a little every call, which ages the older counts
    long long hits = 0, copies = 0, direct = 0;
    void* copy_stream = nullptr;       // cudaStream_t: expert copies run here while the cached experts compute
    std::vector<void*> copied;         // per expert: cudaEvent_t recorded when its copy has finished
    std::vector<const FlintLayer*> layer;  // by layer index: the next layer's experts are copied in while this one computes
    std::vector<char> used_prev;       // layer * n_exp + expert: it had tokens the last time that layer ran (what to prefetch)
    std::vector<char> pend;            // slot -> a prefetch copy into it may still be running
    std::vector<void*> pf_ev;          // by layer index: cudaEvent_t recorded after the prefetch copies for that layer
    // second card: experts kept there for good run there when a call has fewer than 16 tokens; only the activations cross the bus
    int dev1 = 0, dev2 = -1, n_slot2 = 0;
    bool placed2 = false;
    char* pool2 = nullptr;
    std::vector<int> slot2_of;         // layer * n_exp + expert -> slot on the second card, -1 when not there
    std::vector<float> use;            // layer * n_exp + expert -> decayed count of calls that routed to it (ranks what goes where)
    void* stream2 = nullptr;           // cudaStream_t on dev2
    void* ev_x = nullptr;              // cudaEvent_t on dev1: the activations for dev2 are in hx
    float *x2 = nullptr, *g2 = nullptr, *u2 = nullptr, *y2 = nullptr, *hx = nullptr, *hy = nullptr;  // dev2 buffers, pinned host buffers (120 rows)
    char* xq2 = nullptr;
    // decode: the router run early on the layer's input predicts its experts; the missing ones are copied while attention runs
    void* psl = nullptr;               // cudaStream_t: the prediction
    void* pev = nullptr;               // cudaEvent_t: the predicted ids are on the host
    float *pfx = nullptr, *plog = nullptr, *pw = nullptr;
    int *pid = nullptr, *hpid = nullptr;  // hpid: pinned
    long long pred_hit = 0, pred_copied = 0;
};

struct FlintLayer {
    int n_embd = 0, n_head = 0, n_kv = 0, head_dim = 0, window = 0;
    bool local = false;  // sliding window + rope; false: whole context, no rope
    float eps = 0, rope_base = 0;
    const float *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *sinks = nullptr, *ffn_norm = nullptr;
    FlintMat q, k, v, o;
    // dense feed-forward (moe == false)
    FlintMat gate, up, down;
    // moe == true: router is BF16 [n_embd -> n_exp]; the *_exps matrices hold n_exp stacked experts (w = expert 0)
    bool moe = false, norm_topk = false;
    int n_exp = 0, n_used = 0;
    float exp_scale = 1.0f;
    const void* router = nullptr;
    FlintMat gate_exps, up_exps, down_exps;
    FlintExpertCache* cache = nullptr;  // when set, the experts are read through it (host_* are expert 0 in host memory)
    int index = 0;
    const void *host_gate = nullptr, *host_up = nullptr, *host_down = nullptr;
    FlintMat sh_gate, sh_up, sh_down;
};

/// The block's cache: rows of n_kv * head_dim floats, one per position.
/// Full layer: positions 0..cap-1 are rows of k/v (device), positions cap..cap+host_cap-1 are rows of hk/hv (pinned host
/// memory the kernels read and write over the bus); the position limit is cap + host_cap.
/// With half set the rows are fp16 (k/v/hk/hv point at __half data): a full layer's cache is most of the memory and of the bus
/// traffic, and fp16 halves both; a ring is small and stays fp32.
/// Local (sliding-window) layer: a ring of cap rows, position p at row p % cap; cap >= window + the most rows one call adds.
struct FlintKV {
    float *k = nullptr, *v = nullptr;
    int cap = 0;
    float *hk = nullptr, *hv = nullptr;
    int host_cap = 0;
    bool half = false;
};

/// h_out = block(h_in) for n_tok rows of n_embd floats (device pointers, may not alias) at positions pos0 ..
/// pos0+n_tok-1; the cache must already hold positions 0..pos0-1 and gets the new rows. pos0 + n_tok <= cap + host_cap for a full layer.
void flint_layer_forward(const FlintLayer& L, FlintKV& kv, int pos0, const float* h_in, float* h_out, int n_tok, void* stream);

/// logits = head(rmsnorm(h_last) * norm_w) for one row of head.n_in floats; logits has head.n_out floats.
void flint_head(const float* h_last, const float* norm_w, const FlintMat& head, float eps, float* logits, void* stream);

}  // namespace strata::kernels
