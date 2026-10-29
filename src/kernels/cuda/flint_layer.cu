// src/kernels/cuda/flint_layer.cu - one flint block on the GPU. See the header; checked by flint_layer_parity.
#include "strata/kernels/flint_layer.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::kernels {
namespace {

constexpr size_t Q8_1_BLOCK = 36;  // fp16 scale + fp16 sum + 32 int8
size_t iq_row_bytes_q6k(int n_in) { return (size_t) n_in / 256 * 210; }

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "flint_layer: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

__device__ float block_sum(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = v;
    __syncthreads();
    float t = 0;
    for (int i = 0; i < (int) blockDim.x / 32; ++i) t += sh[i];
    __syncthreads();
    return t;
}

// out[r] = x[r] * rsqrt(mean(x^2) + eps) * w[r]
__global__ void rmsnorm_kernel(const float* x, const float* w, float* out, int dim, float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t) blockIdx.x * dim;
    float ss = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    const float k = rsqrtf(block_sum(ss, sh) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) out[(size_t) blockIdx.x * dim + i] = xr[i] * k * w[i];
}

// Per-head rms norm with weight, then (rope layers) neox rotation of pairs (i, i+hd/2). grid (n_tok, heads), hd threads.
// x rows are [n_tok][heads*hd].
__global__ void qk_norm_rope_kernel(float* x, const float* w, int heads, int hd, float eps, bool rope, float base, int pos0) {
    extern __shared__ float v[];
    __shared__ float sh[32];
    float* xh = x + ((size_t) blockIdx.x * heads + blockIdx.y) * hd;
    const int i = threadIdx.x;
    const float xi = xh[i];
    const float k = rsqrtf(block_sum(xi * xi, sh) / hd + eps);
    v[i] = xi * k * w[i];
    __syncthreads();
    if (rope) {
        const int half = hd / 2;
        const int j = i % half;
        const float theta = (float) (pos0 + blockIdx.x) * powf(base, -2.0f * j / hd);
        float s, c;
        sincosf(theta, &s, &c);
        const float a = v[j], b = v[j + half];
        xh[i] = i < half ? a * c - b * s : a * s + b * c;
    } else {
        xh[i] = v[i];
    }
}

// One cache row (n_kv heads of hd floats) of position j: a ring for a local layer, else the device rows then the host rows.
__device__ __forceinline__ size_t kv_row(int j, int cap, bool local, bool& host) {
    host = false;
    if (local) return (size_t) (j % cap);
    if (j < cap) return (size_t) j;
    host = true;
    return (size_t) (j - cap);
}

// kv[row(pos0 + r)] = src[r] for n rows of KD floats (stored as fp16 when half); grid n, any block size
__global__ void kv_store_kernel(void* dk, void* dv, void* hk, void* hv, const float* K, const float* V, int KD, int cap, bool local, int pos0, bool half) {
    bool host;
    const size_t row = kv_row(pos0 + blockIdx.x, cap, local, host);
    void* ok = (host ? hk : dk);
    void* ov = (host ? hv : dv);
    for (int i = threadIdx.x; i < KD; i += blockDim.x) {
        const float a = K[(size_t) blockIdx.x * KD + i], b = V[(size_t) blockIdx.x * KD + i];
        if (half) {
            ((__half*) ok)[row * KD + i] = __float2half(a);
            ((__half*) ov)[row * KD + i] = __float2half(b);
        } else {
            ((float*) ok)[row * KD + i] = a;
            ((float*) ov)[row * KD + i] = b;
        }
    }
}

// Causal attention with a per-head sink, split over positions (flash-decoding): block (token, kv head, split) serves the 8 query
// heads that share the kv head, so each K/V row is read once for all of them. It reads its share of the span in tiles of 32
// positions and keeps, per head, a running softmax (max m, sum l, weighted values o); attn_combine_kernel merges the splits and the sink.
// grid (n_tok, n_kv, splits), 256 threads; 8 heads per kv head and head_dim 128 (checked by the caller). Q holds the new tokens,
// so the query of token t sits at position pos0 + t. Partials are stored per (token, head, split).
#define ATTN_TJ 32
#define ATTN_HD 128
__global__ void attn_part_kernel(const float* Q, const FlintKV kv, int n_head, int n_kv, int window, bool local, int pos0,
                                 int splits, float* pm, float* pl, float* po) {
    constexpr int hd = ATTN_HD, KS = hd + 1;
    __shared__ float qs[8 * KS], ks[ATTN_TJ * KS], red[8][8][ATTN_TJ], ps[ATTN_TJ][8], corr_s[8];
    __shared__ const char *kp[ATTN_TJ], *vp[ATTN_TJ];
    const int tq = blockIdx.x, kvh = blockIdx.y, sp = blockIdx.z;
    const int t = pos0 + tq;
    const int lo = (local && t - window + 1 > 0) ? t - window + 1 : 0;
    const int per = (t - lo + 1 + splits - 1) / splits;
    const int a = lo + sp * per, b = min(t + 1, a + per);
    const int tid = threadIdx.x, lane = tid & 31, g = tid >> 5;
    const size_t esz = kv.half ? 2 : 4;
    const float scale = rsqrtf((float) hd);
    const float* q0 = Q + ((size_t) tq * n_head + kvh * 8) * hd;
    for (int i = tid; i < 8 * hd; i += 256) qs[(i / hd) * KS + i % hd] = q0[i];
    // scores: thread = 4 positions (pq) x 2 heads (hp) x 16 dims (dc); values: thread = 2 dims (dp) x 8 heads, every 4th position (jg)
    const int pq = tid & 7, hp = (tid >> 3) & 3, dc = tid >> 5, dp = tid & 63, jg = tid >> 6;
    float o[8][2] = {}, m = -INFINITY, l = 0;
    for (int base = a; base < b; base += ATTN_TJ) {
        const int cnt = min(ATTN_TJ, b - base);
        if (tid < cnt) {
            bool host;
            const size_t off = (kv_row(base + tid, kv.cap, local, host) * n_kv + kvh) * hd * esz;
            kp[tid] = (const char*) (host ? kv.hk : kv.k) + off;
            vp[tid] = (const char*) (host ? kv.hv : kv.v) + off;
        }
        __syncthreads();
        for (int i = tid; i < cnt * (hd / 8); i += 256) {  // 8 values per load
            const int r = i / (hd / 8), c = i % (hd / 8);
            float* dst = ks + r * KS + c * 8;
            if (kv.half) {
                const uint4 u = ((const uint4*) kp[r])[c];
                const __half2* hh = (const __half2*) &u;
                for (int q = 0; q < 4; ++q) {
                    const float2 f = __half22float2(hh[q]);
                    dst[2 * q] = f.x;
                    dst[2 * q + 1] = f.y;
                }
            } else {
                const float4 u0 = ((const float4*) kp[r])[2 * c], u1 = ((const float4*) kp[r])[2 * c + 1];
                dst[0] = u0.x; dst[1] = u0.y; dst[2] = u0.z; dst[3] = u0.w;
                dst[4] = u1.x; dst[5] = u1.y; dst[6] = u1.z; dst[7] = u1.w;
            }
        }
        __syncthreads();
        float acc[2][4] = {};
        for (int i = 0; i < 16; ++i) {
            const int d = 16 * dc + i;
            const float k0 = ks[(4 * pq) * KS + d], k1 = ks[(4 * pq + 1) * KS + d], k2 = ks[(4 * pq + 2) * KS + d], k3 = ks[(4 * pq + 3) * KS + d];
            const float q0v = qs[(2 * hp) * KS + d], q1v = qs[(2 * hp + 1) * KS + d];
            acc[0][0] += q0v * k0; acc[0][1] += q0v * k1; acc[0][2] += q0v * k2; acc[0][3] += q0v * k3;
            acc[1][0] += q1v * k0; acc[1][1] += q1v * k1; acc[1][2] += q1v * k2; acc[1][3] += q1v * k3;
        }
        for (int c = 0; c < 2; ++c)
            for (int r = 0; r < 4; ++r) red[dc][2 * hp + c][4 * pq + r] = acc[c][r];
        __syncthreads();
        // softmax: warp g is head g, lane the position in the tile
        float s = -INFINITY;
        if (lane < cnt) {
            float sum = 0;
            for (int c = 0; c < 8; ++c) sum += red[c][g][lane];
            s = sum * scale;
        }
        float tm = s;
        for (int off = 16; off > 0; off >>= 1) tm = fmaxf(tm, __shfl_xor_sync(0xffffffff, tm, off));
        const float mn = fmaxf(m, tm), corr = expf(m - mn);  // m = -inf the first time: corr = 0
        const float p = lane < cnt ? expf(s - mn) : 0.0f;
        float ts = p;
        for (int off = 16; off > 0; off >>= 1) ts += __shfl_xor_sync(0xffffffff, ts, off);
        ps[lane][g] = p;
        if (lane == 0) corr_s[g] = corr;
        l = l * corr + ts;
        m = mn;
        __syncthreads();
        for (int h = 0; h < 8; ++h) { o[h][0] *= corr_s[h]; o[h][1] *= corr_s[h]; }
        for (int j = jg; j < cnt; j += 4) {
            float2 v;
            if (kv.half) v = __half22float2(((const __half2*) vp[j])[dp]);
            else v = ((const float2*) vp[j])[dp];
            const float4 pa = *(const float4*) &ps[j][0], pb = *(const float4*) &ps[j][4];
            const float pj[8] = {pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w};
            for (int h = 0; h < 8; ++h) { o[h][0] += pj[h] * v.x; o[h][1] += pj[h] * v.y; }
        }
        __syncthreads();
    }
    // sum the 4 position groups through smem (ks is free), then write the partials
    if (jg > 0)
        for (int h = 0; h < 8; ++h) { ks[((jg - 1) * 8 + h) * hd + 2 * dp] = o[h][0]; ks[((jg - 1) * 8 + h) * hd + 2 * dp + 1] = o[h][1]; }
    __syncthreads();
    if (jg == 0)
        for (int h = 0; h < 8; ++h) {
            float x = o[h][0], y = o[h][1];
            for (int k = 0; k < 3; ++k) { x += ks[(k * 8 + h) * hd + 2 * dp]; y += ks[(k * 8 + h) * hd + 2 * dp + 1]; }
            const size_t slot = ((size_t) tq * n_head + kvh * 8 + h) * splits + sp;
            po[slot * hd + 2 * dp] = x;
            po[slot * hd + 2 * dp + 1] = y;
        }
    if (lane == 0) {
        const size_t slot = ((size_t) tq * n_head + kvh * 8 + g) * splits + sp;
        pm[slot] = m;
        pl[slot] = l;
    }
}

__global__ void attn_combine_kernel(const float* sinks, const float* pm, const float* pl, const float* po, float* out, int n_head, int hd, int splits) {
    const int tq = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const size_t slot0 = ((size_t) tq * n_head + h) * splits;
    float mx = sinks[h];
    for (int s = 0; s < splits; ++s) mx = fmaxf(mx, pm[slot0 + s]);
    float den = expf(sinks[h] - mx), num = 0;
    for (int s = 0; s < splits; ++s) {
        const float w = expf(pm[slot0 + s] - mx);  // an empty split has m = -inf and weighs nothing
        den += w * pl[slot0 + s];
        num += w * po[(slot0 + s) * hd + d];
    }
    out[((size_t) tq * n_head + h) * hd + d] = num / den;
}

__global__ void add_kernel(float* dst, const float* a, const float* b, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] + (b ? b[i] : 0.0f);
}

// g[i] = silu(g[i]) * u[i]
__global__ void swiglu_kernel(float* g, const float* u, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}

// acc[i] += w * x[i]
__global__ void axpy_kernel(float* acc, const float* x, float w, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += w * x[i];
}

// logits[t][e] = bf16 router row e . x[t]; one block per (token, expert)
__global__ void router_kernel(const uint16_t* W, const float* x, float* logits, int dim, int n_exp) {
    __shared__ float sh[32];
    const int t = blockIdx.x, e = blockIdx.y;
    float s = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        s += __uint_as_float((uint32_t) W[(size_t) e * dim + i] << 16) * x[(size_t) t * dim + i];
    s = block_sum(s, sh);
    if (threadIdx.x == 0) logits[(size_t) t * n_exp + e] = s;
}

// sigmoid scores, top n_used (ties: lowest index), optionally normalized to sum 1, times scale. One 256-thread block per
// token: n_used rounds of a block-wide arg-max over the scores not yet taken (n_exp <= 1024).
__global__ void topk_kernel(const float* logits, int n_exp, int n_used, bool norm, float scale, int* ids, float* wts,
                            int n_tok) {
    __shared__ float sc[1024], rv[256];
    __shared__ int ri[256];
    __shared__ int chosen[32];
    __shared__ float chosen_w[32];
    const int t = blockIdx.x, tid = threadIdx.x;
    const float* lg = logits + (size_t) t * n_exp;
    for (int e = tid; e < n_exp; e += blockDim.x) sc[e] = 1.0f / (1.0f + expf(-lg[e]));
    __syncthreads();
    float sum = 0;
    for (int k = 0; k < n_used; ++k) {
        float bv = -1.0f;
        int bi = -1;
        for (int e = tid; e < n_exp; e += blockDim.x)
            if (sc[e] > bv) bv = sc[e], bi = e;  // ascending e: the lowest index wins a tie
        rv[tid] = bv, ri[tid] = bi;
        __syncthreads();
        for (int d = blockDim.x / 2; d > 0; d >>= 1) {
            if (tid < d) {
                const float ov = rv[tid + d];
                const int oi = ri[tid + d];
                if (ov > rv[tid] || (ov == rv[tid] && oi >= 0 && (ri[tid] < 0 || oi < ri[tid]))) rv[tid] = ov, ri[tid] = oi;
            }
            __syncthreads();
        }
        if (tid == 0) {
            chosen[k] = ri[0], chosen_w[k] = rv[0];
            if (ri[0] >= 0) sc[ri[0]] = -2.0f;  // taken
        }
        sum += rv[0];
        __syncthreads();
    }
    if (tid < n_used) {
        ids[t * n_used + tid] = chosen[tid];
        wts[t * n_used + tid] = (norm ? chosen_w[tid] / sum : chosen_w[tid]) * scale;
    }
}

// y[c][r] = W[r] . x[c] for Q6_K rows (the one type iq_mmvq has no kernel for; used by a single layer's down experts).
// fp32 activations, one thread per output row: a correctness path, not a fast one.
__global__ void q6k_matvec_kernel(const uint8_t* W, size_t row_bytes, const float* x, float* y, int n_in, int n_out, int ncols) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_out) return;
    for (int c = 0; c < ncols; ++c) {
        float acc = 0;
        const float* xc = x + (size_t) c * n_in;
        for (int b = 0; b < n_in / 256; ++b) {
            const uint8_t* blk = W + (size_t) r * row_bytes + (size_t) b * 210;
            const float d = __half2float(*(const __half*) (blk + 208));
            const uint8_t *ql = blk, *qh = blk + 128;
            const int8_t* sc = (const int8_t*) (blk + 192);
            const float* xb = xc + b * 256;
            for (int half = 0; half < 2; ++half) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    acc += d * (sc[is] * q1 * xb[l] + sc[is + 2] * q2 * xb[l + 32] + sc[is + 4] * q3 * xb[l + 64] + sc[is + 6] * q4 * xb[l + 96]);
                }
                xb += 128; ql += 64; qh += 32; sc += 8;
            }
        }
        y[(size_t) c * n_out + r] = acc;
    }
}

unsigned grid(long long n) { return (unsigned) ((n + 255) / 256); }

// Temporaries for one call, bumped out of one buffer that is kept between calls (a cudaMalloc per matmul used to cost more
// than the matmul). The buffer only changes size between calls, once the stream is idle.
struct Scratch {
    static char* base;
    static size_t cap;
    size_t off = 0;
    std::vector<void*> extra;  // when the buffer is too small for this call: separate blocks, freed with it
    template <class T> T* get(size_t n) {
        const size_t bytes = (n * sizeof(T) + 255) & ~(size_t) 255;
        void* p = nullptr;
        if (off + bytes <= cap) {
            p = base + off;
        } else if (cudaMalloc(&p, bytes) == cudaSuccess) {
            extra.push_back(p);
        } else {
            std::fprintf(stderr, "flint_layer: out of memory for %zu bytes\n", bytes);
            std::exit(1);
        }
        off += bytes;
        return (T*) p;
    }
    ~Scratch() {
        for (void* p : extra) cudaFree(p);
        if (off > cap) {  // grow for the next call
            cudaFree(base);
            cap = off + off / 4;
            if (cudaMalloc(&base, cap) != cudaSuccess) base = nullptr, cap = 0;
        }
    }
};
char* Scratch::base = nullptr;
size_t Scratch::cap = 0;

// xg[j] = x[tok[j]] for j < count (rows of dim floats)
__global__ void gather_rows_kernel(const float* x, const int* tok, float* xg, int dim) {
    const float* src = x + (size_t) tok[blockIdx.x] * dim;
    float* dst = xg + (size_t) blockIdx.x * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i];
}

// acc[t] += w[p] * y[p] over the n_used pairs p of token t, in the order of pairs: the sum does not depend on which expert ran first
__global__ void combine_kernel(float* acc, const int* pairs, const float* w, const float* y, int n_used, int dim) {
    float* dst = acc + (size_t) blockIdx.x * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float a = dst[i];
        for (int j = 0; j < n_used; ++j) {
            const int p = pairs[(size_t) blockIdx.x * n_used + j];
            a += w[p] * y[(size_t) p * dim + i];
        }
        dst[i] = a;
    }
}

}  // namespace

void flint_layer_forward(const FlintLayer& L, FlintKV& kv, int pos0, const float* h_in, float* h_out, int n, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    const int total = pos0 + n;
    if (pos0 < 0 || (L.local ? n + L.window > kv.cap && total > kv.cap : total > kv.cap + kv.host_cap)) {
        std::fprintf(stderr, "flint_layer: positions %d..%d do not fit a cache of %d (+%d on the host)\n", pos0, total - 1, kv.cap, kv.host_cap);
        std::exit(1);
    }
    // positions one query reads, split so a long context spreads over many blocks
    const int span = (L.local && L.window < total) ? L.window : total;
    const int splits = std::max(1, std::min(64, (span + 511) / 512));
    if (L.n_head != 8 * L.n_kv || L.head_dim != ATTN_HD) {
        std::fprintf(stderr, "flint attention needs 8 query heads per kv head and head_dim %d\n", ATTN_HD);
        std::exit(1);
    }
    const int E = L.n_embd, hd = L.head_dim, QD = L.n_head * hd, KD = L.n_kv * hd;
    Scratch m;
    // y = W x for ncols fp32 columns of length W.n_in at x (column c at x + c*n_in)
    auto matmul = [&](const FlintMat& W, const float* x, float* y, int ncols) {
        if (W.type == 14) {
            q6k_matvec_kernel<<<(W.n_out + 127) / 128, 128, 0, s>>>((const uint8_t*) W.w, iq_row_bytes_q6k(W.n_in), x, y, W.n_in, W.n_out, ncols);
            return;
        }
        void* xq = m.get<char>((size_t) ncols * W.n_in / 32 * Q8_1_BLOCK);
        quantize_q8_1_rows(x, ncols, W.n_in, xq, stream);
        iq_mmvq(W.type, W.w, xq, y, W.n_in, W.n_out, ncols, stream);
    };
    float *xn = m.get<float>((size_t) n * E), *Q = m.get<float>((size_t) n * QD), *K = m.get<float>((size_t) n * KD),
          *V = m.get<float>((size_t) n * KD), *ctx = m.get<float>((size_t) n * QD), *o = m.get<float>((size_t) n * E),
          *h1 = m.get<float>((size_t) n * E);
    rmsnorm_kernel<<<n, 256, 0, s>>>(h_in, L.attn_norm, xn, E, L.eps);
    matmul(L.q, xn, Q, n);
    matmul(L.k, xn, K, n);
    matmul(L.v, xn, V, n);
    qk_norm_rope_kernel<<<dim3(n, L.n_head), hd, hd * sizeof(float), s>>>(Q, L.q_norm, L.n_head, hd, L.eps, L.local, L.rope_base, pos0);
    qk_norm_rope_kernel<<<dim3(n, L.n_kv), hd, hd * sizeof(float), s>>>(K, L.k_norm, L.n_kv, hd, L.eps, L.local, L.rope_base, pos0);
    kv_store_kernel<<<n, 256, 0, s>>>(kv.k, kv.v, kv.hk, kv.hv, K, V, KD, kv.cap, L.local, pos0, kv.half);
    // the partial results of the splits take n * n_head * splits * (hd + 2) floats: done in groups of tokens so a long prefill chunk
    // at a deep position stays within 128 MB
    const int group = (int) std::max<size_t>(1, std::min<size_t>(n, (128ull << 20) / ((size_t) L.n_head * splits * (hd + 2) * sizeof(float))));
    const size_t nslot = (size_t) group * L.n_head * splits;
    float *pm = m.get<float>(nslot), *pl = m.get<float>(nslot), *po = m.get<float>(nslot * hd);
    for (int t0 = 0; t0 < n; t0 += group) {
        const int g = std::min(group, n - t0);
        attn_part_kernel<<<dim3(g, L.n_kv, splits), 256, 0, s>>>(Q + (size_t) t0 * QD, kv, L.n_head, L.n_kv, L.window, L.local, pos0 + t0, splits, pm, pl, po);
        attn_combine_kernel<<<dim3(g, L.n_head), hd, 0, s>>>(L.sinks, pm, pl, po, ctx + (size_t) t0 * QD, L.n_head, hd, splits);
    }
    matmul(L.o, ctx, o, n);
    add_kernel<<<grid((long long) n * E), 256, 0, s>>>(h1, h_in, o, (long long) n * E);
    check("attention");

    float* fx = m.get<float>((size_t) n * E);
    rmsnorm_kernel<<<n, 256, 0, s>>>(h1, L.ffn_norm, fx, E, L.eps);
    float* ff = m.get<float>((size_t) n * E);  // the feed-forward output
    if (!L.moe) {
        const long long nf = (long long) n * L.gate.n_out;
        float *g = m.get<float>(nf), *u = m.get<float>(nf);
        matmul(L.gate, fx, g, n);
        matmul(L.up, fx, u, n);
        swiglu_kernel<<<grid(nf), 256, 0, s>>>(g, u, nf);
        matmul(L.down, g, ff, n);
    } else {
        const long long nf = (long long) n * L.sh_gate.n_out;
        float *g = m.get<float>(nf), *u = m.get<float>(nf);
        matmul(L.sh_gate, fx, g, n);
        matmul(L.sh_up, fx, u, n);
        swiglu_kernel<<<grid(nf), 256, 0, s>>>(g, u, nf);
        matmul(L.sh_down, g, ff, n);

        float* logits = m.get<float>((size_t) n * L.n_exp);
        int* ids = m.get<int>((size_t) n * L.n_used);
        float* wts = m.get<float>((size_t) n * L.n_used);
        router_kernel<<<dim3(n, L.n_exp), 256, 0, s>>>((const uint16_t*) L.router, fx, logits, E, L.n_exp);
        topk_kernel<<<n, 256, 0, s>>>(logits, L.n_exp, L.n_used, L.norm_topk, L.exp_scale, ids, wts, n);
        std::vector<int> hid((size_t) n * L.n_used);
        std::vector<float> hw((size_t) n * L.n_used);
        cudaStreamSynchronize(s);
        cudaMemcpy(hid.data(), ids, hid.size() * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(hw.data(), wts, hw.size() * sizeof(float), cudaMemcpyDeviceToHost);
        const int F = L.gate_exps.n_out;
        const size_t gu_stride = (size_t) F * iq_row_bytes(L.gate_exps.type, L.gate_exps.n_in);
        const size_t up_stride = (size_t) F * iq_row_bytes(L.up_exps.type, L.up_exps.n_in);
        const size_t dn_stride = (size_t) E * (L.down_exps.type == 14 ? iq_row_bytes_q6k(L.down_exps.n_in) : iq_row_bytes(L.down_exps.type, L.down_exps.n_in));
        // group the (token, expert) pairs by expert: each expert's weights are read once for all its tokens, which
        // matters most when they are streamed from host memory
        std::vector<std::vector<int>> toks(L.n_exp);
        std::vector<std::vector<float>> wsel(L.n_exp);
        for (int t = 0; t < n; ++t)
            for (int k = 0; k < L.n_used; ++k) {
                const int e = hid[(size_t) t * L.n_used + k];
                toks[e].push_back(t);
                wsel[e].push_back(hw[(size_t) t * L.n_used + k]);
            }
        std::vector<int> tcat;
        std::vector<float> wcat;
        for (int e = 0; e < L.n_exp; ++e) tcat.insert(tcat.end(), toks[e].begin(), toks[e].end()), wcat.insert(wcat.end(), wsel[e].begin(), wsel[e].end());
        int* d_tok = m.get<int>(tcat.size());
        float* d_w = m.get<float>(wcat.size());
        cudaMemcpyAsync(d_tok, tcat.data(), tcat.size() * sizeof(int), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_w, wcat.data(), wcat.size() * sizeof(float), cudaMemcpyHostToDevice, s);
        float *xg = m.get<float>((size_t) n * E), *eg = m.get<float>((size_t) n * F), *eu = m.get<float>((size_t) n * F), *ed = m.get<float>((size_t) n * L.n_used * E);
        FlintExpertCache* C = L.cache;
        std::vector<const char*> pg(L.n_exp), pu(L.n_exp), pd(L.n_exp);  // where each used expert is read from
        for (int e = 0; e < L.n_exp; ++e) {
            pg[e] = (const char*) L.gate_exps.w + (size_t) e * gu_stride;
            pu[e] = (const char*) L.up_exps.w + (size_t) e * up_stride;
            pd[e] = (const char*) L.down_exps.w + (size_t) e * dn_stride;
        }
        std::vector<char> late(L.n_exp, 0);  // experts whose copy the compute must wait for
        cudaStream_t cs = nullptr;
        if (C) {
            if (!C->copy_stream) {
                cudaStreamCreateWithFlags((cudaStream_t*) &C->copy_stream, cudaStreamNonBlocking);
                C->copied.resize(L.n_exp);
                for (void*& ev : C->copied) cudaEventCreateWithFlags((cudaEvent_t*) &ev, cudaEventDisableTiming);
            }
            cs = (cudaStream_t) C->copy_stream;
            ++C->epoch;
            C->inc *= 1.0001;
            if (C->inc > 1e100) { for (double& v : C->score) v /= C->inc; C->inc = 1; }
            const size_t need = gu_stride + up_stride + dn_stride;
            std::vector<int> missing;
            for (int e = 0; e < L.n_exp; ++e) {
                if (toks[e].empty()) continue;
                const int slot = C->slot_of[(size_t) L.index * C->n_exp + e];
                if (slot < 0) { missing.push_back(e); continue; }
                C->locked_at[slot] = C->epoch, C->score[slot] += C->inc, ++C->hits;
                char* b = C->pool + (size_t) slot * C->slot_bytes;
                pg[e] = b, pu[e] = b + gu_stride, pd[e] = b + gu_stride + up_stride;
            }
            for (int e : missing) {
                int victim = -1;
                if (need <= C->slot_bytes)
                    for (int i = 0; i < C->n_slot; ++i)
                        if (C->locked_at[i] != C->epoch && (victim < 0 || C->score[i] < C->score[victim])) victim = i;
                if (victim < 0) { ++C->direct; continue; }  // no free slot: read it from host memory this once
                if (C->owner[victim] >= 0) C->slot_of[C->owner[victim]] = -1;
                C->owner[victim] = L.index * C->n_exp + e, C->slot_of[(size_t) L.index * C->n_exp + e] = victim;
                C->locked_at[victim] = C->epoch, C->score[victim] = C->inc, ++C->copies;
                char* b = C->pool + (size_t) victim * C->slot_bytes;
                cudaMemcpyAsync(b, (const char*) L.host_gate + (size_t) e * gu_stride, gu_stride, cudaMemcpyHostToDevice, cs);
                cudaMemcpyAsync(b + gu_stride, (const char*) L.host_up + (size_t) e * up_stride, up_stride, cudaMemcpyHostToDevice, cs);
                cudaMemcpyAsync(b + gu_stride + up_stride, (const char*) L.host_down + (size_t) e * dn_stride, dn_stride, cudaMemcpyHostToDevice, cs);
                cudaEventRecord((cudaEvent_t) C->copied[e], cs);
                late[e] = 1;
                pg[e] = b, pu[e] = b + gu_stride, pd[e] = b + gu_stride + up_stride;
            }
        }
        // the experts already on the card go first, so their compute hides the copies of the others (everything the previous
        // layer used has finished: the router read above synchronised the stream)
        std::vector<size_t> offs(L.n_exp);
        std::vector<int> order;
        for (int e = 0, o = 0; e < L.n_exp; ++e) offs[e] = o, o += (int) toks[e].size();
        for (int pass = 0; pass < 2; ++pass)
            for (int e = 0; e < L.n_exp; ++e)
                if (!toks[e].empty() && late[e] == pass) order.push_back(e);
        for (int e : order) {
            const int c = (int) toks[e].size();
            const size_t off = offs[e];
            if (late[e]) cudaStreamWaitEvent(s, (cudaEvent_t) C->copied[e], 0);
            FlintMat G = L.gate_exps, U = L.up_exps, D = L.down_exps;
            G.w = pg[e];
            U.w = pu[e];
            D.w = pd[e];
            gather_rows_kernel<<<c, 256, 0, s>>>(fx, d_tok + off, xg, E);
            matmul(G, xg, eg, c);
            matmul(U, xg, eu, c);
            swiglu_kernel<<<grid((long long) c * F), 256, 0, s>>>(eg, eu, (long long) c * F);
            matmul(D, eg, ed + off * E, c);
        }
        // pairs of each token in expert order, whatever order the experts ran in
        std::vector<int> pairs((size_t) n * L.n_used), np(n, 0);
        for (int e = 0; e < L.n_exp; ++e)
            for (size_t i = 0; i < toks[e].size(); ++i) pairs[(size_t) toks[e][i] * L.n_used + np[toks[e][i]]++] = (int) (offs[e] + i);
        int* d_pairs = m.get<int>(pairs.size());
        cudaMemcpyAsync(d_pairs, pairs.data(), pairs.size() * sizeof(int), cudaMemcpyHostToDevice, s);
        combine_kernel<<<n, 256, 0, s>>>(ff, d_pairs, d_w, ed, L.n_used, E);
    }
    add_kernel<<<grid((long long) n * E), 256, 0, s>>>(h_out, h1, ff, (long long) n * E);
    cudaStreamSynchronize(s);
    check("feed-forward");
}

void flint_head(const float* h_last, const float* norm_w, const FlintMat& head, float eps, float* logits, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    Scratch m;
    float* xn = m.get<float>(head.n_in);
    void* xq = m.get<char>((size_t) head.n_in / 32 * Q8_1_BLOCK);
    rmsnorm_kernel<<<1, 256, 0, s>>>(h_last, norm_w, xn, head.n_in, eps);
    quantize_q8_1_rows(xn, 1, head.n_in, xq, stream);
    iq_mmvq(head.type, head.w, xq, logits, head.n_in, head.n_out, 1, stream);
    cudaStreamSynchronize(s);
    check("head");
}

}  // namespace strata::kernels
