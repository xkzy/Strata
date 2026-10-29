// Quantized matrix-vector products for the dense-model reference (pkg/densecpu): y[i][r] = W[r] . x[i].
// One thread block per (output row, input vector); the arithmetic mirrors the CPU kernels in dot.go.
// Build: see build.sh. No peer access, no unified memory: weights are copied once, activations per call.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stddef.h>
#include <stdint.h>

#include "densegpu.h"

enum { T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_BF16 = 30 };

__device__ __forceinline__ float h2f(const uint8_t* p) { return __half2float(*(const __half*) p); }

__device__ __forceinline__ void scale_min(int j, const uint8_t* q, int* d, int* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// A block handles one weight row and up to NB inputs: each weight is decoded once and applied to every input, so a
// prefill chunk reads the matrix once instead of once per token. The dot_* functions add into acc[0..nin).

// K-quants: one work item = 8 consecutive bytes of one 32-byte chunk, i.e. 8 low-nibble and 8 high-nibble weights
// (16 items per 256-element block). Contributions add up, so items are independent.
template <bool FIVE, int NB>
__device__ void dot_k45(const uint8_t* row, const float* x, int cols, int nin, int tid, int nt, float* acc) {
    const int bytes = FIVE ? 176 : 144, nb = cols / 256;
    for (int it = tid; it < nb * 16; it += nt) {
        const int b = it >> 4, j = (it >> 2) & 3, k = it & 3;
        const uint8_t* blk = row + (size_t) b * bytes;
        const float d = h2f(blk), dmin = h2f(blk + 2);
        int s1, m1, s2, m2;
        scale_min(2 * j, blk + 4, &s1, &m1);
        scale_min(2 * j + 1, blk + 4, &s2, &m2);
        const uint2 qv = *(const uint2*) (blk + (FIVE ? 48 : 16) + j * 32 + k * 8);
        uint2 hv = make_uint2(0, 0);
        if (FIVE) hv = *(const uint2*) (blk + 16 + k * 8);
        float v1[8], v2[8];
#pragma unroll
        for (int l = 0; l < 8; l++) {
            const uint32_t qw = l < 4 ? qv.x : qv.y, hw = l < 4 ? hv.x : hv.y;
            const uint32_t qb = (qw >> (8 * (l & 3))) & 0xff, hb = (hw >> (8 * (l & 3))) & 0xff;
            v1[l] = (float) (qb & 0xF);
            v2[l] = (float) (qb >> 4);
            if (FIVE) {
                if (hb & (1u << (2 * j))) v1[l] += 16.f;
                if (hb & (2u << (2 * j))) v2[l] += 16.f;
            }
        }
        const int off = b * 256 + j * 64 + k * 8;
#pragma unroll
        for (int ii = 0; ii < NB; ii++) {
            if (ii >= nin) break;
            const float* x1 = x + (size_t) ii * cols + off;
            const float* x2 = x1 + 32;
            float a1 = 0.f, a2 = 0.f, b1 = 0.f, b2 = 0.f;
#pragma unroll
            for (int l = 0; l < 8; l++) {
                const float xa = __ldg(x1 + l), xb2 = __ldg(x2 + l);
                a1 += v1[l] * xa;
                a2 += v2[l] * xb2;
                b1 += xa;
                b2 += xb2;
            }
            acc[ii] += d * ((float) s1 * a1 + (float) s2 * a2) - dmin * ((float) m1 * b1 + (float) m2 * b2);
        }
    }
}

// Q6_K: one work item = 8 consecutive l of one 128-element half (8 items per 256-element block).
template <int NB>
__device__ void dot_q6k(const uint8_t* row, const float* x, int cols, int nin, int tid, int nt, float* acc) {
    const int nb = cols / 256;
    for (int it = tid; it < nb * 8; it += nt) {
        const int b = it >> 3, n = (it >> 2) & 1, g = it & 3;
        const uint8_t* blk = row + (size_t) b * 210;
        const uint8_t* ql = blk + n * 64 + g * 8;
        const uint8_t* qh = blk + 128 + n * 32 + g * 8;
        const int8_t* sc = (const int8_t*) (blk + 192 + n * 8);
        const int is = g >> 1;
        float w0[8], w1[8], w2[8], w3[8];
#pragma unroll
        for (int l = 0; l < 8; l++) {
            const int lo0 = ql[l], lo1 = ql[l + 32], h = qh[l];
            w0[l] = (float) (((lo0 & 0xF) | (((h >> 0) & 3) << 4)) - 32);
            w1[l] = (float) (((lo1 & 0xF) | (((h >> 2) & 3) << 4)) - 32);
            w2[l] = (float) (((lo0 >> 4) | (((h >> 4) & 3) << 4)) - 32);
            w3[l] = (float) (((lo1 >> 4) | (((h >> 6) & 3) << 4)) - 32);
        }
        const float d = h2f(blk + 208);
        const int off = b * 256 + n * 128 + g * 8;
#pragma unroll
        for (int ii = 0; ii < NB; ii++) {
            if (ii >= nin) break;
            const float* xn = x + (size_t) ii * cols + off;
            float a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f;
#pragma unroll
            for (int l = 0; l < 8; l++) {
                a1 += w0[l] * __ldg(xn + l);
                a2 += w1[l] * __ldg(xn + l + 32);
                a3 += w2[l] * __ldg(xn + l + 64);
                a4 += w3[l] * __ldg(xn + l + 96);
            }
            acc[ii] += d * ((float) sc[is] * a1 + (float) sc[is + 2] * a2 + (float) sc[is + 4] * a3 +
                            (float) sc[is + 6] * a4);
        }
    }
}

template <int NB>
__device__ void dot_q8(const uint8_t* row, const float* x, int cols, int nin, int tid, int nt, float* acc) {
    for (int it = tid; it < cols / 8; it += nt) { // 8 weights per item, 4 items per 34-byte block
        const int b = it >> 2, q = it & 3;
        const uint8_t* blk = row + (size_t) b * 34;
        float w[8];
#pragma unroll
        for (int l = 0; l < 8; l++) w[l] = (float) (int8_t) blk[2 + q * 8 + l];
        const float d = h2f(blk);
#pragma unroll
        for (int ii = 0; ii < NB; ii++) {
            if (ii >= nin) break;
            const float* xb = x + (size_t) ii * cols + b * 32 + q * 8;
            float a = 0.f;
#pragma unroll
            for (int l = 0; l < 8; l++) a += w[l] * __ldg(xb + l);
            acc[ii] += d * a;
        }
    }
}

template <int NB>
__device__ void dot_bf16(const uint8_t* row, const float* x, int cols, int nin, int tid, int nt, float* acc) {
    for (int it = tid; it < cols / 8; it += nt) { // 8 weights (16 bytes) per item; cols % 8 == 0
        const uint4 wv = *(const uint4*) (row + (size_t) it * 16);
        const uint32_t u[4] = {wv.x, wv.y, wv.z, wv.w};
        float w[8];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            w[2 * k] = __uint_as_float(u[k] << 16);
            w[2 * k + 1] = __uint_as_float(u[k] & 0xffff0000u);
        }
#pragma unroll
        for (int ii = 0; ii < NB; ii++) {
            if (ii >= nin) break;
            const float* xi = x + (size_t) ii * cols + it * 8;
            float a = 0.f;
#pragma unroll
            for (int l = 0; l < 8; l++) a += w[l] * __ldg(xi + l);
            acc[ii] += a;
        }
    }
}

enum { NB = 8 };

template <int TYPE>
__global__ void matvec(const uint8_t* w, size_t row_bytes, int rows, int cols, int n, const float* x, float* y) {
    const int r = blockIdx.x, i0 = blockIdx.y * NB;
    const int nin = min((int) NB, n - i0);
    const uint8_t* row = w + (size_t) r * row_bytes;
    const float* xi = x + (size_t) i0 * cols;
    float acc[NB];
#pragma unroll
    for (int k = 0; k < NB; k++) acc[k] = 0.f;
    if (TYPE == T_Q4_K) dot_k45<false, NB>(row, xi, cols, nin, threadIdx.x, blockDim.x, acc);
    else if (TYPE == T_Q5_K) dot_k45<true, NB>(row, xi, cols, nin, threadIdx.x, blockDim.x, acc);
    else if (TYPE == T_Q6_K) dot_q6k<NB>(row, xi, cols, nin, threadIdx.x, blockDim.x, acc);
    else if (TYPE == T_Q8_0) dot_q8<NB>(row, xi, cols, nin, threadIdx.x, blockDim.x, acc);
    else dot_bf16<NB>(row, xi, cols, nin, threadIdx.x, blockDim.x, acc);
    __shared__ float part[NB][32];
#pragma unroll
    for (int k = 0; k < NB; k++) {
        float s = acc[k];
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        if ((threadIdx.x & 31) == 0) part[k][threadIdx.x >> 5] = s;
    }
    __syncthreads();
    if (threadIdx.x < nin) {
        float t = 0.f;
        for (int k = 0; k < (int) (blockDim.x >> 5); k++) t += part[threadIdx.x][k];
        y[(size_t) (i0 + threadIdx.x) * rows + r] = t;
    }
}

// ---- int8 path for prefill chunks (n >= 4): inputs are quantized to int8 per 32 values (scale = max|x| / 127) and the
// weights are applied with dp4a, four multiply-adds per instruction and one 32-bit load per four inputs. ----

__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { // p is only 2-byte aligned in Q6_K / Q8_0 rows
    return (uint32_t) *(const uint16_t*) p | ((uint32_t) *(const uint16_t*) (p + 2) << 16);
}

// One warp per (input, 32-value block).
__global__ void quant_x(const float* x, int cols, int n, int8_t* xq, float* xd) {
    const int blocks = cols / 32;
    const int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (wid >= n * blocks) return;
    const float v = x[(size_t) wid * 32 + lane];
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.f;
    xq[(size_t) wid * 32 + lane] = d > 0.f ? (int8_t) __float2int_rn(v / d) : 0;
    if (lane == 0) xd[wid] = d;
}

__device__ __forceinline__ int rep4(int8_t v) { return (int) (uint8_t) v * 0x01010101; }

template <bool FIVE>
__device__ void mmq_k45(const uint8_t* row, const int8_t* xq, const float* xd, int cols, int nin, int tid, int nt,
                        float* acc) {
    const int bytes = FIVE ? 176 : 144, nb = cols / 256, nb32 = cols / 32;
    for (int it = tid; it < nb * 16; it += nt) {
        const int b = it >> 4, j = (it >> 2) & 3, k = it & 3;
        const uint8_t* blk = row + (size_t) b * bytes;
        const float d = h2f(blk), dmin = h2f(blk + 2);
        int s1, m1, s2, m2;
        scale_min(2 * j, blk + 4, &s1, &m1);
        scale_min(2 * j + 1, blk + 4, &s2, &m2);
        const uint2 qv = *(const uint2*) (blk + (FIVE ? 48 : 16) + j * 32 + k * 8);
        uint32_t a0 = qv.x & 0x0F0F0F0Fu, a1 = qv.y & 0x0F0F0F0Fu; // low nibbles: sub-block 2j
        uint32_t c0 = (qv.x >> 4) & 0x0F0F0F0Fu, c1 = (qv.y >> 4) & 0x0F0F0F0Fu; // high nibbles: sub-block 2j+1
        if (FIVE) {
            const uint2 hv = *(const uint2*) (blk + 16 + k * 8);
            a0 |= ((hv.x >> (2 * j)) & 0x01010101u) << 4;
            a1 |= ((hv.y >> (2 * j)) & 0x01010101u) << 4;
            c0 |= ((hv.x >> (2 * j + 1)) & 0x01010101u) << 4;
            c1 |= ((hv.y >> (2 * j + 1)) & 0x01010101u) << 4;
        }
        const int eA = b * 256 + j * 64 + k * 8, bA = b * 8 + 2 * j;
#pragma unroll
        for (int ii = 0; ii < 8; ii++) {
            if (ii >= nin) break;
            const int8_t* xa = xq + (size_t) ii * cols + eA;
            const int xa0 = *(const int*) xa, xa1 = *(const int*) (xa + 4);
            const int xb0 = *(const int*) (xa + 32), xb1 = *(const int*) (xa + 36);
            const int p1 = __dp4a((int) a0, xa0, __dp4a((int) a1, xa1, 0));
            const int p2 = __dp4a((int) c0, xb0, __dp4a((int) c1, xb1, 0));
            const int q1 = __dp4a(0x01010101, xa0, __dp4a(0x01010101, xa1, 0));
            const int q2 = __dp4a(0x01010101, xb0, __dp4a(0x01010101, xb1, 0));
            const float dA = xd[(size_t) ii * nb32 + bA], dB = xd[(size_t) ii * nb32 + bA + 1];
            acc[ii] += d * ((float) s1 * dA * (float) p1 + (float) s2 * dB * (float) p2) -
                       dmin * ((float) m1 * dA * (float) q1 + (float) m2 * dB * (float) q2);
        }
    }
}

__device__ void mmq_q6k(const uint8_t* row, const int8_t* xq, const float* xd, int cols, int nin, int tid, int nt,
                        float* acc) {
    const int nb = cols / 256, nb32 = cols / 32;
    for (int it = tid; it < nb * 8; it += nt) {
        const int b = it >> 3, n = (it >> 2) & 1, g = it & 3;
        const uint8_t* blk = row + (size_t) b * 210;
        const uint8_t* ql = blk + n * 64 + g * 8;
        const uint8_t* qh = blk + 128 + n * 32 + g * 8;
        const int8_t* sc = (const int8_t*) (blk + 192 + n * 8);
        const int is = g >> 1;
        int w[4][2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint32_t l0 = ld32(ql + 4 * h), l1 = ld32(ql + 32 + 4 * h), hh = ld32(qh + 4 * h);
            w[0][h] = __vsubss4((int) ((l0 & 0x0F0F0F0Fu) | (((hh >> 0) & 0x03030303u) << 4)), 0x20202020);
            w[1][h] = __vsubss4((int) ((l1 & 0x0F0F0F0Fu) | (((hh >> 2) & 0x03030303u) << 4)), 0x20202020);
            w[2][h] = __vsubss4((int) (((l0 >> 4) & 0x0F0F0F0Fu) | (((hh >> 4) & 0x03030303u) << 4)), 0x20202020);
            w[3][h] = __vsubss4((int) (((l1 >> 4) & 0x0F0F0F0Fu) | (((hh >> 6) & 0x03030303u) << 4)), 0x20202020);
        }
        const float d = h2f(blk + 208);
        const int off = b * 256 + n * 128 + g * 8, b0 = b * 8 + n * 4;
#pragma unroll
        for (int ii = 0; ii < 8; ii++) {
            if (ii >= nin) break;
            const int8_t* xn = xq + (size_t) ii * cols + off;
            float t = 0.f;
#pragma unroll
            for (int m = 0; m < 4; m++) {
                const int p = __dp4a(w[m][0], *(const int*) (xn + 32 * m), __dp4a(w[m][1], *(const int*) (xn + 32 * m + 4), 0));
                t += (float) sc[is + 2 * m] * xd[(size_t) ii * nb32 + b0 + m] * (float) p;
            }
            acc[ii] += d * t;
        }
    }
}

__device__ void mmq_q8(const uint8_t* row, const int8_t* xq, const float* xd, int cols, int nin, int tid, int nt,
                       float* acc) {
    const int nb32 = cols / 32;
    for (int it = tid; it < cols / 8; it += nt) {
        const int b = it >> 2, q = it & 3;
        const uint8_t* blk = row + (size_t) b * 34;
        const int w0 = (int) ld32(blk + 2 + q * 8), w1 = (int) ld32(blk + 6 + q * 8);
        const float d = h2f(blk);
#pragma unroll
        for (int ii = 0; ii < 8; ii++) {
            if (ii >= nin) break;
            const int8_t* xb = xq + (size_t) ii * cols + b * 32 + q * 8;
            const int p = __dp4a(w0, *(const int*) xb, __dp4a(w1, *(const int*) (xb + 4), 0));
            acc[ii] += d * xd[(size_t) ii * nb32 + b] * (float) p;
        }
    }
}

template <int TYPE>
__global__ void mmq(const uint8_t* w, size_t row_bytes, int rows, int cols, int n, const int8_t* xq, const float* xd,
                    float* y) {
    const int r = blockIdx.x, i0 = blockIdx.y * NB;
    const int nin = min((int) NB, n - i0);
    const uint8_t* row = w + (size_t) r * row_bytes;
    const int8_t* xqi = xq + (size_t) i0 * cols;
    const float* xdi = xd + (size_t) i0 * (cols / 32);
    float acc[NB];
#pragma unroll
    for (int k = 0; k < NB; k++) acc[k] = 0.f;
    if (TYPE == T_Q4_K) mmq_k45<false>(row, xqi, xdi, cols, nin, threadIdx.x, blockDim.x, acc);
    else if (TYPE == T_Q5_K) mmq_k45<true>(row, xqi, xdi, cols, nin, threadIdx.x, blockDim.x, acc);
    else if (TYPE == T_Q6_K) mmq_q6k(row, xqi, xdi, cols, nin, threadIdx.x, blockDim.x, acc);
    else mmq_q8(row, xqi, xdi, cols, nin, threadIdx.x, blockDim.x, acc);
    __shared__ float part[NB][32];
#pragma unroll
    for (int k = 0; k < NB; k++) {
        float s = acc[k];
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        if ((threadIdx.x & 31) == 0) part[k][threadIdx.x >> 5] = s;
    }
    __syncthreads();
    if (threadIdx.x < nin) {
        float t = 0.f;
        for (int k = 0; k < (int) (blockDim.x >> 5); k++) t += part[threadIdx.x][k];
        y[(size_t) (i0 + threadIdx.x) * rows + r] = t;
    }
}

// ---- prefill on the tensor cores (int8 mma.sync m16n8k32, sm_80+): the same int8 inputs and scales as the dp4a path, 16 rows x
// 32 columns per warp, 64 rows per block. One mma is one 32-value block; the integer result is scaled in fp32 in block order. ----

__device__ __forceinline__ void mma_s8_k32(int* c, const int* a, int b0, int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_s8_k16(int* c, const int* a, int b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b0));
}

template <int TYPE, int NT>
__global__ void __launch_bounds__(128) mma_q(const uint8_t* __restrict__ w, size_t row_bytes, int rows, int cols, int n,
                                            const int8_t* __restrict__ xq, const float* __restrict__ xd, float* __restrict__ y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
    const int r0 = blockIdx.x * 64 + warp * 16 + g, r1 = r0 + 8, c0 = blockIdx.y * 8 * NT;
    const uint8_t* w0 = w + (size_t) min(r0, rows - 1) * row_bytes;
    const uint8_t* w1 = w + (size_t) min(r1, rows - 1) * row_bytes;
    const int nb32 = cols / 32;
    const int8_t* xb[NT];
    for (int c = 0; c < NT; ++c) xb[c] = xq + (size_t) min(c0 + 8 * c + g, n - 1) * cols;
    float acc[NT][4] = {};
    for (int sb = 0; sb < cols / 256; ++sb) {
        if (TYPE == T_Q4_K) {
            const uint8_t* q0 = w0 + (size_t) sb * 144;
            const uint8_t* q1 = w1 + (size_t) sb * 144;
            const float dd[2] = {h2f(q0), h2f(q1)}, dm[2] = {h2f(q0 + 2), h2f(q1 + 2)};
            for (int j = 0; j < 8; ++j) {
                const int p = j >> 1, sh = 4 * (j & 1);
                int a[4];
                for (int h = 0; h < 2; ++h)
                    for (int r = 0; r < 2; ++r)
                        a[2 * h + r] = (*(const int*) ((r ? q1 : q0) + 16 + 32 * p + 4 * t4 + 16 * h) >> sh) & 0x0F0F0F0F;
                float sc[2], mn[2];
                for (int r = 0; r < 2; ++r) {
                    const uint8_t* S = (r ? q1 : q0) + 4;
                    int sv, mv;
                    scale_min(j, S, &sv, &mv);
                    sc[r] = dd[r] * sv;
                    mn[r] = dm[r] * mv;
                }
                const int kb = sb * 8 + j;
                for (int c = 0; c < NT; ++c) {
                    const int b0 = *(const int*) (xb[c] + kb * 32 + 4 * t4), b1 = *(const int*) (xb[c] + kb * 32 + 16 + 4 * t4);
                    int cc[4] = {0, 0, 0, 0}, cs[4] = {0, 0, 0, 0};
                    mma_s8_k32(cc, a, b0, b1);
                    const int ones[4] = {0x01010101, 0x01010101, 0x01010101, 0x01010101};
                    mma_s8_k32(cs, ones, b0, b1);
                    const float v0 = xd[(size_t) min(c0 + 8 * c + 2 * t4, n - 1) * nb32 + kb], v1 = xd[(size_t) min(c0 + 8 * c + 2 * t4 + 1, n - 1) * nb32 + kb];
                    acc[c][0] += v0 * ((float) cc[0] * sc[0] - mn[0] * (float) cs[0]);
                    acc[c][1] += v1 * ((float) cc[1] * sc[0] - mn[0] * (float) cs[1]);
                    acc[c][2] += v0 * ((float) cc[2] * sc[1] - mn[1] * (float) cs[2]);
                    acc[c][3] += v1 * ((float) cc[3] * sc[1] - mn[1] * (float) cs[3]);
                }
            }
        } else {  // Q6_K: a scale covers 16 values, so one 32-value block is two k16 steps
            const uint8_t* q0 = w0 + (size_t) sb * 210;
            const uint8_t* q1 = w1 + (size_t) sb * 210;
            const float dd[2] = {h2f(q0 + 208), h2f(q1 + 208)};
            for (int nn = 0; nn < 2; ++nn)
                for (int k = 0; k < 4; ++k) {
                    int a[2][2];
                    for (int h = 0; h < 2; ++h)
                        for (int r = 0; r < 2; ++r) {
                            const uint8_t* q = r ? q1 : q0;
                            const uint32_t ql = ld32(q + 64 * nn + 32 * (k & 1) + 4 * t4 + 16 * h) >> (4 * (k >> 1));
                            const uint32_t qh = ld32(q + 128 + 32 * nn + 4 * t4 + 16 * h) >> (2 * k);
                            a[h][r] = __vsubss4((int) ((ql & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4)), 0x20202020);
                        }
                    float sc[2][2];
                    for (int h = 0; h < 2; ++h)
                        for (int r = 0; r < 2; ++r) sc[h][r] = dd[r] * (float) ((const int8_t*) ((r ? q1 : q0) + 192))[8 * nn + 2 * k + h];
                    const int kb = sb * 8 + 4 * nn + k;
                    for (int c = 0; c < NT; ++c) {
                        int cc[2][4] = {};
                        for (int h = 0; h < 2; ++h) mma_s8_k16(cc[h], a[h], *(const int*) (xb[c] + kb * 32 + 16 * h + 4 * t4));
                        const float v0 = xd[(size_t) min(c0 + 8 * c + 2 * t4, n - 1) * nb32 + kb], v1 = xd[(size_t) min(c0 + 8 * c + 2 * t4 + 1, n - 1) * nb32 + kb];
                        acc[c][0] += v0 * ((float) cc[0][0] * sc[0][0] + (float) cc[1][0] * sc[1][0]);
                        acc[c][1] += v1 * ((float) cc[0][1] * sc[0][0] + (float) cc[1][1] * sc[1][0]);
                        acc[c][2] += v0 * ((float) cc[0][2] * sc[0][1] + (float) cc[1][2] * sc[1][1]);
                        acc[c][3] += v1 * ((float) cc[0][3] * sc[0][1] + (float) cc[1][3] * sc[1][1]);
                    }
                }
        }
    }
    for (int c = 0; c < NT; ++c)
        for (int i = 0; i < 4; ++i) {
            const int row = (i & 2) ? r1 : r0, col = c0 + 8 * c + 2 * t4 + (i & 1);
            if (row < rows && col < n) y[(size_t) col * rows + row] = acc[c][i];
        }
#endif
}

static bool has_mma() {
    static const bool ok = [] { int d = 0, mj = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&mj, cudaDevAttrComputeCapabilityMajor, d); return mj >= 8; }();
    return ok;
}

struct Scratch {
    float *x = nullptr, *y = nullptr, *xd = nullptr;
    int8_t* xq = nullptr;
    size_t xcap = 0, ycap = 0, qcap = 0;
};
static Scratch g_scratch[16];

extern "C" int dg_device_count() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess ? n : 0;
}

extern "C" int dg_mem(int dev, size_t* free_bytes, size_t* total_bytes) {
    if (cudaSetDevice(dev) != cudaSuccess) return 1;
    return cudaMemGetInfo(free_bytes, total_bytes) == cudaSuccess ? 0 : 2;
}

extern "C" void* dg_upload(int dev, const void* host, size_t bytes) {
    if (cudaSetDevice(dev) != cudaSuccess) return nullptr;
    void* d = nullptr;
    if (cudaMalloc(&d, bytes) != cudaSuccess) return nullptr;
    if (cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d);
        return nullptr;
    }
    return d;
}

extern "C" void dg_free(int dev, void* p) {
    if (p && cudaSetDevice(dev) == cudaSuccess) cudaFree(p);
}

// y = W x for n device-resident fp32 inputs (x, y on dev); the int8 dp4a path for n >= min_q, the exact float kernels below
static int mm_dev(int dev, int type, const void* w, size_t row_bytes, int rows, int cols, int n, const float* x, float* y, int min_q = 4) {
    Scratch& s = g_scratch[dev];
    const dim3 grid(rows, (n + NB - 1) / NB), block(128);
    const uint8_t* wp = (const uint8_t*) w;
    if (n >= min_q && cols % 256 == 0 && (type == T_Q4_K || type == T_Q5_K || type == T_Q6_K || type == T_Q8_0)) {
        const size_t qb = (size_t) n * cols, db = (size_t) n * (cols / 32) * sizeof(float);
        if (qb + db > s.qcap) {
            cudaFree(s.xq);
            cudaFree(s.xd);
            s.xq = nullptr;
            s.xd = nullptr;
            if (cudaMalloc(&s.xq, qb) != cudaSuccess || cudaMalloc(&s.xd, db) != cudaSuccess) return 108;
            s.qcap = qb + db;
        }
        const int warps = n * (cols / 32);
        quant_x<<<(warps + 7) / 8, 256>>>(x, cols, n, s.xq, s.xd);
        if (n >= 16 && has_mma() && (type == T_Q4_K || type == T_Q6_K)) {
            const bool wide = n >= 128;   // 128 columns per block: each weight read serves 4x more columns
            const dim3 g2((rows + 63) / 64, (n + (wide ? 127 : 31)) / (wide ? 128 : 32));
            if (type == T_Q4_K) { if (wide) mma_q<T_Q4_K, 16><<<g2, 128>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, y); else mma_q<T_Q4_K, 4><<<g2, 128>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, y); }
            else { if (wide) mma_q<T_Q6_K, 16><<<g2, 128>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, y); else mma_q<T_Q6_K, 4><<<g2, 128>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, y); }
            return cudaGetLastError() != cudaSuccess ? 106 : 0;
        }
        switch (type) {
        case T_Q4_K: mmq<T_Q4_K><<<grid, block>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, y); break;
        case T_Q5_K: mmq<T_Q5_K><<<grid, block>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, y); break;
        case T_Q6_K: mmq<T_Q6_K><<<grid, block>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, y); break;
        default: mmq<T_Q8_0><<<grid, block>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, y); break;
        }
        return cudaGetLastError() != cudaSuccess ? 106 : 0;
    }
    switch (type) {
    case T_Q4_K: matvec<T_Q4_K><<<grid, block>>>(wp, row_bytes, rows, cols, n, x, y); break;
    case T_Q5_K: matvec<T_Q5_K><<<grid, block>>>(wp, row_bytes, rows, cols, n, x, y); break;
    case T_Q6_K: matvec<T_Q6_K><<<grid, block>>>(wp, row_bytes, rows, cols, n, x, y); break;
    case T_Q8_0: matvec<T_Q8_0><<<grid, block>>>(wp, row_bytes, rows, cols, n, x, y); break;
    case T_BF16: matvec<T_BF16><<<grid, block>>>(wp, row_bytes, rows, cols, n, x, y); break;
    default: return 105;
    }
    return cudaGetLastError() != cudaSuccess ? 106 : 0;
}

extern "C" int dg_matvec(int dev, int type, const void* w, size_t row_bytes, int rows, int cols, const float* x,
                         int n, float* y) {
    if (dev < 0 || dev >= 16 || rows <= 0 || cols <= 0 || n <= 0 || rows > 1 << 30 || n > 65535) return 100;
    if (cudaSetDevice(dev) != cudaSuccess) return 101;
    Scratch& s = g_scratch[dev];
    const size_t xb = (size_t) n * cols * sizeof(float), yb = (size_t) n * rows * sizeof(float);
    if (xb > s.xcap) {
        cudaFree(s.x);
        s.x = nullptr;
        if (cudaMalloc(&s.x, xb) != cudaSuccess) return 102;
        s.xcap = xb;
    }
    if (yb > s.ycap) {
        cudaFree(s.y);
        s.y = nullptr;
        if (cudaMalloc(&s.y, yb) != cudaSuccess) return 103;
        s.ycap = yb;
    }
    if (cudaMemcpy(s.x, x, xb, cudaMemcpyHostToDevice) != cudaSuccess) return 104;
    if (const int rc = mm_dev(dev, type, w, row_bytes, rows, cols, n, s.x, s.y)) return rc;
    return cudaMemcpy(y, s.y, yb, cudaMemcpyDeviceToHost) != cudaSuccess ? 107 : 0;
}

// ======================= Qwen3.8 dense forward with the activations on the card (see dg_qwen_run) =======================

__global__ void rmsnorm_k(const float* x, const float* w, float* out, int dim, float eps) {
    const float* xi = x + (size_t) blockIdx.x * dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xi[i] * xi[i];
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    float t = 0.f;
    for (int k = 0; k < (int) (blockDim.x >> 5); k++) t += part[k];
    const float k = rsqrtf(t / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) out[(size_t) blockIdx.x * dim + i] = xi[i] * k * w[i];
}

__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float silu_(float x) { return x * sigm(x); }

__global__ void add_k(float* h, const float* y, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] += y[i];
}

__global__ void silu_mul_k(float* g, const float* u, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = silu_(g[i]) * u[i];
}

// causal depthwise conv (taps cw[c*K .. c*K+K)) over ext = [3 rows of history; n new rows], then silu
__global__ void gdn_conv_k(const float* ext, const float* cw, float* out, int ch, int n, int K) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * ch) return;
    const int t = (int) (i / ch), c = (int) (i % ch);
    float s = 0.f;
    for (int k = 0; k < K; k++) s += cw[c * K + k] * ext[(size_t) (t + k) * ch + c];
    out[i] = silu_(s);
}

__global__ void gdn_tail_k(const float* ext, float* tail, int ch, int n, int K) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (K - 1) * ch) return;
    tail[i] = ext[(size_t) n * ch + i];  // rows n .. n+K-2 of ext
}

// q and k of every token, per head: x / sqrt(sum x^2 + eps); block = one (token, head) of the 2 * nk heads
__global__ void l2_k(float* conv, int ch, int hk, float eps) {
    float* v = conv + (size_t) blockIdx.x * ch + blockIdx.y * hk;
    float ss = 0.f;
    for (int i = threadIdx.x; i < hk; i += blockDim.x) ss += v[i] * v[i];
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    float t = 0.f;
    for (int k = 0; k < (int) (blockDim.x >> 5); k++) t += part[k];
    const float k = rsqrtf(t + eps);
    for (int i = threadIdx.x; i < hk; i += blockDim.x) v[i] *= k;
}

// the delta-rule recurrence of one value head per block, thread j owns column j of the 128 x 128 state (kept in registers
// across all n tokens). hk == hv == 128.
__global__ void __launch_bounds__(128) gdn_rec_k(const float* conv, const float* alpha, const float* beta, const float* dt,
                                                const float* a, float* S, float* out, int n, int ch, int nk, int nv, int d_inner) {
    const int h = blockIdx.x, j = threadIdx.x, kh = h % nk;
    float* Sh = S + (size_t) h * 128 * 128;
    float s[128];
#pragma unroll
    for (int i = 0; i < 128; i++) s[i] = Sh[i * 128 + j];
    __shared__ float ks[128], qs[128];
    const float qscale = rsqrtf(128.f);
    for (int t = 0; t < n; t++) {
        const float* row = conv + (size_t) t * ch;
        ks[j] = row[nk * 128 + kh * 128 + j];
        qs[j] = row[kh * 128 + j];
        __syncthreads();
        const float v = row[2 * nk * 128 + h * 128 + j];
        const float x = alpha[t * nv + h] + dt[h];
        const float sp = x > 20.f ? x : log1pf(expf(x));
        const float decay = expf(sp * a[h]), b = sigm(beta[t * nv + h]);
        float d = 0.f;
#pragma unroll
        for (int i = 0; i < 128; i++) {
            s[i] *= decay;
            d += s[i] * ks[i];
        }
        d = (v - d) * b;
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < 128; i++) {
            s[i] += ks[i] * d;
            o += s[i] * qs[i] * qscale;
        }
        out[(size_t) t * d_inner + h * 128 + j] = o;
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 128; i++) Sh[i * 128 + j] = s[i];
}

// o = rms(o per head, w) * silu(z), in place; block = (token, head), 128 threads
__global__ void gdn_gate_k(float* o, const float* z, const float* w, int d_inner, float eps) {
    const size_t base = (size_t) blockIdx.x * d_inner + blockIdx.y * 128;
    const int j = threadIdx.x;
    const float v = o[base + j];
    float ss = v * v;
    for (int q = 16; q > 0; q >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, q);
    __shared__ float part[4];
    if ((j & 31) == 0) part[j >> 5] = ss;
    __syncthreads();
    const float k = rsqrtf((part[0] + part[1] + part[2] + part[3]) / 128.f + eps);
    o[base + j] = v * k * w[j] * silu_(z[base + j]);
}

// per (token, head): rms-norm with w, neox rope on the first n_rot values; blockIdx.y < n_head: a query head of qf (query and
// gate interleaved, 2 * hd per head) written to Q; otherwise a key head of Kb written to the half cache with its value head
__global__ void __launch_bounds__(256) qk_prep_k(const float* qf, const float* Kb, const float* Vb, float* Q, __half* Kc, __half* Vc, const float* qn,
                                                 const float* kn, int n_head, int n_kv, int hd, int n_rot, float eps, float base, int pos0) {
    const int t = blockIdx.x, hh = blockIdx.y, i = threadIdx.x;
    const bool isq = hh < n_head;
    const int head = isq ? hh : hh - n_head;
    const float* src = isq ? qf + ((size_t) t * n_head + head) * 2 * hd : Kb + ((size_t) t * n_kv + head) * hd;
    const float* w = isq ? qn : kn;
    __shared__ float v[256], part[8];
    const float x = src[i];
    float ss = x * x;
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((i & 31) == 0) part[i >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
    for (int k = 0; k < hd / 32; k++) tot += part[k];
    v[i] = x * rsqrtf(tot / hd + eps) * w[i];
    __syncthreads();
    float r = v[i];
    const int half = n_rot / 2;
    if (i < n_rot) {  // pair (i % half, i % half + half)
        const int p = i % half;
        const double theta = (double) (pos0 + t) * pow((double) base, -2.0 * p / n_rot);
        double sn, cs;
        sincos(theta, &sn, &cs);
        const float a = v[p], b = v[p + half];
        r = i < half ? (float) (a * cs - b * sn) : (float) (a * sn + b * cs);
    }
    if (isq) Q[((size_t) t * n_head + head) * hd + i] = r;
    else {
        const size_t o = ((size_t) (pos0 + t) * n_kv + head) * hd + i;
        Kc[o] = __float2half(r);
        Vc[o] = __float2half(Vb[((size_t) t * n_kv + head) * hd + i]);
    }
}

// causal attention, one block per (token, query head), hd == 256: a warp walks every 8th position with an online softmax,
// then the 8 partial results are merged; the output is multiplied by sigmoid(gate)
// splits > 1: blockIdx.z takes a slice of the positions and writes its unnormalized result (max, sum, values) to pm/pl/po for
// attn_combine_k; splits == 1 finishes the head here.
__global__ void __launch_bounds__(256) attn_k(const float* Q, const float* qf, const __half* Kc, const __half* Vc, float* ctx, int n_head, int n_kv, int pos0,
                                             int splits, float* pm, float* pl, float* po) {
    const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int kvh = h / (n_head / n_kv), last = pos0 + t, row = n_kv * 256;
    float q[8];
    for (int e = 0; e < 8; e++) q[e] = Q[((size_t) t * n_head + h) * 256 + lane * 8 + e];
    const float scale = rsqrtf(256.f);
    float m = -INFINITY, l = 0.f, o[8] = {};
    const int per = (last + 1 + splits - 1) / splits, jlo = blockIdx.z * per, jhi = min(last + 1, jlo + per);
    for (int j = jlo + warp; j < jhi; j += 8) {
        const uint4 kv = *(const uint4*) (Kc + (size_t) j * row + kvh * 256 + lane * 8);
        const __half2* kh = (const __half2*) &kv;
        float s = 0.f;
        for (int e = 0; e < 4; e++) {
            const float2 f = __half22float2(kh[e]);
            s += q[2 * e] * f.x + q[2 * e + 1] * f.y;
        }
        for (int d = 16; d > 0; d >>= 1) s += __shfl_xor_sync(0xffffffffu, s, d);
        s *= scale;
        const float mn = fmaxf(m, s), corr = __expf(m - mn), p = __expf(s - mn);
        l = l * corr + p;
        const uint4 vv = *(const uint4*) (Vc + (size_t) j * row + kvh * 256 + lane * 8);
        const __half2* vh = (const __half2*) &vv;
        for (int e = 0; e < 4; e++) {
            const float2 f = __half22float2(vh[e]);
            o[2 * e] = o[2 * e] * corr + p * f.x;
            o[2 * e + 1] = o[2 * e + 1] * corr + p * f.y;
        }
        m = mn;
    }
    __shared__ float wm[8], wl[8], wo[8][256];
    if (lane == 0) wm[warp] = m, wl[warp] = l;
    for (int e = 0; e < 8; e++) wo[warp][lane * 8 + e] = o[e];
    __syncthreads();
    float M = -INFINITY;
    for (int w = 0; w < 8; w++) M = fmaxf(M, wm[w]);
    float L = 0.f, acc = 0.f;
    for (int w = 0; w < 8; w++) {
        const float c = wm[w] == -INFINITY ? 0.f : __expf(wm[w] - M);
        L += wl[w] * c;
        acc += wo[w][tid] * c;
    }
    if (splits > 1) {
        const size_t slot = ((size_t) t * n_head + h) * splits + blockIdx.z;
        if (tid == 0) pm[slot] = M, pl[slot] = L;
        po[slot * 256 + tid] = acc;
        return;
    }
    ctx[((size_t) t * n_head + h) * 256 + tid] = acc / L * sigm(qf[((size_t) t * n_head + h) * 512 + 256 + tid]);
}

__global__ void __launch_bounds__(256) attn_combine_k(const float* pm, const float* pl, const float* po, const float* qf, float* ctx, int n_head, int splits) {
    const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const size_t base = ((size_t) t * n_head + h) * splits;
    float M = -INFINITY;
    for (int s = 0; s < splits; s++) M = fmaxf(M, pm[base + s]);
    float L = 0.f, acc = 0.f;
    for (int s = 0; s < splits; s++) {
        const float c = pm[base + s] == -INFINITY ? 0.f : __expf(pm[base + s] - M);
        L += pl[base + s] * c;
        acc += po[(base + s) * 256 + tid] * c;
    }
    ctx[((size_t) t * n_head + h) * 256 + tid] = acc / L * sigm(qf[((size_t) t * n_head + h) * 512 + 256 + tid]);
}

enum { kMaxSplitTokens = 8, kMaxSplits = 32 };

struct QBuf {
    float *h = nullptr, *xn, *ext, *pm, *pl, *po, *z, *beta, *alpha, *conv, *o, *y, *g, *u, *qf, *Q, *Kb, *Vb;
    size_t cap = 0;  // tokens
};
static QBuf g_qb[16];

static bool qalloc(QBuf& b, int n, const dg_cfg& c) {
    if ((size_t) n <= b.cap) return true;
    cudaFree(b.h);
    const size_t E = c.n_embd, ch = 2 * (size_t) c.nk * c.hk + c.d_inner, nq = (size_t) c.n_head * c.hd, nkv = (size_t) c.n_kv * c.hd;
    const size_t per = 3 * E + (ch + 3 * ch) + 2 * c.d_inner + 2 * (size_t) c.nv + E + 2 * (size_t) c.n_ff + 2 * nq + nq + 2 * nkv;
    float* p = nullptr;
    const size_t part = (size_t) kMaxSplitTokens * c.n_head * kMaxSplits;   // slots of the split attention
    if (cudaMalloc(&p, (per * n + 3 * ch + part * 258) * sizeof(float)) != cudaSuccess) return false;
    size_t off = 0;
    auto take = [&](size_t cnt) { float* r = p + off; off += cnt; return r; };
    b.h = take((size_t) n * E);
    b.xn = take((size_t) n * E);
    b.ext = take((size_t) (n + 3) * ch);
    b.z = take((size_t) n * c.d_inner);
    b.beta = take((size_t) n * c.nv);
    b.alpha = take((size_t) n * c.nv);
    b.conv = take((size_t) n * ch);
    b.o = take((size_t) n * c.d_inner);
    b.y = take((size_t) n * E);
    b.g = take((size_t) n * c.n_ff);
    b.u = take((size_t) n * c.n_ff);
    b.qf = take((size_t) n * 2 * nq);
    b.Q = take((size_t) n * nq);
    b.Kb = take((size_t) n * nkv);
    b.Vb = take((size_t) n * nkv);
    b.pm = take(part);
    b.pl = take(part);
    b.po = take(part * 256);
    b.cap = n;
    return true;
}

extern "C" void* dg_alloc_zero(int dev, size_t bytes) {
    if (cudaSetDevice(dev) != cudaSuccess) return nullptr;
    void* d = nullptr;
    if (cudaMalloc(&d, bytes) != cudaSuccess) return nullptr;
    cudaMemset(d, 0, bytes);
    return d;
}

extern "C" int dg_zero(int dev, void* p, size_t bytes) {
    return cudaSetDevice(dev) == cudaSuccess && cudaMemset(p, 0, bytes) == cudaSuccess ? 0 : 1;
}

// device read bandwidth in GB/s, measured with a device-to-device copy
extern "C" double dg_bandwidth(int dev) {
    if (cudaSetDevice(dev) != cudaSuccess) return 0;
    const size_t sz = 256 << 20;
    void *a = nullptr, *b = nullptr;
    if (cudaMalloc(&a, sz) != cudaSuccess || cudaMalloc(&b, sz) != cudaSuccess) return 0;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaMemcpy(b, a, sz, cudaMemcpyDeviceToDevice);
    cudaEventRecord(e0);
    for (int i = 0; i < 4; i++) cudaMemcpy(b, a, sz, cudaMemcpyDeviceToDevice);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaFree(a);
    cudaFree(b);
    return 4.0 * sz * 2 / (ms * 1e6);  // a copy reads and writes
}

// DG_PROF=1: wait after every section of dg_qwen_run and add its time to a category; dg_prof_dump prints and clears them
#include <chrono>
#include <cstdio>
#include <cstdlib>
static const char* kCat[] = {"norm+add", "gdn proj (qkv,z,beta,alpha)", "gdn conv/l2/rec/gate", "gdn out", "attn proj (q,k,v)", "attn prep+core", "attn out", "ffn gate+up", "silu_mul", "ffn down"};
static double g_prof[10];
static bool g_prof_on = getenv("DG_PROF") != nullptr;
static std::chrono::steady_clock::time_point g_t0;
static void prof_mark(int cat) {
    if (!g_prof_on) return;
    cudaDeviceSynchronize();
    const auto t = std::chrono::steady_clock::now();
    if (cat >= 0) g_prof[cat] += std::chrono::duration<double, std::milli>(t - g_t0).count();
    g_t0 = t;
}
extern "C" void dg_prof_dump() {
    for (int i = 0; i < 10; i++) {
        if (g_prof[i] > 0) fprintf(stderr, "  %-30s %8.1f ms\n", kCat[i], g_prof[i]);
        g_prof[i] = 0;
    }
}

#define QCHK(x) do { if ((x) != cudaSuccess) return 200 + __LINE__ % 100; } while (0)

// Runs layers L[0..nl) (all on dev) over n new tokens whose hidden states are h (n x n_embd, host), starting at position pos0;
// the result replaces h. Returns 0 on success.
extern "C" int dg_qwen_run(int dev, const dg_cfg* c, const dg_layer* L, int nl, float* h, int n, int pos0) {
    if (n <= 0 || n > 4096 || c->hd != 256 || c->hk != 128 || c->d_inner != c->nv * 128 || c->n_head % c->n_kv) return 100;
    QCHK(cudaSetDevice(dev));
    QBuf& b = g_qb[dev];
    if (!qalloc(b, n, *c)) return 101;
    const int E = c->n_embd, ch = 2 * c->nk * c->hk + c->d_inner;
    QCHK(cudaMemcpy(b.h, h, (size_t) n * E * sizeof(float), cudaMemcpyHostToDevice));
    auto mm = [&](const dg_mat& m, const float* x, float* y) {
        return mm_dev(dev, m.type, m.w, m.rb, m.rows, m.cols, n, x, y, 1);
    };
    auto blocks = [](size_t cnt) { return (unsigned) ((cnt + 255) / 256); };
    for (int li = 0; li < nl; li++) {
        const dg_layer& l = L[li];
        int rc = 0;
        prof_mark(-1);
        rmsnorm_k<<<n, 256>>>(b.h, l.attn_norm, b.xn, E, c->eps);
        prof_mark(0);
        if (!l.attn) {
            const dg_mat *qkv = &l.m[0], *z = &l.m[1], *be = &l.m[2], *al = &l.m[3], *out = &l.m[4];
            QCHK(cudaMemcpyAsync(b.ext, l.tail, (size_t) 3 * ch * sizeof(float), cudaMemcpyDeviceToDevice));
            if ((rc = mm(*qkv, b.xn, b.ext + (size_t) 3 * ch)) || (rc = mm(*z, b.xn, b.z)) || (rc = mm(*be, b.xn, b.beta)) || (rc = mm(*al, b.xn, b.alpha))) return rc;
            prof_mark(1);
            gdn_conv_k<<<blocks((size_t) n * ch), 256>>>(b.ext, l.f[2], b.conv, ch, n, c->conv_k);
            gdn_tail_k<<<blocks((size_t) 3 * ch), 256>>>(b.ext, l.tail, ch, n, c->conv_k);
            l2_k<<<dim3(n, 2 * c->nk), 128>>>(b.conv, ch, c->hk, c->eps);
            gdn_rec_k<<<c->nv, 128>>>(b.conv, b.alpha, b.beta, l.f[0], l.f[1], l.S, b.o, n, ch, c->nk, c->nv, c->d_inner);
            gdn_gate_k<<<dim3(n, c->nv), 128>>>(b.o, b.z, l.f[3], c->d_inner, c->eps);
            prof_mark(2);
            if ((rc = mm(*out, b.o, b.y))) return rc;
            prof_mark(3);
        } else {
            if ((c->cap < pos0 + n)) return 102;
            const dg_mat *q = &l.m[0], *k = &l.m[1], *v = &l.m[2], *o = &l.m[3];
            if ((rc = mm(*q, b.xn, b.qf)) || (rc = mm(*k, b.xn, b.Kb)) || (rc = mm(*v, b.xn, b.Vb))) return rc;
            prof_mark(4);
            qk_prep_k<<<dim3(n, c->n_head + c->n_kv), 256>>>(b.qf, b.Kb, b.Vb, b.Q, (__half*) l.Kc, (__half*) l.Vc, l.f[0], l.f[1], c->n_head, c->n_kv, c->hd,
                                                              c->n_rot, c->eps, c->base, pos0);
            // a few tokens on a long context: split the positions over blocks, then combine
            const int splits = n <= kMaxSplitTokens && !getenv("DG_NOSPLIT") ? std::max(1, std::min((int) kMaxSplits, (pos0 + n) / 256)) : 1;
            attn_k<<<dim3(n, c->n_head, splits), 256>>>(b.Q, b.qf, (const __half*) l.Kc, (const __half*) l.Vc, b.o, c->n_head, c->n_kv, pos0, splits, b.pm, b.pl, b.po);
            if (splits > 1) attn_combine_k<<<dim3(n, c->n_head), 256>>>(b.pm, b.pl, b.po, b.qf, b.o, c->n_head, splits);
            prof_mark(5);
            if ((rc = mm(*o, b.o, b.y))) return rc;
            prof_mark(6);
        }
        add_k<<<blocks((size_t) n * E), 256>>>(b.h, b.y, (size_t) n * E);
        rmsnorm_k<<<n, 256>>>(b.h, l.post_norm, b.xn, E, c->eps);
        prof_mark(0);
        if ((rc = mm(l.ffn_gate, b.xn, b.g)) || (rc = mm(l.ffn_up, b.xn, b.u))) return rc;
        prof_mark(7);
        silu_mul_k<<<blocks((size_t) n * c->n_ff), 256>>>(b.g, b.u, (size_t) n * c->n_ff);
        prof_mark(8);
        if ((rc = mm(l.ffn_down, b.g, b.y))) return rc;
        prof_mark(9);
        add_k<<<blocks((size_t) n * E), 256>>>(b.h, b.y, (size_t) n * E);
        prof_mark(0);
        QCHK(cudaGetLastError());
    }
    QCHK(cudaMemcpy(h, b.h, (size_t) n * E * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}
