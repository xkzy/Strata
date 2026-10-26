// Quantized matrix-vector products for the dense-model reference (pkg/densecpu): y[i][r] = W[r] . x[i].
// One thread block per (output row, input vector); the arithmetic mirrors the CPU kernels in dot.go.
// Build: see build.sh. No peer access, no unified memory: weights are copied once, activations per call.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "densegpu.h"

#include <chrono>
#include <cstdio>
// Per entry point (0 matvec, 1 attn, 2 gdn): host-to-device, kernel, device-to-host seconds and bytes moved.
static double g_t[3][3];
static double g_b[3][2];
static long g_calls[3];
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
extern "C" void dg_stats(double* t, double* b, long* calls) {
    for (int i = 0; i < 3; i++) {
        calls[i] = g_calls[i];
        for (int j = 0; j < 3; j++) t[i * 3 + j] = g_t[i][j];
        for (int j = 0; j < 2; j++) b[i * 2 + j] = g_b[i][j];
    }
}
extern "C" void dg_stats_reset() { for (int i = 0; i < 3; i++) { g_calls[i] = 0; for (int j = 0; j < 3; j++) g_t[i][j] = 0; g_b[i][0] = g_b[i][1] = 0; } }

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
__global__ void quant_x(const float* x, int cols, int n, int8_t* xq, float* xd, int* xs) {
    const int blocks = cols / 32;
    const int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (wid >= n * blocks) return;
    const float v = x[(size_t) wid * 32 + lane];
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.f;
    const int q = d > 0.f ? __float2int_rn(v / d) : 0;
    xq[(size_t) wid * 32 + lane] = (int8_t) q;
    int sum = q;
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) xd[wid] = d, xs[wid] = sum;
}

// Small batches (1 to 3 inputs, i.e. decode): one block per weight row, int8 inputs with dp4a.
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

// Tensor-core matrix product for batches of inputs. A block computes 64 weight rows x 32 inputs, walking the columns in
// chunks of 256 (one k-quant super-block). Per chunk the weights are decoded once into int8 in shared memory next to the
// quantized inputs, and each warp runs int8 mma.sync over groups of G columns (32; 16 for Q6_K, whose scales are per 16):
//   y[row][tok] += xd * (d*scale * sum(q*xq) - dmin*min * sum(xq))      (the minimum term only for Q4_K / Q5_K)
// where xd is the input's scale for that 32-block and sum(xq) comes from quant_x.
enum { BM = 64, BN = 32, LDS = 272 };

__device__ __forceinline__ void mma32(int* c, int a0, int a1, int a2, int a3, int b0, int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma16(int* c, int a0, int a1, int b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}

// Decodes half `h` (128 columns) of one weight row's super-block into dst (int8) and its group scales / minimums.
template <int TYPE>
__device__ void decode_half(const uint8_t* blk, int h, int8_t* dst, float* ds, float* dm) {
    if (TYPE == T_Q4_K || TYPE == T_Q5_K) {
        const bool five = TYPE == T_Q5_K;
        const float d = h2f(blk), dmin = h2f(blk + 2);
        for (int jj = 0; jj < 2; jj++) {
            const int j = 2 * h + jj;
            int s1, m1, s2, m2;
            scale_min(2 * j, blk + 4, &s1, &m1);
            scale_min(2 * j + 1, blk + 4, &s2, &m2);
            ds[2 * j] = d * s1, dm[2 * j] = dmin * m1, ds[2 * j + 1] = d * s2, dm[2 * j + 1] = dmin * m2;
            const uint8_t* qs = blk + (five ? 48 : 16) + j * 32;
            for (int m = 0; m < 8; m++) {
                const uint32_t wv = ld32(qs + 4 * m);
                uint32_t lo = wv & 0x0F0F0F0Fu, hi = (wv >> 4) & 0x0F0F0F0Fu;
                if (five) {
                    const uint32_t hv = ld32(blk + 16 + 4 * m);
                    lo |= ((hv >> (2 * j)) & 0x01010101u) << 4;
                    hi |= ((hv >> (2 * j + 1)) & 0x01010101u) << 4;
                }
                *(uint32_t*) (dst + j * 64 + 4 * m) = lo;
                *(uint32_t*) (dst + j * 64 + 32 + 4 * m) = hi;
            }
        }
    } else if (TYPE == T_Q6_K) {
        const float d = h2f(blk + 208);
        const int8_t* sc = (const int8_t*) (blk + 192 + h * 8);
        for (int t = 0; t < 8; t++) ds[h * 8 + t] = d * (float) sc[t];
        const uint8_t *ql = blk + h * 64, *qh = blk + 128 + h * 32;
        for (int w4 = 0; w4 < 8; w4++) {
            const uint32_t l0 = ld32(ql + 4 * w4), l1 = ld32(ql + 32 + 4 * w4), hh = ld32(qh + 4 * w4);
            int8_t* o = dst + h * 128 + 4 * w4;
            *(int*) (o) = __vsubss4((int) ((l0 & 0x0F0F0F0Fu) | (((hh >> 0) & 0x03030303u) << 4)), 0x20202020);
            *(int*) (o + 32) = __vsubss4((int) ((l1 & 0x0F0F0F0Fu) | (((hh >> 2) & 0x03030303u) << 4)), 0x20202020);
            *(int*) (o + 64) = __vsubss4((int) (((l0 >> 4) & 0x0F0F0F0Fu) | (((hh >> 4) & 0x03030303u) << 4)), 0x20202020);
            *(int*) (o + 96) = __vsubss4((int) (((l1 >> 4) & 0x0F0F0F0Fu) | (((hh >> 6) & 0x03030303u) << 4)), 0x20202020);
        }
    } else { // Q8_0: 34-byte blocks of 32
        for (int bb = 0; bb < 4; bb++) {
            const int b = 4 * h + bb;
            const uint8_t* p = blk + b * 34;
            ds[b] = h2f(p);
            for (int m = 0; m < 8; m++) *(uint32_t*) (dst + b * 32 + 4 * m) = ld32(p + 2 + 4 * m);
        }
    }
}

template <int TYPE>
__global__ void __launch_bounds__(128) mma_gemm(const uint8_t* w, size_t row_bytes, int rows, int cols, int n,
                                                const int8_t* xq, const float* xd, const int* xs, float* y) {
    constexpr bool Q6 = TYPE == T_Q6_K, HASMIN = TYPE == T_Q4_K || TYPE == T_Q5_K;
    constexpr int G = Q6 ? 16 : 32, NG = 256 / G, SUPER = TYPE == T_Q4_K ? 144 : TYPE == T_Q5_K ? 176 : TYPE == T_Q6_K ? 210 : 272;
    __shared__ __align__(16) int8_t sw[BM * LDS];
    __shared__ __align__(16) int8_t sx[BN * LDS];
    __shared__ float sds[BM][NG], sdm[BM][HASMIN ? 8 : 1], sxd[BN][8];
    __shared__ int sxs[BN][8];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
    const int r0 = blockIdx.x * BM, i0 = blockIdx.y * BN, wr = warp * 16;
    float acc[4][4];
    for (int a = 0; a < 4; a++)
        for (int b = 0; b < 4; b++) acc[a][b] = 0.f;
    for (int ch = 0; ch < cols / 256; ch++) {
        __syncthreads();
        {
            const int lr = tid >> 1, h = tid & 1, r = r0 + lr;
            if (r < rows) {
                decode_half<TYPE>(w + (size_t) r * row_bytes + (size_t) ch * SUPER, h, sw + lr * LDS, sds[lr], sdm[lr]);
            } else {
                for (int k = 0; k < 128; k += 4) *(int*) (sw + lr * LDS + h * 128 + k) = 0;
                for (int k = 0; k < NG; k++) sds[lr][k] = 0.f;
                if (HASMIN)
                    for (int k = 0; k < 8; k++) sdm[lr][k] = 0.f;
            }
        }
        for (int q = tid; q < BN * 16; q += 128) {
            const int tt = q >> 4, c16 = q & 15;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (i0 + tt < n) v = *(const uint4*) (xq + (size_t) (i0 + tt) * cols + ch * 256 + c16 * 16);
            *(uint4*) (sx + tt * LDS + c16 * 16) = v;
        }
        for (int q = tid; q < BN * 8; q += 128) {
            const int tt = q >> 3, s = q & 7;
            const bool ok = i0 + tt < n;
            sxd[tt][s] = ok ? xd[(size_t) (i0 + tt) * (cols / 32) + ch * 8 + s] : 0.f;
            sxs[tt][s] = ok ? xs[(size_t) (i0 + tt) * (cols / 32) + ch * 8 + s] : 0;
        }
        __syncthreads();
        for (int sb = 0; sb < NG; sb++) {
            const int koff = sb * G, s32 = koff / 32;
            const int8_t* ar0 = sw + (wr + g) * LDS + koff + tig * 4;
            const int8_t* ar8 = ar0 + 8 * LDS;
            const int a0 = *(const int*) ar0, a1 = *(const int*) ar8;
            const int a2 = Q6 ? 0 : *(const int*) (ar0 + 16), a3 = Q6 ? 0 : *(const int*) (ar8 + 16);
            const float ds0 = sds[wr + g][sb], ds8 = sds[wr + g + 8][sb];
            float dm0 = 0.f, dm8 = 0.f;
            if (HASMIN) dm0 = sdm[wr + g][sb], dm8 = sdm[wr + g + 8][sb];
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const int8_t* bp = sx + (nt * 8 + g) * LDS + koff + tig * 4;
                int c[4] = {0, 0, 0, 0};
                if (Q6) mma16(c, a0, a1, *(const int*) bp);
                else mma32(c, a0, a1, a2, a3, *(const int*) bp, *(const int*) (bp + 16));
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    const int tok = nt * 8 + tig * 2 + (k & 1);
                    const float dx = sxd[tok][s32];
                    float v = dx * (k < 2 ? ds0 : ds8) * (float) c[k];
                    if (HASMIN) v -= dx * (k < 2 ? dm0 : dm8) * (float) sxs[tok][s32];
                    acc[nt][k] += v;
                }
            }
        }
    }
#pragma unroll
    for (int nt = 0; nt < 4; nt++)
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int tok = i0 + nt * 8 + tig * 2 + (k & 1), row = r0 + wr + g + (k >> 1) * 8;
            if (tok < n && row < rows) y[(size_t) tok * rows + row] = acc[nt][k];
        }
}

struct Scratch {
    float *x = nullptr, *y = nullptr, *xd = nullptr, *g = nullptr, *u = nullptr;
    int* xs = nullptr;
    int8_t* xq = nullptr;
    size_t xcap = 0, ycap = 0, qcap = 0, gcap = 0, ucap = 0;
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

static int finish_matvec(float* y, const float* dy, size_t yb, double t0, double t1) {
    if (cudaDeviceSynchronize() != cudaSuccess) return 106;
    const double t2 = now_s();
    if (cudaMemcpy(y, dy, yb, cudaMemcpyDeviceToHost) != cudaSuccess) return 107;
    const double t3 = now_s();
    g_t[0][0] += t1 - t0; g_t[0][1] += t2 - t1; g_t[0][2] += t3 - t2;
    g_b[0][1] += yb; g_calls[0]++;
    return 0;
}

// Reports a failed launch on stderr (the Go side only sees the code).
static int launch_ok(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return 0;
    fprintf(stderr, "densegpu: %s: %s\n", what, cudaGetErrorString(e));
    return 106;
}

// Grows a device buffer to at least `bytes` (contents are not kept).
static bool grow(void** p, size_t* cap, size_t bytes) {
    if (bytes <= *cap) return true;
    cudaFree(*p);
    *p = nullptr;
    *cap = 0;
    if (cudaMalloc(p, bytes) != cudaSuccess) return false;
    *cap = bytes;
    return true;
}

// dy[i*rows + r] = row r of w dot dx[i*cols ..], all on the card; dx and dy are device pointers, nothing is copied.
static int matvec_core(Scratch& s, int type, const void* w, size_t row_bytes, int rows, int cols, const float* dx, int n,
                       float* dy) {
    const dim3 grid(rows, (n + NB - 1) / NB), block(128);
    const uint8_t* wp = (const uint8_t*) w;
    if (cols % 256 == 0 && (type == T_Q4_K || type == T_Q5_K || type == T_Q6_K || type == T_Q8_0)) {
        const size_t qb = (size_t) n * cols, db = (size_t) n * (cols / 32) * sizeof(float);
        if (qb + 2 * db > s.qcap) {
            cudaFree(s.xq);
            cudaFree(s.xd);
            cudaFree(s.xs);
            s.xq = nullptr;
            s.xd = nullptr;
            s.xs = nullptr;
            if (cudaMalloc(&s.xq, qb) != cudaSuccess || cudaMalloc(&s.xd, db) != cudaSuccess || cudaMalloc(&s.xs, db) != cudaSuccess)
                return 108;
            s.qcap = qb + 2 * db;
        }
        const int warps = n * (cols / 32);
        quant_x<<<(warps + 7) / 8, 256>>>(dx, cols, n, s.xq, s.xd, s.xs);
        if (n < 4) {
            switch (type) {
            case T_Q4_K: mmq<T_Q4_K><<<grid, block>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, dy); break;
            case T_Q5_K: mmq<T_Q5_K><<<grid, block>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, dy); break;
            case T_Q6_K: mmq<T_Q6_K><<<grid, block>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, dy); break;
            default: mmq<T_Q8_0><<<grid, block>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, dy); break;
            }
            return launch_ok("matvec_core");
        }
        const dim3 mg((rows + BM - 1) / BM, (n + BN - 1) / BN);
        switch (type) {
        case T_Q4_K: mma_gemm<T_Q4_K><<<mg, 128>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, s.xs, dy); break;
        case T_Q5_K: mma_gemm<T_Q5_K><<<mg, 128>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, s.xs, dy); break;
        case T_Q6_K: mma_gemm<T_Q6_K><<<mg, 128>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, s.xs, dy); break;
        default: mma_gemm<T_Q8_0><<<mg, 128>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, s.xs, dy); break;
        }
        return launch_ok("matvec_core");
    }
    switch (type) {
    case T_Q4_K: matvec<T_Q4_K><<<grid, block>>>(wp, row_bytes, rows, cols, n, dx, dy); break;
    case T_Q5_K: matvec<T_Q5_K><<<grid, block>>>(wp, row_bytes, rows, cols, n, dx, dy); break;
    case T_Q6_K: matvec<T_Q6_K><<<grid, block>>>(wp, row_bytes, rows, cols, n, dx, dy); break;
    case T_Q8_0: matvec<T_Q8_0><<<grid, block>>>(wp, row_bytes, rows, cols, n, dx, dy); break;
    case T_BF16: matvec<T_BF16><<<grid, block>>>(wp, row_bytes, rows, cols, n, dx, dy); break;
    default: return 105;
    }
    return launch_ok("matvec_core");
}

extern "C" int dg_matvec(int dev, int type, const void* w, size_t row_bytes, int rows, int cols, const float* x,
                         int n, float* y) {
    if (dev < 0 || dev >= 16 || rows <= 0 || cols <= 0 || n <= 0 || rows > 1 << 30 || n > 65535) return 100;
    if (cudaSetDevice(dev) != cudaSuccess) return 101;
    const double t0 = now_s();
    Scratch& s = g_scratch[dev];
    const size_t xb = (size_t) n * cols * sizeof(float), yb = (size_t) n * rows * sizeof(float);
    if (!grow((void**) &s.x, &s.xcap, xb)) return 102;
    if (!grow((void**) &s.y, &s.ycap, yb)) return 103;
    if (cudaMemcpy(s.x, x, xb, cudaMemcpyHostToDevice) != cudaSuccess) return 104;
    g_b[0][0] += xb;
    const double t1 = now_s();
    if (int rc = matvec_core(s, type, w, row_bytes, rows, cols, s.x, n, s.y)) return rc;
    return finish_matvec(y, s.y, yb, t0, t1);
}

// h = act(g) * u in place in g. act 0: silu, 1: gelu (tanh form); the formulas match silu and geluTanh in the Go code.
__global__ void glu_kernel(float* __restrict__ g, const float* __restrict__ u, size_t total, int act) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < total; i += (size_t) gridDim.x * blockDim.x) {
        const float x = g[i];
        const float a = act == 0 ? x / (1.f + expf(-x)) : 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        g[i] = a * u[i];
    }
}

// The whole feed-forward block on one card: y = down(act(gate x) * up x). x goes up once and y comes back once; the
// gate, up and hidden activations never leave the card. All three weights must live on `dev`.
extern "C" int dg_ffn(int dev, int act, int tg, const void* wg, size_t rbg, int tu, const void* wu, size_t rbu, int td,
                      const void* wd, size_t rbd, int n_ff, int n_embd, const float* x, int n, float* y) {
    if (dev < 0 || dev >= 16 || n_ff <= 0 || n_embd <= 0 || n <= 0 || n > 65535) return 100;
    if (cudaSetDevice(dev) != cudaSuccess) return 101;
    const double t0 = now_s();
    Scratch& s = g_scratch[dev];
    const size_t xb = (size_t) n * n_embd * sizeof(float), fb = (size_t) n * n_ff * sizeof(float);
    if (!grow((void**) &s.x, &s.xcap, xb) || !grow((void**) &s.y, &s.ycap, xb)) return 102;
    if (!grow((void**) &s.g, &s.gcap, fb) || !grow((void**) &s.u, &s.ucap, fb)) return 103;
    if (cudaMemcpy(s.x, x, xb, cudaMemcpyHostToDevice) != cudaSuccess) return 104;
    g_b[0][0] += xb;
    const double t1 = now_s();
    if (int rc = matvec_core(s, tg, wg, rbg, n_ff, n_embd, s.x, n, s.g)) return rc;
    if (int rc = matvec_core(s, tu, wu, rbu, n_ff, n_embd, s.x, n, s.u)) return rc;
    const size_t total = (size_t) n * n_ff;
    glu_kernel<<<(unsigned) ((total + 255) / 256 > 4096 ? 4096 : (total + 255) / 256), 256>>>(s.g, s.u, total, act);
    if (cudaGetLastError() != cudaSuccess) return 106;
    if (int rc = matvec_core(s, td, wd, rbd, n_embd, n_ff, s.g, n, s.y)) return rc;
    return finish_matvec(y, s.y, xb, t0, t1);
}

// Attention, one block per (token, head): each of the 8 warps walks every 8th position with an online softmax
// (lane l owns dims l, l+32, ...), then the warps are merged through shared memory.
__global__ void attn_kernel(const float* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
                            int first, int window, int b0, int cap, int nHead, int nKV, int hd, float scale,
                            float* __restrict__ out) {
    __shared__ float sm[8], sl[8], sacc[8][512];
    const int i = blockIdx.x, h = blockIdx.y, kv = h / (nHead / nKV), lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int per = hd / 32, last = first + i, lo = window > 0 ? max(0, last - window + 1) : 0;
    const float* qh = Q + ((size_t) i * nHead + h) * hd;
    float q[16], acc[16];
    for (int k = 0; k < per; k++) q[k] = qh[lane + 32 * k], acc[k] = 0.f;
    float m = -INFINITY, l = 0.f;
    for (int j = lo + w; j <= last; j += 8) {
        const int slot = cap ? (b0 + j) % cap : j; // ring buffer: row j is position b0+j, kept in slot position % cap
        const __half* kp = K + ((size_t) slot * nKV + kv) * hd;
        float s = 0.f;
        for (int k = 0; k < per; k++) s += q[k] * __half2float(kp[lane + 32 * k]);
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        s *= scale;
        const float mn = fmaxf(m, s), c = __expf(m - mn), p = __expf(s - mn);
        const __half* vp = V + ((size_t) slot * nKV + kv) * hd;
        l = l * c + p;
        for (int k = 0; k < per; k++) acc[k] = acc[k] * c + p * __half2float(vp[lane + 32 * k]);
        m = mn;
    }
    for (int k = 0; k < per; k++) sacc[w][lane + 32 * k] = acc[k];
    if (lane == 0) sm[w] = m, sl[w] = l;
    __syncthreads();
    float gm = sm[0];
    for (int x = 1; x < 8; x++) gm = fmaxf(gm, sm[x]);
    float L = 0.f;
    for (int x = 0; x < 8; x++) L += sl[x] * __expf(sm[x] - gm);
    for (int d = threadIdx.x; d < hd; d += 256) {
        float o = 0.f;
        for (int x = 0; x < 8; x++) o += sacc[x][d] * __expf(sm[x] - gm);
        out[((size_t) i * nHead + h) * hd + d] = o / L;
    }
}

extern "C" int dg_attn_dev(int dev, int n, int first, int window, int b0, int cap, int nHead, int nKV, int hd, float scale, const float* q,
                           const void* dk, const void* dv, float* out) {
    if (dev < 0 || dev >= 16 || n <= 0 || first < 0 || window < 0 || b0 < 0 || cap < 0 || nKV <= 0 || nHead % nKV || hd % 32 || hd > 512 || n > 1 << 30) return 100;
    if (cudaSetDevice(dev) != cudaSuccess) return 101;
    Scratch& s = g_scratch[dev];
    const size_t qb = (size_t) n * nHead * hd * sizeof(float);
    if (!grow((void**) &s.x, &s.xcap, qb) || !grow((void**) &s.y, &s.ycap, qb)) return 102;
    if (cudaMemcpy(s.x, q, qb, cudaMemcpyHostToDevice) != cudaSuccess) return 104;
    attn_kernel<<<dim3(n, nHead), 256>>>(s.x, (const __half*) dk, (const __half*) dv, first, window, b0, cap, nHead, nKV, hd, scale, s.y);
    if (cudaGetLastError() != cudaSuccess) return 106;
    return cudaMemcpy(out, s.y, qb, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : 107;
}

// Device memory helpers for the resident KV cache and delta-net state.
extern "C" void* dg_alloc(int dev, size_t bytes) {
    void* p = nullptr;
    return cudaSetDevice(dev) == cudaSuccess && cudaMalloc(&p, bytes) == cudaSuccess ? p : nullptr;
}
extern "C" int dg_h2d(int dev, void* dst, size_t off, const void* src, size_t bytes) {
    return cudaSetDevice(dev) == cudaSuccess && cudaMemcpy((char*) dst + off, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : 1;
}
extern "C" int dg_d2h(int dev, void* dst, const void* src, size_t bytes) {
    return cudaSetDevice(dev) == cudaSuccess && cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : 1;
}
extern "C" int dg_d2d(int dev, void* dst, const void* src, size_t bytes) {
    return cudaSetDevice(dev) == cudaSuccess && cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice) == cudaSuccess ? 0 : 1;
}

// Delta-net: one block per value head, one thread per value column j; the thread keeps column j of the state in
// registers, so every sum runs over i in order exactly as the CPU loop in Qwen.gdn does.
__global__ void gdn_kernel(const float* __restrict__ conv, const float* __restrict__ decay, const float* __restrict__ beta,
                           float* __restrict__ S, float* __restrict__ out, int n, int numK, int numV, int hk, int hv,
                           float qs) {
    __shared__ float sq[128], sk[128];
    const int h = blockIdx.x, j = threadIdx.x, kh = h % numK, ch = 2 * numK * hk + numV * hv;
    float* Sh = S + (size_t) h * hk * hv;
    float col[128];
    for (int i = 0; i < hk; i++) col[i] = Sh[(size_t) i * hv + j];
    for (int t = 0; t < n; t++) {
        const float* row = conv + (size_t) t * ch;
        __syncthreads();
        if (j < hk) sq[j] = row[kh * hk + j], sk[j] = row[numK * hk + kh * hk + j];
        __syncthreads();
        const float dc = decay[(size_t) t * numV + h], b = beta[(size_t) t * numV + h];
        float d = 0.f;
        for (int i = 0; i < hk; i++) {
            col[i] *= dc;
            d += col[i] * sk[i];
        }
        d = (row[2 * numK * hk + h * hv + j] - d) * b;
        float o = 0.f;
        for (int i = 0; i < hk; i++) {
            col[i] += sk[i] * d;
            o += col[i] * sq[i] * qs;
        }
        out[((size_t) t * numV + h) * hv + j] = o;
    }
    for (int i = 0; i < hk; i++) Sh[(size_t) i * hv + j] = col[i];
}

// ds: the [numV][hk*hv] state, resident on the card and advanced in place.
extern "C" int dg_gdn_dev(int dev, int n, int numK, int numV, int hk, int hv, float qscale, const float* conv,
                          const float* decay, const float* beta, float* ds, float* out) {
    if (dev < 0 || dev >= 16 || n <= 0 || numK <= 0 || numV <= 0 || numV % numK || hk <= 0 || hk > 128 || hv < hk || hv > 1024)
        return 100;
    if (cudaSetDevice(dev) != cudaSuccess) return 101;
    Scratch& s = g_scratch[dev];
    const size_t ch = (size_t) 2 * numK * hk + (size_t) numV * hv, cb = n * ch * 4, ob = (size_t) n * numV * hv * 4, ab = (size_t) n * numV * 4;
    if (!grow((void**) &s.x, &s.xcap, cb) || !grow((void**) &s.y, &s.ycap, ob) || !grow((void**) &s.g, &s.gcap, ab) ||
        !grow((void**) &s.u, &s.ucap, ab))
        return 102;
    if (cudaMemcpy(s.x, conv, cb, cudaMemcpyHostToDevice) || cudaMemcpy(s.g, decay, ab, cudaMemcpyHostToDevice) ||
        cudaMemcpy(s.u, beta, ab, cudaMemcpyHostToDevice))
        return 104;
    gdn_kernel<<<numV, hv>>>(s.x, s.g, s.u, ds, s.y, n, numK, numV, hk, hv, qscale);
    if (cudaGetLastError() != cudaSuccess) return 106;
    return cudaMemcpy(out, s.y, ob, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : 107;
}

// Several matrices applied to the same n inputs: x goes up once, the results come back in one buffer laid out as
// [weight k][input i][row r] (weight k's block starts at n * sum of the earlier row counts).
extern "C" int dg_multi(int dev, int cnt, const int* types, const void* const* ws, const size_t* rbs, const int* rows,
                        int cols, const float* x, int n, float* y) {
    if (dev < 0 || dev >= 16 || cnt <= 0 || cnt > 8 || cols <= 0 || n <= 0 || n > 65535) return 100;
    if (cudaSetDevice(dev) != cudaSuccess) return 101;
    const double t0 = now_s();
    Scratch& s = g_scratch[dev];
    size_t total = 0;
    for (int k = 0; k < cnt; k++) total += (size_t) rows[k];
    const size_t xb = (size_t) n * cols * sizeof(float), yb = (size_t) n * total * sizeof(float);
    if (!grow((void**) &s.x, &s.xcap, xb)) return 102;
    if (!grow((void**) &s.y, &s.ycap, yb)) return 103;
    if (cudaMemcpy(s.x, x, xb, cudaMemcpyHostToDevice) != cudaSuccess) return 104;
    g_b[0][0] += xb;
    const double t1 = now_s();
    size_t off = 0;
    for (int k = 0; k < cnt; k++) {
        if (int rc = matvec_core(s, types[k], ws[k], rbs[k], rows[k], cols, s.x, n, s.y + off)) return rc;
        off += (size_t) n * rows[k];
    }
    return finish_matvec(y, s.y, yb, t0, t1);
}
