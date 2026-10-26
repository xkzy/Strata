// Quantized matrix-vector products for the dense-model reference (pkg/densecpu): y[i][r] = W[r] . x[i].
// One thread block per (output row, input vector); the arithmetic mirrors the CPU kernels in dot.go.
// Build: see build.sh. No peer access, no unified memory: weights are copied once, activations per call.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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
    const dim3 grid(rows, (n + NB - 1) / NB), block(128);
    const uint8_t* wp = (const uint8_t*) w;
    if (n >= 4 && cols % 256 == 0 && (type == T_Q4_K || type == T_Q5_K || type == T_Q6_K || type == T_Q8_0)) {
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
        quant_x<<<(warps + 7) / 8, 256>>>(s.x, cols, n, s.xq, s.xd);
        switch (type) {
        case T_Q4_K: mmq<T_Q4_K><<<grid, block>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, s.y); break;
        case T_Q5_K: mmq<T_Q5_K><<<grid, block>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, s.y); break;
        case T_Q6_K: mmq<T_Q6_K><<<grid, block>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, s.y); break;
        default: mmq<T_Q8_0><<<grid, block>>>(wp, row_bytes, rows, cols, n, s.xq, s.xd, s.y); break;
        }
        if (cudaGetLastError() != cudaSuccess) return 106;
        if (cudaMemcpy(y, s.y, yb, cudaMemcpyDeviceToHost) != cudaSuccess) return 107;
        return 0;
    }
    switch (type) {
    case T_Q4_K: matvec<T_Q4_K><<<grid, block>>>(wp, row_bytes, rows, cols, n, s.x, s.y); break;
    case T_Q5_K: matvec<T_Q5_K><<<grid, block>>>(wp, row_bytes, rows, cols, n, s.x, s.y); break;
    case T_Q6_K: matvec<T_Q6_K><<<grid, block>>>(wp, row_bytes, rows, cols, n, s.x, s.y); break;
    case T_Q8_0: matvec<T_Q8_0><<<grid, block>>>(wp, row_bytes, rows, cols, n, s.x, s.y); break;
    case T_BF16: matvec<T_BF16><<<grid, block>>>(wp, row_bytes, rows, cols, n, s.x, s.y); break;
    default: return 105;
    }
    if (cudaGetLastError() != cudaSuccess) return 106;
    if (cudaMemcpy(y, s.y, yb, cudaMemcpyDeviceToHost) != cudaSuccess) return 107;
    return 0;
}
