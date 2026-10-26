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

// K-quants: one work item = one 64-element chunk (two 32-element groups) of a 256-element block.
// K-quants: one work item = 8 consecutive bytes of one 32-byte chunk, i.e. 8 low-nibble and 8 high-nibble weights
// (16 items per 256-element block). Contributions add up, so items are independent.
template <bool FIVE>
__device__ float dot_k45(const uint8_t* row, const float* x, int cols, int tid, int nt) {
    const int bytes = FIVE ? 176 : 144, nb = cols / 256;
    float total = 0.f;
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
        const float* x1 = x + b * 256 + j * 64 + k * 8;
        const float* x2 = x1 + 32;
        float a1 = 0.f, a2 = 0.f, b1 = 0.f, b2 = 0.f;
#pragma unroll
        for (int l = 0; l < 8; l++) {
            const uint32_t qw = l < 4 ? qv.x : qv.y, hw = l < 4 ? hv.x : hv.y;
            const uint32_t qb = (qw >> (8 * (l & 3))) & 0xff, hb = (hw >> (8 * (l & 3))) & 0xff;
            float v1 = (float) (qb & 0xF), v2 = (float) (qb >> 4);
            if (FIVE) {
                if (hb & (1u << (2 * j))) v1 += 16.f;
                if (hb & (2u << (2 * j))) v2 += 16.f;
            }
            const float xa = __ldg(x1 + l), xb2 = __ldg(x2 + l);
            a1 += v1 * xa;
            a2 += v2 * xb2;
            b1 += xa;
            b2 += xb2;
        }
        total += d * ((float) s1 * a1 + (float) s2 * a2) - dmin * ((float) m1 * b1 + (float) m2 * b2);
    }
    return total;
}

// Q6_K: one work item = 8 consecutive l of one 128-element half (8 items per 256-element block).
__device__ float dot_q6k(const uint8_t* row, const float* x, int cols, int tid, int nt) {
    const int nb = cols / 256;
    float total = 0.f;
    for (int it = tid; it < nb * 8; it += nt) {
        const int b = it >> 3, n = (it >> 2) & 1, g = it & 3;
        const uint8_t* blk = row + (size_t) b * 210;
        const uint8_t* ql = blk + n * 64 + g * 8;
        const uint8_t* qh = blk + 128 + n * 32 + g * 8;
        const int8_t* sc = (const int8_t*) (blk + 192 + n * 8);
        const int is = g >> 1;
        const float* xn = x + b * 256 + n * 128 + g * 8;
        float a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f;
#pragma unroll
        for (int l = 0; l < 8; l++) {
            const int lo0 = ql[l], lo1 = ql[l + 32], h = qh[l];
            a1 += (float) (((lo0 & 0xF) | (((h >> 0) & 3) << 4)) - 32) * __ldg(xn + l);
            a2 += (float) (((lo1 & 0xF) | (((h >> 2) & 3) << 4)) - 32) * __ldg(xn + l + 32);
            a3 += (float) (((lo0 >> 4) | (((h >> 4) & 3) << 4)) - 32) * __ldg(xn + l + 64);
            a4 += (float) (((lo1 >> 4) | (((h >> 6) & 3) << 4)) - 32) * __ldg(xn + l + 96);
        }
        total += h2f(blk + 208) * ((float) sc[is] * a1 + (float) sc[is + 2] * a2 + (float) sc[is + 4] * a3 +
                                   (float) sc[is + 6] * a4);
    }
    return total;
}

__device__ float dot_q8(const uint8_t* row, const float* x, int cols, int tid, int nt) {
    float total = 0.f;
    for (int it = tid; it < cols / 8; it += nt) { // 8 weights per item, 4 items per 34-byte block
        const int b = it >> 2, q = it & 3;
        const uint8_t* blk = row + (size_t) b * 34;
        const float* xb = x + b * 32 + q * 8;
        float a = 0.f;
#pragma unroll
        for (int l = 0; l < 8; l++) a += (float) (int8_t) blk[2 + q * 8 + l] * __ldg(xb + l);
        total += h2f(blk) * a;
    }
    return total;
}

__device__ float dot_bf16(const uint8_t* row, const float* x, int cols, int tid, int nt) {
    float total = 0.f;
    for (int it = tid; it < cols / 8; it += nt) { // 8 weights (16 bytes) per item; cols % 8 == 0
        const uint4 w = *(const uint4*) (row + (size_t) it * 16);
        const uint32_t u[4] = {w.x, w.y, w.z, w.w};
        const float* xi = x + it * 8;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            total += __uint_as_float(u[k] << 16) * __ldg(xi + 2 * k);
            total += __uint_as_float(u[k] & 0xffff0000u) * __ldg(xi + 2 * k + 1);
        }
    }
    return total;
}

template <int TYPE>
__global__ void matvec(const uint8_t* w, size_t row_bytes, int rows, int cols, const float* x, float* y) {
    const int r = blockIdx.x, i = blockIdx.y;
    const uint8_t* row = w + (size_t) r * row_bytes;
    const float* xi = x + (size_t) i * cols;
    float s;
    if (TYPE == T_Q4_K) s = dot_k45<false>(row, xi, cols, threadIdx.x, blockDim.x);
    else if (TYPE == T_Q5_K) s = dot_k45<true>(row, xi, cols, threadIdx.x, blockDim.x);
    else if (TYPE == T_Q6_K) s = dot_q6k(row, xi, cols, threadIdx.x, blockDim.x);
    else if (TYPE == T_Q8_0) s = dot_q8(row, xi, cols, threadIdx.x, blockDim.x);
    else s = dot_bf16(row, xi, cols, threadIdx.x, blockDim.x);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    __shared__ float part[32];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int k = 0; k < (int) (blockDim.x >> 5); k++) t += part[k];
        y[(size_t) i * rows + r] = t;
    }
}

struct Scratch {
    float *x = nullptr, *y = nullptr;
    size_t xcap = 0, ycap = 0;
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
    const dim3 grid(rows, n), block(128);
    const uint8_t* wp = (const uint8_t*) w;
    switch (type) {
    case T_Q4_K: matvec<T_Q4_K><<<grid, block>>>(wp, row_bytes, rows, cols, s.x, s.y); break;
    case T_Q5_K: matvec<T_Q5_K><<<grid, block>>>(wp, row_bytes, rows, cols, s.x, s.y); break;
    case T_Q6_K: matvec<T_Q6_K><<<grid, block>>>(wp, row_bytes, rows, cols, s.x, s.y); break;
    case T_Q8_0: matvec<T_Q8_0><<<grid, block>>>(wp, row_bytes, rows, cols, s.x, s.y); break;
    case T_BF16: matvec<T_BF16><<<grid, block>>>(wp, row_bytes, rows, cols, s.x, s.y); break;
    default: return 105;
    }
    if (cudaGetLastError() != cudaSuccess) return 106;
    if (cudaMemcpy(y, s.y, yb, cudaMemcpyDeviceToHost) != cudaSuccess) return 107;
    return 0;
}
