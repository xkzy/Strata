#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
int dg_device_count(void);
int dg_mem(int dev, size_t* free_bytes, size_t* total_bytes);
void* dg_upload(int dev, const void* host, size_t bytes);
void dg_free(int dev, void* p);
// y[i*rows + r] = row r of the quantized matrix w (ggml type `type`) dot x[i*cols .. ]; 0 on success.
int dg_matvec(int dev, int type, const void* w, size_t row_bytes, int rows, int cols, const float* x, int n, float* y);
// Causal attention for n new tokens over a half-precision cache of first+n rows; new token i is row first+i and sees
// rows max(0, first+i-window+1)..first+i (window 0: all earlier rows).
// q, out: [n][nHead*hd] floats; k, v: [first+n][nKV*hd] half bits. hd is a multiple of 32, at most 512. 0 on success.
int dg_attn(int dev, int n, int first, int window, int nHead, int nKV, int hd, float scale, const float* q, const unsigned short* k,
            const unsigned short* v, float* out);
// Gated delta-net recurrence over n tokens for numV value heads (key heads tiled: head h uses key head h % numK).
// conv: [n][ch] rows holding q (numK*hk, already L2-normalised), k (same), v (numV*hv); decay, beta: [n][numV];
// s: [numV][hk*hv] state, updated in place; out: [n][numV*hv]. hk <= 128, hv <= 1024. 0 on success.
int dg_gdn(int dev, int n, int numK, int numV, int hk, int hv, float qscale, const float* conv, const float* decay,
           const float* beta, float* s, float* out);
// Diagnostics for dg_matvec: seconds {copy in, kernel, copy out}, bytes {in, out} and calls, since the last reset.
void dg_stats(double* t, double* b, long* calls);
void dg_stats_reset(void);
// y = down(act(gate x) * up x) for n inputs, entirely on card `dev` (act 0: silu, 1: gelu-tanh). The three weights are
// resident on dev; only x goes in and y comes out. Types and row sizes as for dg_matvec. 0 on success.
int dg_ffn(int dev, int act, int tg, const void* wg, size_t rbg, int tu, const void* wu, size_t rbu, int td, const void* wd,
           size_t rbd, int n_ff, int n_embd, const float* x, int n, float* y);
// cnt (<= 8) matrices of the same width `cols` on card dev applied to the same x: x goes up once. y holds weight 0's
// [n][rows[0]] block, then weight 1's, and so on. 0 on success.
int dg_multi(int dev, int cnt, const int* types, const void* const* ws, const size_t* rbs, const int* rows, int cols,
             const float* x, int n, float* y);
#ifdef __cplusplus
}
#endif
