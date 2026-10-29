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

typedef struct { int type; const void* w; size_t rb; int rows, cols; } dg_mat;
// One layer of the Qwen3.8 dense model, everything on the device. attn == 0: gated delta-net, m = {attn_qkv, attn_gate,
// ssm_beta, ssm_alpha, ssm_out}, f = {ssm_dt, ssm_a, ssm_conv1d, ssm_norm}, S/tail are its state. attn == 1: m = {q, k, v, output},
// f = {q_norm, k_norm}, Kc/Vc the half caches (cap positions x n_kv x hd).
typedef struct {
  int attn;
  const float *attn_norm, *post_norm;
  dg_mat ffn_gate, ffn_up, ffn_down;
  dg_mat m[5];
  const float* f[4];
  float *S, *tail, *S1, *tail1;   // S1/tail1: the state after the first token of a snapshotting call
  void *Kc, *Vc;
} dg_layer;
typedef struct { int n_embd, n_head, n_kv, hd, n_rot, hk, nk, nv, conv_k, d_inner, n_ff, cap; float eps, base; } dg_cfg;
void* dg_alloc_zero(int dev, size_t bytes);
int dg_zero(int dev, void* p, size_t bytes);
double dg_bandwidth(int dev);
void dg_prof_dump(void);
// snap != 0 saves the delta-net state after the first token (dg_qwen_restore puts it back)
int dg_qwen_run(int dev, const dg_cfg* c, const dg_layer* L, int nl, float* h, int n, int pos0, int snap);
int dg_qwen_restore(int dev, const dg_cfg* c, const dg_layer* L, int nl);
// Causal attention for n new tokens over a half-precision cache of first+n rows already on card `dev` (dk, dv); new
// token i is row first+i and sees rows max(0, first+i-window+1)..first+i (window 0: all earlier rows). cap 0: row j is
// at slot j; otherwise a ring buffer of cap slots where row j (position b0+j) is at slot (b0+j) % cap.
// q, out: [n][nHead*hd] floats on the host; hd is a multiple of 32, at most 512. 0 on success.
int dg_attn_dev(int dev, int n, int first, int window, int b0, int cap, int nHead, int nKV, int hd, float scale, const float* q, const void* dk,
                const void* dv, float* out);
// Device memory for the resident KV cache and delta-net state. dg_alloc returns NULL on failure; the copies return 0.
void* dg_alloc(int dev, size_t bytes);
int dg_h2d(int dev, void* dst, size_t off, const void* src, size_t bytes);
int dg_d2h(int dev, void* dst, const void* src, size_t bytes);
int dg_d2d(int dev, void* dst, const void* src, size_t bytes);
// Gated delta-net recurrence over n tokens for numV value heads (key heads tiled: head h uses key head h % numK).
// conv: [n][ch] host rows holding q (numK*hk, already L2-normalised), k (same), v (numV*hv); decay, beta: [n][numV];
// ds: the [numV][hk*hv] state on card `dev`, updated in place; out: [n][numV*hv]. hk <= 128, hv <= 1024. 0 on success.
int dg_gdn_dev(int dev, int n, int numK, int numV, int hk, int hv, float qscale, const float* conv, const float* decay,
               const float* beta, float* ds, float* out);
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
