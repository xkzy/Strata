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
#ifdef __cplusplus
}
#endif
