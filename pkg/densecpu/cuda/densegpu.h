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
#ifdef __cplusplus
}
#endif
