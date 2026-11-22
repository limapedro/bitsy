/* PR #50: optional CUDA-native execution for fused pointwise ops.
 * Dynamically loaded, fails soft on CUDA-less machines.
 */
#ifndef RT_CUDANATIVE_H
#define RT_CUDANATIVE_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
int rt_cn_available(void);            /* 1 if a CUDA context is usable */
void* rt_cn_alloc(size_t bytes);      /* CUDA device allocation */
int rt_cn_upload(void* dptr, const float* host, size_t bytes);
void* rt_cn_out(size_t bytes);        /* rotating output buffer (8 slots) */
void* rt_cn_run(const int* params, int pwords, int n, void* out);
/* returns device output pointer, or NULL on failure */
int rt_cn_read(const void* dptr, float* host, size_t bytes);
const float* rt_cn_ptr(const void* dptr, size_t bytes);  /* rotating pinned host */
void rt_cn_flush(void);
/* PR #55: single-kernel CUDA reduction (sum/mean/max/min). */
int rt_cn_reduce_fast(void* in,void* out,int outer,int stride,int inner,int kind);
int rt_cn_gemm_fast(void* A,void* B,void* bias,void* C,int M,int N,int K,int flags);
#ifdef __cplusplus
}
#endif
#endif
