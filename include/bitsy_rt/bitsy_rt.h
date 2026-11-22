/* Bitsy 0.2 — compact lazy runtime (tinygrad-inspired).
 *
 * One interned node type, a fusion scheduler, parameterized kernels, and a
 * capture/replay executor. This header is the entire public C surface.
 */
#ifndef BITSY_RT_H
#define BITSY_RT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(BITSY_RT_BUILD_SHARED)
#  define RT_API __declspec(dllexport)
#elif defined(_WIN32)
#  define RT_API __declspec(dllimport)
#else
#  define RT_API __attribute__((visibility("default")))
#endif

typedef struct RT RT;
typedef struct RTTensor RTTensor;

/* Device: 0 = cpu reference, 1 = vulkan compute. */
#define RT_CPU 0
#define RT_VULKAN 1

/* Ops. Pointwise ops fuse; cut ops materialise a buffer. */
typedef enum {
  RT_ADD=0, RT_SUB, RT_MUL, RT_DIV, RT_NEG,
  RT_EXP, RT_LOG, RT_SQRT, RT_TANH, RT_SIGM, RT_RELU, RT_GELU,
  RT_POW, RT_MAXIMUM, RT_MINIMUM, RT_ABS, RT_SIGN,
  RT_GT, RT_GE, RT_LT, RT_LE, RT_EQ, RT_NE, RT_WHERE,
  RT_MATMUL, RT_REDUCE, RT_TRANSPOSE, RT_RESHAPE,
  /* internal derivative primitives (fuse like pointwise) */
  RT_GT0, RT_GELUB,
  RT_GEMM_RELU,  /* PR #23: GEMM + bias + ReLU in one dispatch */
  RT_SOFTMAX,    /* PR #24: fused softmax/log_softmax along last axis */
  RT_LAYERNORM   /* PR #25: fused LayerNorm along last axis */
} RTOp;

/* Reduce kind (rt_reduce). */
#define RT_REDUCE_SUM  0
#define RT_REDUCE_MEAN 1
#define RT_REDUCE_MAX  2
#define RT_REDUCE_MIN  3

/* Context lifecycle */
RT_API RT* rt_create(int device);
RT_API void rt_destroy(RT*);
RT_API int  rt_device(const RT*);
RT_API const char* rt_last_error(const RT*);
RT_API int  rt_dispatch_count(const RT*);   /* # dispatches since last reset */
RT_API void rt_reset_dispatch_count(RT*);

/* Tensor construction (host data copied into the chosen device) */
RT_API RTTensor* rt_from_host(RT*, const float* data, int ndim,
                              const int64_t* dims, int requires_grad);
RT_API RTTensor* rt_zeros(RT*, int ndim, const int64_t* dims);
RT_API void rt_release(RTTensor*);

/* Lazy ops — build graph nodes, no compute happens here.
 * The context is threaded explicitly so multiple contexts can coexist. */
RT_API RTTensor* rt_add(RT*, RTTensor*, RTTensor*);
RT_API RTTensor* rt_sub(RT*, RTTensor*, RTTensor*);
RT_API RTTensor* rt_mul(RT*, RTTensor*, RTTensor*);
RT_API RTTensor* rt_div(RT*, RTTensor*, RTTensor*);
RT_API RTTensor* rt_neg(RT*, RTTensor*);
RT_API RTTensor* rt_exp(RT*, RTTensor*);
RT_API RTTensor* rt_log(RT*, RTTensor*);
RT_API RTTensor* rt_sqrt(RT*, RTTensor*);
RT_API RTTensor* rt_tanh(RT*, RTTensor*);
RT_API RTTensor* rt_sigm(RT*, RTTensor*);
RT_API RTTensor* rt_relu(RT*, RTTensor*);
RT_API RTTensor* rt_gelu(RT*, RTTensor*);
/* PR #2 additional pointwise primitives (all fuse into the op-program) */
RT_API RTTensor* rt_pow(RT*, RTTensor*, RTTensor*);
RT_API RTTensor* rt_maximum(RT*, RTTensor*, RTTensor*);
RT_API RTTensor* rt_minimum(RT*, RTTensor*, RTTensor*);
RT_API RTTensor* rt_abs(RT*, RTTensor*);
RT_API RTTensor* rt_sign(RT*, RTTensor*);
RT_API RTTensor* rt_gt(RT*, RTTensor*, RTTensor*);
RT_API RTTensor* rt_ge(RT*, RTTensor*, RTTensor*);
RT_API RTTensor* rt_lt(RT*, RTTensor*, RTTensor*);
RT_API RTTensor* rt_le(RT*, RTTensor*, RTTensor*);
RT_API RTTensor* rt_eq(RT*, RTTensor*, RTTensor*);
RT_API RTTensor* rt_ne(RT*, RTTensor*, RTTensor*);
RT_API RTTensor* rt_where(RT*, RTTensor* cond, RTTensor* a, RTTensor* b);
/* GEMM with optional bias epilogue and transpose flags (PR #2) */
RT_API RTTensor* rt_gemm(RT*, RTTensor* a, RTTensor* b, RTTensor* bias,
                         int trans_a, int trans_b);
RT_API RTTensor* rt_matmul(RT*, RTTensor*, RTTensor*);
/* Generic reduce: kind = RT_REDUCE_SUM/MEAN/MAX/MIN */
RT_API RTTensor* rt_reduce(RT*, RTTensor*, int kind, int axis, int keepdim);
RT_API RTTensor* rt_softmax(RT*, RTTensor*, int kind);  /* kind 0=softmax, 1=log_softmax */
RT_API RTTensor* rt_layernorm(RT*, RTTensor* x, RTTensor* w, RTTensor* b, float eps);
RT_API RTTensor* rt_reduce_sum(RT*, RTTensor*, int axis, int keepdim);
RT_API RTTensor* rt_transpose(RT*, RTTensor*);
RT_API RTTensor* rt_reshape(RT*, RTTensor*, int ndim, const int64_t* dims);
/* scalar constant tensor */
RT_API RTTensor* rt_scalar(RT*, float value);

/* Execution + data */
RT_API void rt_realize(RT*, RTTensor*);
RT_API void rt_to_host(RTTensor*, float* out);
RT_API void rt_upload(RT*, RTTensor*, const float* data);  /* update leaf data in place */
RT_API int  rt_ndim(const RTTensor*);
RT_API void rt_shape(const RTTensor*, int64_t* out);

/* Autograd (in C) */
RT_API void rt_backward(RT*, RTTensor*);
RT_API RTTensor* rt_grad(const RTTensor*);

/* JIT capture/replay: record the dispatches of a subgraph, replay with new data */
RT_API void rt_capture_begin(RT*);
RT_API void rt_capture_end(RT*);
RT_API void rt_replay(RT*);
RT_API void rt_replay_submit_only(RT*);  /* submit + wait, skip output download (profiling) */
RT_API void rt_replay_submit_async(RT*); /* submit without wait; read waits automatically */

/* Zero-copy readback: returns a pointer to a persistent HOST_CACHED buffer
 * holding t's data (downloads if dirty). Valid for N_READBACK_SLOT consecutive
 * calls; each call rotates to a distinct buffer. The pointer stays valid until
 * the RT context is destroyed. */
#define RT_READBACK_SLOTS 32
RT_API const float* rt_read_ptr(RT*, RTTensor*, int slot);

/* Version */
RT_API const char* rt_version(void);

#ifdef __cplusplus
}
#endif
#endif
