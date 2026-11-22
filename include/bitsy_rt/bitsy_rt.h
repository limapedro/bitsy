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
  RT_MATMUL, RT_REDUCE_SUM, RT_TRANSPOSE,
  /* internal derivative primitives (fuse like pointwise) */
  RT_GT0, RT_GELUB
} RTOp;

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
RT_API RTTensor* rt_matmul(RT*, RTTensor*, RTTensor*);
RT_API RTTensor* rt_reduce_sum(RT*, RTTensor*, int axis, int keepdim);
RT_API RTTensor* rt_transpose(RT*, RTTensor*);

/* Execution + data */
RT_API void rt_realize(RT*, RTTensor*);
RT_API void rt_to_host(RTTensor*, float* out);
RT_API int  rt_ndim(const RTTensor*);
RT_API void rt_shape(const RTTensor*, int64_t* out);

/* Autograd (in C) */
RT_API void rt_backward(RT*, RTTensor*);
RT_API RTTensor* rt_grad(const RTTensor*);

/* JIT capture/replay: record the dispatches of a subgraph, replay with new data */
RT_API void rt_capture_begin(RT*);
RT_API void rt_capture_end(RT*);
RT_API void rt_replay(RT*);

/* Version */
RT_API const char* rt_version(void);

#ifdef __cplusplus
}
#endif
#endif
