#ifndef BITSY_BT_OPS_H
#define BITSY_BT_OPS_H

#include <bitsy/bt_export.h>
#include <bitsy/bt_error.h>
#include <bitsy/bt_tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Broadcasting elementwise ops; each allocates a new output on a's device.
 * M0 supports float32 on cpu_ref. M2 adds the same kernels on Vulkan.
 * Unsupported combinations return BT_ERR_NOT_IMPLEMENTED, never wrong data. */
BT_API bt_status_t bt_add(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t** out);
BT_API bt_status_t bt_sub(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t** out);
BT_API bt_status_t bt_mul(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t** out);
BT_API bt_status_t bt_div(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t** out);

/* Unary ops take one tensor; the dispatcher passes it as both operands. */
BT_API bt_status_t bt_relu(const bt_tensor_t* a, bt_tensor_t** out);
BT_API bt_status_t bt_sigmoid(const bt_tensor_t* a, bt_tensor_t** out);
BT_API bt_status_t bt_tanh(const bt_tensor_t* a, bt_tensor_t** out);
BT_API bt_status_t bt_exp(const bt_tensor_t* a, bt_tensor_t** out);
BT_API bt_status_t bt_log(const bt_tensor_t* a, bt_tensor_t** out);

/* First fused pointwise kernel: out = relu(a + b) in one dispatch. */
BT_API bt_status_t bt_add_relu(const bt_tensor_t* a, const bt_tensor_t* b,
                               bt_tensor_t** out);

/* Row-major 2D matrix multiply: (M,K) x (K,N) -> (M,N).
 * FP32 accumulate; output allocated by the caller of the C API. */
BT_API bt_status_t bt_matmul(const bt_tensor_t* a, const bt_tensor_t* b,
                             bt_tensor_t** out);

/* 2D transpose: (M,N) -> (N,M). Used by matmul backward. */
BT_API bt_status_t bt_transpose(const bt_tensor_t* a, bt_tensor_t** out);

/* exp_017: fused SGD update out = w - lr*grad (one dispatch, scalar lr). */
BT_API bt_status_t bt_sgd_step(const bt_tensor_t* w, const bt_tensor_t* grad,
                               float lr, bt_tensor_t** out);

/* exp_018: 2D axis reduction sum; axis 0 reduces rows, axis 1 reduces columns.
 * `keepdim` selects (1,N)/(M,1) vs (N,)/(M,) output shapes. */
BT_API bt_status_t bt_reduce_sum(const bt_tensor_t* a, uint32_t axis, int keepdim,
                                 bt_tensor_t** out);

#ifdef __cplusplus
}
#endif

#endif /* BITSY_BT_OPS_H */
