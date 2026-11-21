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

#ifdef __cplusplus
}
#endif

#endif /* BITSY_BT_OPS_H */
