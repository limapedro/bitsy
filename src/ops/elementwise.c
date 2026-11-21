#include <bitsy/bt_ops.h>

#include "../core/dispatcher_internal.h"
#include "../core/error_internal.h"

/* Ops are thin: validate, broadcast, allocate, dispatch. No kernel logic. */
static bt_status_t binary(const bt_tensor_t* a, const bt_tensor_t* b,
                          bt_tensor_t** out, bt_op_id_t op) {
  bt_clear_error();
  if (!a || !b || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_elementwise: NULL");

  bt_dtype_t dtype = bt_tensor_dtype(a);
  if (dtype != BT_FLOAT32 || bt_tensor_dtype(b) != BT_FLOAT32)
    return bt_fail(BT_ERR_UNSUPPORTED_DTYPE, "elementwise: float32 only for M0/M2");

  bt_device_t device = bt_tensor_device(a);
  if (!bt_device_equal(device, bt_tensor_device(b)))
    return bt_fail(BT_ERR_UNSUPPORTED_DEVICE, "elementwise: operands must share a device");

  bt_shape_t shape;
  bt_status_t status = bt_broadcast_shapes(bt_tensor_shape(a), bt_tensor_shape(b), &shape);
  if (status != BT_OK) return status;

  bt_tensor_t* result = bt_tensor_empty(shape.dims, shape.ndim, dtype, device);
  if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "elementwise: allocation failed");
  status = bt_dispatch_binary(op, bt_backend_for_device(device), dtype, a, b, result);
  if (status != BT_OK) { bt_tensor_release(result); return status; }
  *out = result;
  return BT_OK;
}

static bt_status_t unary(const bt_tensor_t* a, bt_tensor_t** out, bt_op_id_t op) {
  return binary(a, a, out, op); // b is unused by unary kernels
}

bt_status_t bt_add(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t** out) { return binary(a, b, out, BT_OP_ADD); }
bt_status_t bt_sub(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t** out) { return binary(a, b, out, BT_OP_SUB); }
bt_status_t bt_mul(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t** out) { return binary(a, b, out, BT_OP_MUL); }
bt_status_t bt_div(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t** out) { return binary(a, b, out, BT_OP_DIV); }
bt_status_t bt_relu(const bt_tensor_t* a, bt_tensor_t** out) { return unary(a, out, BT_OP_RELU); }
bt_status_t bt_sigmoid(const bt_tensor_t* a, bt_tensor_t** out) { return unary(a, out, BT_OP_SIGMOID); }
bt_status_t bt_tanh(const bt_tensor_t* a, bt_tensor_t** out) { return unary(a, out, BT_OP_TANH); }
bt_status_t bt_exp(const bt_tensor_t* a, bt_tensor_t** out) { return unary(a, out, BT_OP_EXP); }
