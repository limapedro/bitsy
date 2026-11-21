#include <bitsy/bt_ops.h>

#include "../core/dispatcher_internal.h"
#include "../core/error_internal.h"

/* 2D row-major matmul. Batched matmul comes later; no silent wrong shapes. */
bt_status_t bt_matmul(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t** out) {
  bt_clear_error();
  if (!a || !b || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_matmul: NULL");
  bt_dtype_t dtype = bt_tensor_dtype(a);
  if (dtype != BT_FLOAT32 || bt_tensor_dtype(b) != BT_FLOAT32)
    return bt_fail(BT_ERR_UNSUPPORTED_DTYPE, "bt_matmul: float32 only");
  bt_device_t device = bt_tensor_device(a);
  if (!bt_device_equal(device, bt_tensor_device(b)))
    return bt_fail(BT_ERR_UNSUPPORTED_DEVICE, "bt_matmul: operands must share a device");
  if (bt_tensor_shape(a)->ndim != 2 || bt_tensor_shape(b)->ndim != 2)
    return bt_fail(BT_ERR_SHAPE_MISMATCH, "bt_matmul: expects 2D operands");
  if (bt_tensor_shape(a)->dims[1] != bt_tensor_shape(b)->dims[0])
    return bt_fail(BT_ERR_SHAPE_MISMATCH, "bt_matmul: inner dimensions disagree");

  int64_t dims[2] = { bt_tensor_shape(a)->dims[0], bt_tensor_shape(b)->dims[1] };
  bt_tensor_t* result = bt_tensor_empty(dims, 2, dtype, device);
  if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_matmul: allocation failed");
  bt_status_t status = bt_dispatch_binary(BT_OP_MATMUL, bt_backend_for_device(device),
                                          dtype, a, b, result);
  if (status != BT_OK) { bt_tensor_release(result); return status; }
  *out = result;
  return BT_OK;
}

bt_status_t bt_transpose(const bt_tensor_t* a, bt_tensor_t** out) {
  bt_clear_error();
  if (!a || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_transpose: NULL");
  if (bt_tensor_dtype(a) != BT_FLOAT32)
    return bt_fail(BT_ERR_UNSUPPORTED_DTYPE, "bt_transpose: float32 only");
  if (bt_tensor_shape(a)->ndim != 2)
    return bt_fail(BT_ERR_SHAPE_MISMATCH, "bt_transpose: expects 2D input");
  int64_t dims[2] = { bt_tensor_shape(a)->dims[1], bt_tensor_shape(a)->dims[0] };
  bt_device_t device = bt_tensor_device(a);
  bt_tensor_t* result = bt_tensor_empty(dims, 2, BT_FLOAT32, device);
  if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_transpose: allocation failed");
  bt_status_t status = bt_dispatch_binary(BT_OP_TRANSPOSE, bt_backend_for_device(device),
                                          BT_FLOAT32, a, a, result);
  if (status != BT_OK) { bt_tensor_release(result); return status; }
  *out = result;
  return BT_OK;
}
