#include <bitsy/bt_ops.h>

#include "../core/dispatcher_internal.h"
#include "../core/error_internal.h"

/* Thin op: validate, broadcast, allocate, dispatch. No backend logic here. */
bt_status_t bt_add(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t** out) {
  bt_clear_error();
  if (!a || !b || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_add: NULL argument");

  bt_dtype_t dtype = bt_tensor_dtype(a);
  if (dtype != BT_FLOAT32 || bt_tensor_dtype(b) != BT_FLOAT32)
    return bt_fail(BT_ERR_UNSUPPORTED_DTYPE, "bt_add: only float32 is implemented in M0");

  bt_device_t device = bt_tensor_device(a);
  if (!bt_device_equal(device, bt_tensor_device(b)))
    return bt_fail(BT_ERR_UNSUPPORTED_DEVICE, "bt_add: operands must share a device");
  if (device.type == BT_DEVICE_TYPE_VULKAN)
    return bt_fail(BT_ERR_NOT_IMPLEMENTED, "bt_add: the vulkan backend arrives in M2");

  bt_shape_t shape;
  bt_status_t status = bt_broadcast_shapes(bt_tensor_shape(a), bt_tensor_shape(b), &shape);
  if (status != BT_OK) return status;

  bt_tensor_t* result = bt_tensor_empty(shape.dims, shape.ndim, dtype, device);
  if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_add: could not allocate output");

  status = bt_dispatch_binary(BT_OP_ADD, bt_backend_for_device(device), dtype, a, b, result);
  if (status != BT_OK) { bt_tensor_release(result); return status; }
  *out = result;
  return BT_OK;
}
