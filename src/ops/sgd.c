#include <bitsy/bt_ops.h>

#include "../core/error_internal.h"
#include "../core/dispatcher_internal.h"

#ifdef BT_ENABLE_VULKAN
#include "../backends/vulkan/vk_internal.h"
#endif

/* exp_017: fused SGD weight update. Vulkan issues one dispatch with lr as a
 * push constant; the CPU reference uses existing ops with the same fp32
 * multiply-then-subtract order so parity is preserved. */
bt_status_t bt_sgd_step(const bt_tensor_t* w, const bt_tensor_t* grad,
                        float lr, bt_tensor_t** out) {
  bt_clear_error();
  if (!w || !grad || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_sgd_step: NULL");
  if (bt_tensor_dtype(w) != BT_FLOAT32 || bt_tensor_dtype(grad) != BT_FLOAT32)
    return bt_fail(BT_ERR_UNSUPPORTED_DTYPE, "bt_sgd_step: float32 only");
  if (!bt_shape_equal(bt_tensor_shape(w), bt_tensor_shape(grad)))
    return bt_fail(BT_ERR_SHAPE_MISMATCH, "bt_sgd_step: shape mismatch");
  bt_device_t device = bt_tensor_device(w);

  if (bt_backend_for_device(device) == BT_BACKEND_VULKAN) {
#ifdef BT_ENABLE_VULKAN
    const bt_shape_t* shape = bt_tensor_shape(w);
    int64_t dims[8];
    for (int i = 0; i < shape->ndim; ++i) dims[i] = shape->dims[i];
    bt_tensor_t* result = bt_tensor_empty(dims, (int)shape->ndim, BT_FLOAT32, device);
    if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_sgd_step: allocation failed");
    bt_status_t status = bt_vk_sgd_step(w, grad, lr, result);
    if (status != BT_OK) { bt_tensor_release(result); return status; }
    *out = result;
    return BT_OK;
#else
    return bt_fail(BT_ERR_NOT_IMPLEMENTED, "bt_sgd_step: Vulkan unavailable");
#endif
  }

  int64_t one = 1;
  bt_tensor_t* lrt = bt_tensor_create(&one, 1, BT_FLOAT32, device, &lr);
  if (!lrt) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_sgd_step: lr tensor OOM");
  bt_tensor_t* scaled = NULL;
  bt_status_t status = bt_mul(grad, lrt, &scaled);
  bt_tensor_release(lrt);
  if (status != BT_OK) return status;
  status = bt_sub(w, scaled, out);
  bt_tensor_release(scaled);
  return status;
}
