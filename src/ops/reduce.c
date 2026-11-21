#include <bitsy/bt_ops.h>

#include <stdlib.h>

#include "../core/error_internal.h"
#include "../core/dispatcher_internal.h"

#ifdef BT_ENABLE_VULKAN
#include "../backends/vulkan/vk_internal.h"
#endif

/* exp_018: dedicated 2D reduction. The CPU path loops in the same order as
 * the shader so Vulkan and CPU parity stay consistent. */
bt_status_t bt_reduce_sum(const bt_tensor_t* a, uint32_t axis, int keepdim,
                          bt_tensor_t** out) {
  bt_clear_error();
  if (!a || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_reduce_sum: NULL");
  if (bt_tensor_dtype(a) != BT_FLOAT32)
    return bt_fail(BT_ERR_UNSUPPORTED_DTYPE, "bt_reduce_sum: float32 only");
  const bt_shape_t* shape = bt_tensor_shape(a);
  if (shape->ndim != 2 || axis > 1)
    return bt_fail(BT_ERR_SHAPE_MISMATCH, "bt_reduce_sum: 2D input, axis 0/1 only");
  int64_t M = shape->dims[0], N = shape->dims[1];
  int64_t dims[2];
  int64_t ndim;
  if (keepdim) {
    dims[0] = axis == 0 ? 1 : M;
    dims[1] = axis == 0 ? N : 1;
    ndim = 2;
  } else {
    dims[0] = axis == 0 ? N : M;
    ndim = 1;
  }
  bt_device_t device = bt_tensor_device(a);

  if (bt_backend_for_device(device) == BT_BACKEND_VULKAN) {
#ifdef BT_ENABLE_VULKAN
    bt_tensor_t* result = bt_tensor_empty(dims, ndim, BT_FLOAT32, device);
    if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_reduce_sum: allocation failed");
    bt_status_t status = bt_vk_reduce_sum(a, axis, result);
    if (status != BT_OK) { bt_tensor_release(result); return status; }
    *out = result;
    return BT_OK;
#else
    return bt_fail(BT_ERR_NOT_IMPLEMENTED, "bt_reduce_sum: Vulkan unavailable");
#endif
  }

  size_t count = (size_t)(M * N);
  float* host = (float*)malloc(count * sizeof(float));
  if (!host) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_reduce_sum: host OOM");
  if (bt_tensor_copy_to_host(a, host, count * sizeof(float)) != BT_OK) {
    free(host);
    return bt_fail(BT_ERR_BACKEND, "bt_reduce_sum: copy failed");
  }
  int64_t outn = dims[0] * (ndim == 2 ? dims[1] : 1);
  float* sums = (float*)malloc((size_t)outn * sizeof(float));
  if (!sums) { free(host); return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_reduce_sum: sums OOM"); }
  if (axis == 1) {
    for (int64_t i = 0; i < M; ++i) {
      float s = 0.0f;
      for (int64_t j = 0; j < N; ++j) s += host[i * N + j];
      sums[i] = s;
    }
  } else {
    for (int64_t j = 0; j < N; ++j) {
      float s = 0.0f;
      for (int64_t i = 0; i < M; ++i) s += host[i * N + j];
      sums[j] = s;
    }
  }
  bt_tensor_t* result = bt_tensor_create(dims, ndim, BT_FLOAT32, device, sums);
  free(sums);
  free(host);
  if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_reduce_sum: result OOM");
  *out = result;
  return BT_OK;
}
