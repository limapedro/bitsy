#include <bitsy/bt_ops.h>

#include <stdlib.h>
#include <math.h>

#ifdef BT_ENABLE_VULKAN
#include "../backends/vulkan/vk_internal.h"
#endif

#include "../core/error_internal.h"
#include "../core/dispatcher_internal.h"

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

bt_status_t bt_ce_loss(const bt_tensor_t* ls, const bt_tensor_t* onehot,
                       bt_tensor_t** out) {
  bt_clear_error();
  if (!ls || !onehot || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_ce_loss: NULL");
  if (bt_tensor_dtype(ls) != BT_FLOAT32 || bt_tensor_shape(ls)->ndim != 2 ||
      !bt_shape_equal(bt_tensor_shape(ls), bt_tensor_shape(onehot)))
    return bt_fail(BT_ERR_SHAPE_MISMATCH, "bt_ce_loss: 2D fp32 matching shapes");
  int64_t M = bt_tensor_shape(ls)->dims[0], N = bt_tensor_shape(ls)->dims[1];
  bt_device_t device = bt_tensor_device(ls);
  if (bt_backend_for_device(device) == BT_BACKEND_VULKAN) {
#ifdef BT_ENABLE_VULKAN
    int64_t dims[2] = { 1, 1 };
    bt_tensor_t* result = bt_tensor_empty(dims, 2, BT_FLOAT32, device);
    if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_ce_loss: allocation failed");
    bt_status_t status = bt_vk_ce_loss(ls, onehot, result);
    if (status != BT_OK) { bt_tensor_release(result); return status; }
    *out = result;
    return BT_OK;
#else
    return bt_fail(BT_ERR_NOT_IMPLEMENTED, "bt_ce_loss: Vulkan unavailable");
#endif
  }
  size_t count = (size_t)(M * N);
  float* a = (float*)malloc(count * sizeof(float));
  float* b = (float*)malloc(count * sizeof(float));
  if (!a || !b) { free(a); free(b); return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_ce_loss: host OOM"); }
  if (bt_tensor_copy_to_host(ls, a, count * sizeof(float)) != BT_OK ||
      bt_tensor_copy_to_host(onehot, b, count * sizeof(float)) != BT_OK) {
    free(a); free(b); return bt_fail(BT_ERR_BACKEND, "bt_ce_loss: copy failed");
  }
  float sum = 0.0f;
  for (size_t i = 0; i < count; ++i) sum += a[i] * b[i];
  float v = -sum / (float)M;
  free(a); free(b);
  int64_t dims[2] = { 1, 1 };
  bt_tensor_t* result = bt_tensor_create(dims, 2, BT_FLOAT32, device, &v);
  if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_ce_loss: result OOM");
  *out = result;
  return BT_OK;
}

bt_status_t bt_log_softmax(const bt_tensor_t* x, bt_tensor_t** out) {
  bt_clear_error();
  if (!x || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_log_softmax: NULL");
  if (bt_tensor_dtype(x) != BT_FLOAT32 || bt_tensor_shape(x)->ndim != 2)
    return bt_fail(BT_ERR_SHAPE_MISMATCH, "bt_log_softmax: 2D fp32 only");
  int64_t M = bt_tensor_shape(x)->dims[0], N = bt_tensor_shape(x)->dims[1];
  bt_device_t device = bt_tensor_device(x);
  if (bt_backend_for_device(device) == BT_BACKEND_VULKAN) {
#ifdef BT_ENABLE_VULKAN
    int64_t dims[2] = { M, N };
    bt_tensor_t* result = bt_tensor_empty(dims, 2, BT_FLOAT32, device);
    if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_log_softmax: allocation failed");
    bt_status_t status = bt_vk_log_softmax(x, result);
    if (status != BT_OK) { bt_tensor_release(result); return status; }
    *out = result;
    return BT_OK;
#else
    return bt_fail(BT_ERR_NOT_IMPLEMENTED, "bt_log_softmax: Vulkan unavailable");
#endif
  }
  size_t count = (size_t)(M * N);
  float* h = (float*)malloc(count * sizeof(float));
  if (!h) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_log_softmax: host OOM");
  if (bt_tensor_copy_to_host(x, h, count * sizeof(float)) != BT_OK) {
    free(h); return bt_fail(BT_ERR_BACKEND, "bt_log_softmax: copy failed");
  }
  for (int64_t r = 0; r < M; ++r) {
    float sum = 0.0f;
    for (int64_t c = 0; c < N; ++c) sum += expf(h[r * N + c]);
    float ls = logf(sum);
    for (int64_t c = 0; c < N; ++c) h[r * N + c] -= ls;
  }
  int64_t dims[2] = { M, N };
  bt_tensor_t* result = bt_tensor_create(dims, 2, BT_FLOAT32, device, h);
  free(h);
  if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_log_softmax: result OOM");
  *out = result;
  return BT_OK;
}

bt_status_t bt_broadcast_grad(const bt_tensor_t* g, const int64_t* dims,
                              int64_t ndim, bt_tensor_t** out) {
  bt_clear_error();
  if (!g || !dims || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_broadcast_grad: NULL");
  if (ndim != 2 || bt_tensor_shape(g)->ndim != 2)
    return bt_fail(BT_ERR_SHAPE_MISMATCH, "bt_broadcast_grad: 2D only");
  for (int i = 0; i < 2; ++i)
    if (bt_tensor_shape(g)->dims[i] != 1 && bt_tensor_shape(g)->dims[i] != dims[i])
      return bt_fail(BT_ERR_SHAPE_MISMATCH, "bt_broadcast_grad: incompatible dims");
  bt_device_t device = bt_tensor_device(g);
  if (bt_backend_for_device(device) == BT_BACKEND_VULKAN) {
#ifdef BT_ENABLE_VULKAN
    bt_tensor_t* result = bt_tensor_empty(dims, ndim, BT_FLOAT32, device);
    if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_broadcast_grad: allocation failed");
    bt_status_t status = bt_vk_broadcast_grad(g, result);
    if (status != BT_OK) { bt_tensor_release(result); return status; }
    *out = result;
    return BT_OK;
#else
    return bt_fail(BT_ERR_NOT_IMPLEMENTED, "bt_broadcast_grad: Vulkan unavailable");
#endif
  }
  int64_t M = dims[0], N = dims[1];
  int64_t gM = bt_tensor_shape(g)->dims[0], gN = bt_tensor_shape(g)->dims[1];
  size_t count = (size_t)(M * N);
  float* gh = (float*)malloc((size_t)(gM * gN) * sizeof(float));
  float* oh = (float*)malloc(count * sizeof(float));
  if (!gh || !oh) { free(gh); free(oh); return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_broadcast_grad: host OOM"); }
  if (bt_tensor_copy_to_host(g, gh, (size_t)(gM * gN) * sizeof(float)) != BT_OK) {
    free(gh); free(oh); return bt_fail(BT_ERR_BACKEND, "bt_broadcast_grad: copy failed");
  }
  for (int64_t r = 0; r < M; ++r)
    for (int64_t c = 0; c < N; ++c)
      oh[r * N + c] = gh[(gM == 1 ? 0 : r) * gN + (gN == 1 ? 0 : c)];
  bt_tensor_t* result = bt_tensor_create(dims, ndim, BT_FLOAT32, device, oh);
  free(gh); free(oh);
  if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_broadcast_grad: result OOM");
  *out = result;
  return BT_OK;
}
