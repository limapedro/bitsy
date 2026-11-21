#include <bitsy/bt_ops.h>

#include <stdlib.h>

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

bt_status_t bt_sgd_step_fused(const bt_tensor_t* w[4], const bt_tensor_t* grad[4],
                              float lr, bt_tensor_t** out[4]) {
  bt_clear_error();
  if (!w || !grad || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_sgd_step_fused: NULL");
  const bt_tensor_t* ws[4] = { w[0], w[1], w[2], w[3] };
  const bt_tensor_t* gs[4] = { grad[0], grad[1], grad[2], grad[3] };
  for (int i = 0; i < 4; ++i)
    if ((ws[i] && !gs[i]) || (!ws[i] && gs[i]))
      return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_sgd_step_fused: unpaired param/grad");
#ifdef BT_ENABLE_VULKAN
  bt_tensor_t* outs[4] = { NULL, NULL, NULL, NULL };
  int all_vk = 1, count = 0;
  for (int i = 0; i < 4; ++i) if (ws[i]) {
    ++count;
    if (bt_backend_for_device(bt_tensor_device(ws[i])) != BT_BACKEND_VULKAN) all_vk = 0;
  }
  if (all_vk && count > 0) {
    int64_t dims[8];
    for (int i = 0; i < 4; ++i) if (ws[i]) {
      const bt_shape_t* sh = bt_tensor_shape(ws[i]);
      for (int d = 0; d < sh->ndim; ++d) dims[d] = sh->dims[d];
      outs[i] = bt_tensor_empty(dims, (int)sh->ndim, BT_FLOAT32, bt_tensor_device(ws[i]));
      if (!outs[i]) { for (int j = 0; j < 4; ++j) if (outs[j]) bt_tensor_release(outs[j]);
                      return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_sgd_step_fused: allocation failed"); }
    }
    const bt_tensor_t* nullw[4] = { NULL, NULL, NULL, NULL };
    const bt_tensor_t* nullg[4] = { NULL, NULL, NULL, NULL };
    const bt_tensor_t* wo[4], *go[4], *oo[4];
    for (int i = 0; i < 4; ++i) {
      wo[i] = ws[i] ? ws[i] : nullw[i];
      go[i] = gs[i] ? gs[i] : nullg[i];
      oo[i] = outs[i] ? (const bt_tensor_t*)outs[i] : nullw[i];
    }
    bt_status_t status = bt_vk_sgd_fused(wo, go, (const bt_tensor_t**)oo, lr);
    if (status != BT_OK) { for (int j = 0; j < 4; ++j) if (outs[j]) bt_tensor_release(outs[j]); return status; }
    for (int i = 0; i < 4; ++i) if (outs[i]) *out[i] = outs[i];
    return BT_OK;
  }
  for (int j = 0; j < 4; ++j) if (outs[j]) bt_tensor_release(outs[j]);
#endif
  for (int i = 0; i < 4; ++i)
    if (ws[i]) {
      bt_status_t status = bt_sgd_step(ws[i], gs[i], lr, out[i]);
      if (status != BT_OK) return status;
    }
  return BT_OK;
}

bt_status_t bt_tanh_backward(const bt_tensor_t* g, const bt_tensor_t* t,
                             bt_tensor_t** out) {
  bt_clear_error();
  if (!g || !t || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_tanh_backward: NULL");
  if (!bt_shape_equal(bt_tensor_shape(g), bt_tensor_shape(t)))
    return bt_fail(BT_ERR_SHAPE_MISMATCH, "bt_tanh_backward: shape mismatch");
  bt_device_t device = bt_tensor_device(g);
  int64_t n = bt_shape_numel(bt_tensor_shape(g));
  if (bt_backend_for_device(device) == BT_BACKEND_VULKAN) {
#ifdef BT_ENABLE_VULKAN
    int64_t dims[8];
    for (int i = 0; i < bt_tensor_shape(g)->ndim; ++i) dims[i] = bt_tensor_shape(g)->dims[i];
    bt_tensor_t* result = bt_tensor_empty(dims, (int)bt_tensor_shape(g)->ndim, BT_FLOAT32, device);
    if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_tanh_backward: allocation failed");
    bt_status_t status = bt_vk_tanh_backward(g, t, result);
    if (status != BT_OK) { bt_tensor_release(result); return status; }
    *out = result;
    return BT_OK;
#else
    return bt_fail(BT_ERR_NOT_IMPLEMENTED, "bt_tanh_backward: Vulkan unavailable");
#endif
  }
  size_t count = (size_t)n;
  float* gh = (float*)malloc(count * sizeof(float));
  float* th = (float*)malloc(count * sizeof(float));
  float* oh = (float*)malloc(count * sizeof(float));
  if (!gh || !th || !oh) { free(gh); free(th); free(oh); return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_tanh_backward: host OOM"); }
  if (bt_tensor_copy_to_host(g, gh, count * sizeof(float)) != BT_OK ||
      bt_tensor_copy_to_host(t, th, count * sizeof(float)) != BT_OK) {
    free(gh); free(th); free(oh); return bt_fail(BT_ERR_BACKEND, "bt_tanh_backward: copy failed");
  }
  for (size_t i = 0; i < count; ++i) {
    float sq = th[i] * th[i];
    oh[i] = gh[i] * (1.0f - sq);
  }
  bt_tensor_t* result = bt_tensor_create(bt_tensor_shape(g)->dims, (int64_t)bt_tensor_shape(g)->ndim,
                                         BT_FLOAT32, device, oh);
  free(gh); free(th); free(oh);
  if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_tanh_backward: result OOM");
  *out = result;
  return BT_OK;
}

bt_status_t bt_dual_mul(const bt_tensor_t* g, const bt_tensor_t* a,
                        const bt_tensor_t* b, bt_tensor_t** out) {
  bt_clear_error();
  if (!g || !a || !b || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_dual_mul: NULL");
  int64_t n = bt_shape_numel(bt_tensor_shape(g));
  int64_t dims[2] = { 2, n };
  bt_device_t device = bt_tensor_device(g);
  if (bt_backend_for_device(device) == BT_BACKEND_VULKAN) {
#ifdef BT_ENABLE_VULKAN
    bt_tensor_t* result = bt_tensor_empty(dims, 2, BT_FLOAT32, device);
    if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_dual_mul: allocation failed");
    bt_status_t status = bt_vk_dual_mul(g, a, b, result);
    if (status != BT_OK) { bt_tensor_release(result); return status; }
    *out = result;
    return BT_OK;
#else
    return bt_fail(BT_ERR_NOT_IMPLEMENTED, "bt_dual_mul: Vulkan unavailable");
#endif
  }
  size_t count = (size_t)n;
  float* gh = (float*)malloc(count * sizeof(float));
  float* ah = (float*)malloc(count * sizeof(float));
  float* bh = (float*)malloc(count * sizeof(float));
  float* oh = (float*)malloc(2 * count * sizeof(float));
  if (!gh || !ah || !bh || !oh) { free(gh); free(ah); free(bh); free(oh); return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_dual_mul: host OOM"); }
  if (bt_tensor_copy_to_host(g, gh, count * sizeof(float)) != BT_OK ||
      bt_tensor_copy_to_host(a, ah, count * sizeof(float)) != BT_OK ||
      bt_tensor_copy_to_host(b, bh, count * sizeof(float)) != BT_OK) {
    free(gh); free(ah); free(bh); free(oh); return bt_fail(BT_ERR_BACKEND, "bt_dual_mul: copy failed");
  }
  for (size_t i = 0; i < count; ++i) { oh[i] = gh[i] * ah[i]; oh[count + i] = gh[i] * bh[i]; }
  bt_tensor_t* result = bt_tensor_create(dims, 2, BT_FLOAT32, device, oh);
  free(gh); free(ah); free(bh); free(oh);
  if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_dual_mul: result OOM");
  *out = result;
  return BT_OK;
}
