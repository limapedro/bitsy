#include <bitsy/bt_ops.h>

#include <stdlib.h>
#include <math.h>

#include "../core/dispatcher_internal.h"
#include "../core/error_internal.h"

#ifdef BT_ENABLE_VULKAN
#include "../backends/vulkan/vk_internal.h"
#endif

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
bt_status_t bt_log(const bt_tensor_t* a, bt_tensor_t** out) { return unary(a, out, BT_OP_LOG); }
static float gelu_erf(float xv) {
  float s = xv < 0.0f ? -1.0f : 1.0f;
  float ax = fabsf(xv);
  float t = 1.0f / (1.0f + 0.3275911f * ax);
  float p = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t
              - 0.284496736f) * t + 0.254829592f) * t;
  return s * (1.0f - p * expf(-ax * ax));
}

static bt_status_t gelu_common(const bt_tensor_t* x, const bt_tensor_t* g,
                               uint32_t mode, bt_tensor_t** out) {
  bt_clear_error();
  if (!x || !out || (mode == 1 && !g))
    return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_gelu: NULL");
  bt_device_t dev = bt_tensor_device(x);
  if (mode == 1 && !bt_device_equal(dev, bt_tensor_device(g)))
    return bt_fail(BT_ERR_UNSUPPORTED_DEVICE, "bt_gelu: device mismatch");
#ifdef BT_ENABLE_VULKAN
  if (dev.type == BT_DEVICE_TYPE_VULKAN) {
    int64_t dims[8];
    for (int d = 0; d < bt_tensor_shape(x)->ndim; ++d) dims[d] = bt_tensor_shape(x)->dims[d];
    bt_tensor_t* r = bt_tensor_empty(dims, (int)bt_tensor_shape(x)->ndim, BT_FLOAT32, dev);
    if (!r) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_gelu: allocation failed");
    bt_status_t st = bt_vk_gelu(x, g, mode, r);
    if (st != BT_OK) { bt_tensor_release(r); return st; }
    *out = r;
    return BT_OK;
  }
#endif
  int64_t n = bt_shape_numel(bt_tensor_shape(x));
  float* xh = (float*)malloc((size_t)n * sizeof(float));
  float* gh = mode == 1 ? (float*)malloc((size_t)n * sizeof(float)) : NULL;
  if (!xh || (mode == 1 && !gh)) {
    free(xh); free(gh); return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_gelu: OOM");
  }
  if (bt_tensor_copy_to_host(x, xh, (size_t)n * sizeof(float)) != BT_OK ||
      (mode == 1 && bt_tensor_copy_to_host(g, gh, (size_t)n * sizeof(float)) != BT_OK)) {
    free(xh); free(gh); return bt_fail(BT_ERR_BACKEND, "bt_gelu: copy failed");
  }
  for (int64_t i = 0; i < n; ++i) {
    float e = gelu_erf(xh[i] * 0.7071067811865476f);
    float base = 0.5f * (1.0f + e);
    if (mode == 0) xh[i] *= base;
    else {
      float phi = 0.3989422804014327f * expf(-0.5f * xh[i] * xh[i]);
      xh[i] = gh[i] * (base + xh[i] * phi);
    }
  }
  free(gh);
  int64_t dims[8];
  for (int d = 0; d < bt_tensor_shape(x)->ndim; ++d) dims[d] = bt_tensor_shape(x)->dims[d];
  bt_tensor_t* r = bt_tensor_create(dims, (int)bt_tensor_shape(x)->ndim, BT_FLOAT32, dev, xh);
  free(xh);
  if (!r) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_gelu: result OOM");
  *out = r;
  return BT_OK;
}

bt_status_t bt_gelu(const bt_tensor_t* x, bt_tensor_t** out) {
  return gelu_common(x, NULL, 0, out);
}

bt_status_t bt_gelu_backward(const bt_tensor_t* g, const bt_tensor_t* x,
                             bt_tensor_t** out) {
  return gelu_common(x, g, 1, out);
}

bt_status_t bt_relu_backward(const bt_tensor_t* g, const bt_tensor_t* out_fwd,
                             bt_tensor_t** out) {
  bt_clear_error();
  if (!g || !out_fwd || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_relu_backward: NULL");
  int64_t n = bt_shape_numel(bt_tensor_shape(g));
  if (!bt_shape_equal(bt_tensor_shape(g), bt_tensor_shape(out_fwd)))
    return bt_fail(BT_ERR_SHAPE_MISMATCH, "bt_relu_backward: shape mismatch");
#ifdef BT_ENABLE_VULKAN
  if (bt_tensor_device(g).type == BT_DEVICE_TYPE_VULKAN) {
    int64_t dims[8];
    for (int d = 0; d < bt_tensor_shape(g)->ndim; ++d) dims[d] = bt_tensor_shape(g)->dims[d];
    bt_tensor_t* result = bt_tensor_empty(dims, (int)bt_tensor_shape(g)->ndim,
                                          BT_FLOAT32, bt_tensor_device(g));
    if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_relu_backward: allocation failed");
    bt_status_t status = bt_vk_elementwise(g, out_fwd, result, 11);
    if (status != BT_OK) { bt_tensor_release(result); return status; }
    *out = result;
    return BT_OK;
  }
#endif
  float* gh = (float*)malloc((size_t)n * sizeof(float));
  float* oh = (float*)malloc((size_t)n * sizeof(float));
  if (!gh || !oh) { free(gh); free(oh); return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_relu_backward: OOM"); }
  if (bt_tensor_copy_to_host(g, gh, (size_t)n * sizeof(float)) != BT_OK ||
      bt_tensor_copy_to_host(out_fwd, oh, (size_t)n * sizeof(float)) != BT_OK) {
    free(gh); free(oh); return bt_fail(BT_ERR_BACKEND, "bt_relu_backward: copy failed");
  }
  for (int64_t i = 0; i < n; ++i) oh[i] = gh[i] * (oh[i] > 0.0f ? 1.0f : 0.0f);
  int64_t dims[8];
  for (int d = 0; d < bt_tensor_shape(g)->ndim; ++d) dims[d] = bt_tensor_shape(g)->dims[d];
  bt_tensor_t* result = bt_tensor_create(dims, (int)bt_tensor_shape(g)->ndim,
                                         BT_FLOAT32, bt_tensor_device(g), oh);
  free(gh); free(oh);
  if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_relu_backward: result OOM");
  *out = result;
  return BT_OK;
}

bt_status_t bt_sqrt(const bt_tensor_t* a, bt_tensor_t** out) {
  return unary(a, out, BT_OP_SQRT);
}

bt_status_t bt_add_relu(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t** out) {
  return binary(a, b, out, BT_OP_FUSED_ADD_RELU);
}
