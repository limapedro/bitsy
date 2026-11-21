#include <bitsy/bt_conv.h>

#include "../core/conv_dispatch_internal.h"
#include "../core/dispatcher_internal.h"
#include "../core/error_internal.h"

static bt_status_t dispatch_conv(bt_conv_kind_t kind, const bt_tensor_t* a,
                                 const bt_tensor_t* b, const bt_tensor_t* c,
                                 bt_conv_params_t params, const int64_t* dims,
                                 int64_t ndim, bt_tensor_t** out) {
  bt_device_t device = bt_tensor_device(a);
  bt_tensor_t* result = bt_tensor_empty(dims, ndim, BT_FLOAT32, device);
  if (!result) return bt_fail(BT_ERR_OUT_OF_MEMORY, "conv: allocation failed");
  bt_status_t status = bt_dispatch_conv(kind, bt_backend_for_device(device),
                                        BT_FLOAT32, a, b, c, params, kind, result);
  if (status != BT_OK) { bt_tensor_release(result); return status; }
  *out = result;
  return BT_OK;
}

bt_status_t bt_conv2d(const bt_tensor_t* x, const bt_tensor_t* w,
                      const bt_tensor_t* b, bt_conv_params_t params,
                      bt_tensor_t** out) {
  if (!x || !w || !b || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_conv2d: NULL");
  if (bt_tensor_shape(x)->ndim != 4 || bt_tensor_shape(w)->ndim != 4)
    return bt_fail(BT_ERR_SHAPE_MISMATCH, "bt_conv2d: expects 4D x and w");
  int64_t N = bt_tensor_shape(x)->dims[0], C = bt_tensor_shape(x)->dims[1];
  int64_t H = bt_tensor_shape(x)->dims[2], Wd = bt_tensor_shape(x)->dims[3];
  int64_t O = bt_tensor_shape(w)->dims[0], kH = bt_tensor_shape(w)->dims[2],
          kW = bt_tensor_shape(w)->dims[3];
  if (bt_tensor_shape(w)->dims[1] != C)
    return bt_fail(BT_ERR_SHAPE_MISMATCH, "bt_conv2d: channel mismatch");
  int64_t OH = (H + 2 * params.pad - kH) / params.stride + 1;
  int64_t OW = (Wd + 2 * params.pad - kW) / params.stride + 1;
  int64_t dims[4] = { N, O, OH, OW };
  return dispatch_conv(BT_CONV_FORWARD, x, w, b, params, dims, 4, out);
}

bt_status_t bt_conv2d_backward_x(const bt_tensor_t* dy, const bt_tensor_t* w,
                                 bt_shape_t x_shape, bt_conv_params_t params,
                                 bt_tensor_t** out) {
  if (!dy || !w || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_conv2d_backward_x: NULL");
  int64_t dims[4] = { x_shape.dims[0], x_shape.dims[1], x_shape.dims[2], x_shape.dims[3] };
  return dispatch_conv(BT_CONV_DX, dy, w, NULL, params, dims, 4, out);
}

bt_status_t bt_conv2d_backward_w(const bt_tensor_t* x, const bt_tensor_t* dy,
                                 bt_shape_t w_shape, bt_conv_params_t params,
                                 bt_tensor_t** out) {
  if (!x || !dy || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_conv2d_backward_w: NULL");
  int64_t dims[4] = { w_shape.dims[0], w_shape.dims[1], w_shape.dims[2], w_shape.dims[3] };
  return dispatch_conv(BT_CONV_DW, x, dy, NULL, params, dims, 4, out);
}

bt_status_t bt_conv2d_backward_b(const bt_tensor_t* dy, int64_t out_channels,
                                 bt_tensor_t** out) {
  if (!dy || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_conv2d_backward_b: NULL");
  return dispatch_conv(BT_CONV_DB, dy, NULL, NULL, (bt_conv_params_t){ 0, 1 },
                       &out_channels, 1, out);
}
