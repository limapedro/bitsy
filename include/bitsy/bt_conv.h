#ifndef BITSY_BT_CONV_H
#define BITSY_BT_CONV_H

#include <bitsy/bt_export.h>
#include <bitsy/bt_error.h>
#include <bitsy/bt_shape.h>
#include <bitsy/bt_tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/* NCHW 2D convolution with shared padding/stride. Shapes:
 * x (N,C,H,W), w (O,C,kH,kW), b (O) -> out (N,O,OH,OW). */
typedef struct bt_conv_params_t {
  int pad;
  int stride;
} bt_conv_params_t;

BT_API bt_status_t bt_conv2d(const bt_tensor_t* x, const bt_tensor_t* w,
                             const bt_tensor_t* b, bt_conv_params_t params,
                             bt_tensor_t** out);

/* Backward kernels. `x_shape` gives the original input shape for dX. */
BT_API bt_status_t bt_conv2d_backward_x(const bt_tensor_t* dy, const bt_tensor_t* w,
                                        bt_shape_t x_shape, bt_conv_params_t params,
                                        bt_tensor_t** out);
BT_API bt_status_t bt_conv2d_backward_w(const bt_tensor_t* x, const bt_tensor_t* dy,
                                        bt_shape_t w_shape, bt_conv_params_t params,
                                        bt_tensor_t** out);
BT_API bt_status_t bt_conv2d_backward_b(const bt_tensor_t* dy, int64_t out_channels,
                                        bt_tensor_t** out);

#ifdef __cplusplus
}
#endif

#endif /* BITSY_BT_CONV_H */
