#ifndef BITSY_SRC_CORE_CONV_DISPATCH_INTERNAL_H
#define BITSY_SRC_CORE_CONV_DISPATCH_INTERNAL_H

#include <bitsy/bt_conv.h>
#include <bitsy/bt_dtype.h>
#include <bitsy/bt_error.h>

typedef enum bt_conv_kind_t {
  BT_CONV_FORWARD = 0, BT_CONV_DX, BT_CONV_DW, BT_CONV_DB, BT_CONV_KIND_COUNT
} bt_conv_kind_t;

/* One generic kernel signature: unused arguments are NULL for each kind. */
typedef bt_status_t (*bt_conv_kernel_fn)(const bt_tensor_t* a,
                                          const bt_tensor_t* b,
                                          const bt_tensor_t* c,
                                          bt_conv_params_t params,
                                          bt_conv_kind_t kind,
                                          bt_tensor_t* out);

bt_status_t bt_register_conv(bt_conv_kind_t kind, int backend, bt_dtype_t dtype,
                             bt_conv_kernel_fn fn);
bt_status_t bt_dispatch_conv(bt_conv_kind_t kind, int backend, bt_dtype_t dtype,
                             const bt_tensor_t* a, const bt_tensor_t* b,
                             const bt_tensor_t* c, bt_conv_params_t params,
                             bt_conv_kind_t kernel_kind, bt_tensor_t* out);
void bt_conv_dispatcher_reset(void);

#endif /* BITSY_SRC_CORE_CONV_DISPATCH_INTERNAL_H */
