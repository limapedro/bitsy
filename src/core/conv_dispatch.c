#include "conv_dispatch_internal.h"

#include <string.h>

#include "dispatcher_internal.h"
#include "error_internal.h"

static bt_conv_kernel_fn g_kernels[BT_CONV_KIND_COUNT][BT_BACKEND_COUNT][BT_DTYPE_COUNT];

void bt_conv_dispatcher_reset(void) { memset(g_kernels, 0, sizeof(g_kernels)); }

bt_status_t bt_register_conv(bt_conv_kind_t kind, int backend, bt_dtype_t dtype,
                             bt_conv_kernel_fn fn) {
  if (kind < 0 || kind >= BT_CONV_KIND_COUNT || backend < 0 ||
      backend >= BT_BACKEND_COUNT || dtype < 0 || dtype >= BT_DTYPE_COUNT || !fn)
    return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_register_conv: invalid key");
  g_kernels[kind][backend][dtype] = fn;
  return BT_OK;
}

bt_status_t bt_dispatch_conv(bt_conv_kind_t kind, int backend, bt_dtype_t dtype,
                             const bt_tensor_t* a, const bt_tensor_t* b,
                             const bt_tensor_t* c, bt_conv_params_t params,
                             bt_conv_kind_t kernel_kind, bt_tensor_t* out) {
  if (kind < 0 || kind >= BT_CONV_KIND_COUNT || backend < 0 ||
      backend >= BT_BACKEND_COUNT || dtype < 0 || dtype >= BT_DTYPE_COUNT)
    return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_dispatch_conv: invalid key");
  bt_conv_kernel_fn fn = g_kernels[kind][backend][dtype];
  if (!fn) return bt_fail(BT_ERR_NOT_IMPLEMENTED, "no conv kernel for kind=%d", (int)kind);
  return fn(a, b, c, params, kernel_kind, out);
}
