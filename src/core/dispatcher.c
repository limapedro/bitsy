#include "dispatcher_internal.h"

#include <string.h>

#include "error_internal.h"

/* One kernel per (op, backend, dtype). A NULL slot is an honest "no kernel". */
static bt_binary_kernel_fn g_table[BT_OP_COUNT][BT_BACKEND_COUNT][BT_DTYPE_COUNT];

void bt_dispatcher_reset(void) { memset(g_table, 0, sizeof(g_table)); }

bt_status_t bt_register_binary(bt_op_id_t op, bt_backend_t backend,
                               bt_dtype_t dtype, bt_binary_kernel_fn fn) {
  if (op < 0 || op >= BT_OP_COUNT || backend < 0 || backend >= BT_BACKEND_COUNT ||
      dtype < 0 || dtype >= BT_DTYPE_COUNT || !fn)
    return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_register_binary: invalid key");
  g_table[op][backend][dtype] = fn;
  return BT_OK;
}

bt_backend_t bt_backend_for_device(bt_device_t device) {
  return device.type == BT_DEVICE_TYPE_VULKAN ? BT_BACKEND_VULKAN : BT_BACKEND_CPU_REF;
}

bt_status_t bt_dispatch_binary(bt_op_id_t op, bt_backend_t backend, bt_dtype_t dtype,
                               const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t* out) {
  if (op < 0 || op >= BT_OP_COUNT || backend < 0 || backend >= BT_BACKEND_COUNT ||
      dtype < 0 || dtype >= BT_DTYPE_COUNT)
    return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_dispatch_binary: invalid key");
  bt_binary_kernel_fn fn = g_table[op][backend][dtype];
  if (!fn) return bt_fail(BT_ERR_NOT_IMPLEMENTED,
                          "no kernel for op=%d backend=%d dtype=%d", (int)op, (int)backend, (int)dtype);
  return fn(a, b, out);
}
