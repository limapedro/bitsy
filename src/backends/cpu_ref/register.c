#include "cpu_ref.h"

#include "../../core/dispatcher_internal.h"

bt_status_t bt_cpu_ref_register(void) {
  return bt_register_binary(BT_OP_ADD, BT_BACKEND_CPU_REF, BT_FLOAT32, bt_cpu_ref_add_float32);
}
