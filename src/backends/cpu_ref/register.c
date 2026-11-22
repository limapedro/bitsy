#include "cpu_ref.h"

#include "../../core/dispatcher_internal.h"

bt_status_t bt_cpu_ref_register_conv(void);

/* Thin adapters: the dispatcher's kernel signature has no op argument. */
#define KERNEL(name, op) \
  static bt_status_t name(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t* out) { \
    return bt_cpu_ref_elementwise(a, b, out, op); \
  }

KERNEL(add, BT_OP_ADD)
KERNEL(sub, BT_OP_SUB)
KERNEL(mul, BT_OP_MUL)
KERNEL(op_div, BT_OP_DIV)
KERNEL(relu, BT_OP_RELU)
KERNEL(sigmoid, BT_OP_SIGMOID)
KERNEL(op_tanh, BT_OP_TANH)
KERNEL(op_exp, BT_OP_EXP)
KERNEL(op_log, BT_OP_LOG)
KERNEL(op_sqrt, BT_OP_SQRT)

#define REG(op, fn) bt_register_binary(op, BT_BACKEND_CPU_REF, BT_FLOAT32, fn)

bt_status_t bt_cpu_ref_register(void) {
  bt_status_t status = BT_OK;
  if ((status = REG(BT_OP_ADD, add))) return status;
  if ((status = REG(BT_OP_SUB, sub))) return status;
  if ((status = REG(BT_OP_MUL, mul))) return status;
  if ((status = REG(BT_OP_DIV, op_div))) return status;
  if ((status = REG(BT_OP_RELU, relu))) return status;
  if ((status = REG(BT_OP_SIGMOID, sigmoid))) return status;
  if ((status = REG(BT_OP_TANH, op_tanh))) return status;
  if ((status = REG(BT_OP_EXP, op_exp))) return status;
  if ((status = REG(BT_OP_LOG, op_log))) return status;
  if ((status = REG(BT_OP_SQRT, op_sqrt))) return status;
  if ((status = REG(BT_OP_FUSED_ADD_RELU, bt_cpu_ref_add_relu))) return status;
  if ((status = REG(BT_OP_MATMUL, bt_cpu_ref_matmul))) return status;
  if ((status = REG(BT_OP_TRANSPOSE, bt_cpu_ref_transpose))) return status;
  for (int k = 0; k < BT_CONV_KIND_COUNT; ++k) {
    status = bt_register_conv((bt_conv_kind_t)k, BT_BACKEND_CPU_REF,
                              BT_FLOAT32, bt_cpu_ref_conv);
    if (status != BT_OK) return status;
  }
  return BT_OK;
}
