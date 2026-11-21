#include "cpu_ref.h"

#include "../../core/dispatcher_internal.h"

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
  return REG(BT_OP_EXP, op_exp);
}
