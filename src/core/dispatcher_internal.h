#ifndef BITSY_SRC_CORE_DISPATCHER_INTERNAL_H
#define BITSY_SRC_CORE_DISPATCHER_INTERNAL_H

#include <bitsy/bt_dtype.h>
#include <bitsy/bt_device.h>
#include <bitsy/bt_error.h>
#include <bitsy/bt_tensor.h>

typedef enum bt_op_id_t {
  BT_OP_ADD = 0, BT_OP_SUB, BT_OP_MUL, BT_OP_DIV,
  BT_OP_RELU, BT_OP_SIGMOID, BT_OP_TANH, BT_OP_EXP,
  BT_OP_MATMUL, BT_OP_TRANSPOSE, BT_OP_LOG,
  BT_OP_COUNT
} bt_op_id_t;

typedef enum bt_backend_t {
  BT_BACKEND_CPU_REF = 0,
  BT_BACKEND_VULKAN = 1,
  BT_BACKEND_COUNT
} bt_backend_t;

typedef bt_status_t (*bt_binary_kernel_fn)(const bt_tensor_t* a,
                                            const bt_tensor_t* b,
                                            bt_tensor_t* out);

/* NULL slots are simply unsupported; one table to audit kernel coverage. */
bt_status_t bt_register_binary(bt_op_id_t op, bt_backend_t backend,
                               bt_dtype_t dtype, bt_binary_kernel_fn fn);
bt_status_t bt_dispatch_binary(bt_op_id_t op, bt_backend_t backend,
                               bt_dtype_t dtype, const bt_tensor_t* a,
                               const bt_tensor_t* b, bt_tensor_t* out);
bt_backend_t bt_backend_for_device(bt_device_t device);
void bt_dispatcher_reset(void);

#endif /* BITSY_SRC_CORE_DISPATCHER_INTERNAL_H */
