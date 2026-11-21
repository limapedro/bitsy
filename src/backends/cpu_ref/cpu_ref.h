#ifndef BITSY_SRC_BACKENDS_CPU_REF_CPU_REF_H
#define BITSY_SRC_BACKENDS_CPU_REF_CPU_REF_H

#include "../../core/conv_dispatch_internal.h"

#include <bitsy/bt_error.h>
#include <bitsy/bt_tensor.h>

/* Install the naive kernels in the dispatcher table. */
bt_status_t bt_cpu_ref_register(void);

/* Naive broadcasting elementwise kernel shared by all registered ops. */
bt_status_t bt_cpu_ref_elementwise(const bt_tensor_t* a, const bt_tensor_t* b,
                                   bt_tensor_t* out, int op);
bt_status_t bt_cpu_ref_add_relu(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t* out);

bt_status_t bt_cpu_ref_matmul(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t* out);
bt_status_t bt_cpu_ref_transpose(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t* out);
bt_status_t bt_cpu_ref_conv(const bt_tensor_t* a, const bt_tensor_t* b,
                            const bt_tensor_t* c, bt_conv_params_t params,
                            bt_conv_kind_t kind, bt_tensor_t* out);

#endif /* BITSY_SRC_BACKENDS_CPU_REF_CPU_REF_H */
