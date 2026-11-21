#ifndef BITSY_SRC_BACKENDS_CPU_REF_CPU_REF_H
#define BITSY_SRC_BACKENDS_CPU_REF_CPU_REF_H

#include <bitsy/bt_error.h>
#include <bitsy/bt_tensor.h>

/* Install the naive kernels in the dispatcher table. */
bt_status_t bt_cpu_ref_register(void);

bt_status_t bt_cpu_ref_add_float32(const bt_tensor_t* a, const bt_tensor_t* b,
                                   bt_tensor_t* out);

#endif /* BITSY_SRC_BACKENDS_CPU_REF_CPU_REF_H */
