#ifndef HAGANE_SRC_CORE_DISPATCHER_INTERNAL_H
#define HAGANE_SRC_CORE_DISPATCHER_INTERNAL_H

#include <hagane/hg_dtype.h>
#include <hagane/hg_device.h>
#include <hagane/hg_error.h>
#include <hagane/hg_tensor.h>

typedef enum hg_op_id_t {
    HG_OP_ADD = 0,
    HG_OP_COUNT
} hg_op_id_t;

typedef enum hg_backend_t {
    HG_BACKEND_CPU_REF = 0,
    HG_BACKEND_VULKAN = 1,
    HG_BACKEND_COUNT
} hg_backend_t;

/** Kernel signature for a binary operation producing one output. */
typedef hg_status_t (*hg_binary_kernel_fn)(const hg_tensor_t* a,
                                            const hg_tensor_t* b,
                                            hg_tensor_t* out);

/** Install a kernel; later registrations for the same key replace earlier ones. */
hg_status_t hg_register_binary(hg_op_id_t op, hg_backend_t backend,
                               hg_dtype_t dtype, hg_binary_kernel_fn fn);

/** Look up and run a kernel. Returns HG_ERR_NOT_IMPLEMENTED when absent. */
hg_status_t hg_dispatch_binary(hg_op_id_t op, hg_backend_t backend,
                               hg_dtype_t dtype, const hg_tensor_t* a,
                               const hg_tensor_t* b, hg_tensor_t* out);

/** Map a device to the backend that implements its kernels. */
hg_backend_t hg_backend_for_device(hg_device_t device);

/** Reset the dispatcher table (used by hg_shutdown and tests). */
void hg_dispatcher_reset(void);

#endif /* HAGANE_SRC_CORE_DISPATCHER_INTERNAL_H */
