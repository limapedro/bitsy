#include "dispatcher_internal.h"

#include <string.h>

#include "error_internal.h"
#include "tensor_internal.h"

/* One kernel slot per (op, backend, dtype). A NULL slot means no support. */
static hg_binary_kernel_fn g_table[HG_OP_COUNT][HG_BACKEND_COUNT][HG_DTYPE_COUNT];

void hg_dispatcher_reset(void)
{
    memset(g_table, 0, sizeof(g_table));
}

hg_status_t hg_register_binary(hg_op_id_t op, hg_backend_t backend,
                               hg_dtype_t dtype, hg_binary_kernel_fn fn)
{
    if (op < 0 || op >= HG_OP_COUNT || backend < 0 || backend >= HG_BACKEND_COUNT ||
        dtype < 0 || dtype >= HG_DTYPE_COUNT || fn == NULL) {
        return hg_fail(HG_ERR_INVALID_ARGUMENT,
                       "hg_register_binary: invalid dispatch key");
    }
    g_table[op][backend][dtype] = fn;
    return HG_OK;
}

hg_backend_t hg_backend_for_device(hg_device_t device)
{
    return device.type == HG_DEVICE_TYPE_VULKAN ? HG_BACKEND_VULKAN
                                                : HG_BACKEND_CPU_REF;
}

hg_status_t hg_dispatch_binary(hg_op_id_t op, hg_backend_t backend,
                               hg_dtype_t dtype, const hg_tensor_t* a,
                               const hg_tensor_t* b, hg_tensor_t* out)
{
    if (op < 0 || op >= HG_OP_COUNT || backend < 0 || backend >= HG_BACKEND_COUNT ||
        dtype < 0 || dtype >= HG_DTYPE_COUNT) {
        return hg_fail(HG_ERR_INVALID_ARGUMENT, "hg_dispatch_binary: invalid key");
    }
    hg_binary_kernel_fn fn = g_table[op][backend][dtype];
    if (fn == NULL) {
        return hg_fail(HG_ERR_NOT_IMPLEMENTED,
                       "no kernel registered for op=%d backend=%d dtype=%d",
                       (int)op, (int)backend, (int)dtype);
    }
    return fn(a, b, out);
}
