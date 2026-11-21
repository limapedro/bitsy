#include <hagane/hg_ops.h>

#include "../core/dispatcher_internal.h"
#include "../core/error_internal.h"

hg_status_t hg_add(const hg_tensor_t* a, const hg_tensor_t* b,
                   hg_tensor_t** out)
{
    hg_clear_error();

    if (a == NULL || b == NULL || out == NULL) {
        return hg_fail(HG_ERR_INVALID_ARGUMENT, "hg_add: NULL argument");
    }

    hg_dtype_t dtype = hg_tensor_dtype(a);
    if (dtype != HG_FLOAT32 || hg_tensor_dtype(b) != HG_FLOAT32) {
        return hg_fail(HG_ERR_UNSUPPORTED_DTYPE,
                       "hg_add: only float32 is implemented in M0");
    }

    hg_device_t device = hg_tensor_device(a);
    if (!hg_device_equal(device, hg_tensor_device(b))) {
        return hg_fail(HG_ERR_UNSUPPORTED_DEVICE,
                       "hg_add: operands must be on the same device");
    }
    if (device.type == HG_DEVICE_TYPE_VULKAN) {
        return hg_fail(HG_ERR_NOT_IMPLEMENTED,
                       "hg_add: the vulkan backend arrives in M2");
    }

    hg_shape_t shape;
    hg_status_t status = hg_broadcast_shapes(hg_tensor_shape(a),
                                             hg_tensor_shape(b), &shape);
    if (status != HG_OK) {
        return status;
    }

    hg_tensor_t* result = hg_tensor_empty(shape.dims, shape.ndim, dtype, device);
    if (result == NULL) {
        return hg_fail(HG_ERR_OUT_OF_MEMORY, "hg_add: could not allocate output");
    }

    status = hg_dispatch_binary(HG_OP_ADD, hg_backend_for_device(device),
                                dtype, a, b, result);
    if (status != HG_OK) {
        hg_tensor_release(result);
        return status;
    }

    *out = result;
    return HG_OK;
}
