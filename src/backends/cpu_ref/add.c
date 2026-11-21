#include "cpu_ref.h"

#include <stdint.h>

#include "../../core/error_internal.h"
#include "../../core/tensor_internal.h"

/*
 * Effective stride of an input along output dimension `out_dim`.
 * A dimension that was added by broadcasting, or that has size 1, never
 * advances the input pointer.
 */
static int64_t broadcast_stride(const hg_tensor_t* input, int64_t out_dim,
                                int64_t out_ndim)
{
    int64_t in_ndim = input->shape.ndim;
    int64_t in_dim = out_dim - (out_ndim - in_ndim);
    if (in_dim < 0 || in_dim >= in_ndim || input->shape.dims[in_dim] == 1) {
        return 0;
    }
    return input->strides[in_dim];
}

hg_status_t hg_cpu_ref_add_float32(const hg_tensor_t* a, const hg_tensor_t* b,
                                   hg_tensor_t* out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return hg_fail(HG_ERR_INVALID_ARGUMENT,
                       "hg_cpu_ref_add_float32: NULL argument");
    }
    if (a->dtype != HG_FLOAT32 || b->dtype != HG_FLOAT32 ||
        out->dtype != HG_FLOAT32) {
        return hg_fail(HG_ERR_UNSUPPORTED_DTYPE,
                       "hg_cpu_ref_add_float32: only float32 is implemented");
    }

    const hg_shape_t* shape = &out->shape;
    int64_t ndim = shape->ndim;
    int64_t total = hg_shape_numel(shape);
    if (total == 0) {
        return HG_OK;
    }

    const float* adata = (const float*)hg_storage_data(a->storage) + a->offset;
    const float* bdata = (const float*)hg_storage_data(b->storage) + b->offset;
    float* odata = (float*)hg_storage_data(out->storage) + out->offset;

    int64_t stride_a[HG_MAX_DIMS];
    int64_t stride_b[HG_MAX_DIMS];
    for (int64_t d = 0; d < ndim; ++d) {
        stride_a[d] = broadcast_stride(a, d, ndim);
        stride_b[d] = broadcast_stride(b, d, ndim);
    }

    int64_t coord[HG_MAX_DIMS];
    for (int64_t d = 0; d < ndim; ++d) {
        coord[d] = 0;
    }

    for (int64_t n = 0; n < total; ++n) {
        int64_t ia = 0;
        int64_t ib = 0;
        int64_t io = 0;
        for (int64_t d = 0; d < ndim; ++d) {
            ia += coord[d] * stride_a[d];
            ib += coord[d] * stride_b[d];
            io += coord[d] * out->strides[d];
        }
        /* Plain IEEE-754 addition: bit-identical to PyTorch CPU for fp32. */
        odata[io] = adata[ia] + bdata[ib];

        for (int64_t d = ndim - 1; d >= 0; --d) {
            if (++coord[d] < shape->dims[d]) {
                break;
            }
            coord[d] = 0;
        }
    }

    hg_storage_bump_version(out->storage);
    return HG_OK;
}
