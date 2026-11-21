#include <hagane/hg_shape.h>

#include <stdio.h>

#include "error_internal.h"

hg_status_t hg_shape_make(const int64_t* dims, int64_t ndim, hg_shape_t* out)
{
    if (out == NULL || ndim < 0 || ndim > HG_MAX_DIMS ||
        (ndim > 0 && dims == NULL)) {
        return hg_fail(HG_ERR_INVALID_ARGUMENT,
                       "hg_shape_make: ndim must be in [0, %d]", HG_MAX_DIMS);
    }
    out->ndim = ndim;
    for (int64_t i = 0; i < ndim; ++i) {
        if (dims[i] < 0) {
            return hg_fail(HG_ERR_INVALID_ARGUMENT,
                           "hg_shape_make: negative dimension at index %lld",
                           (long long)i);
        }
        out->dims[i] = dims[i];
    }
    for (int64_t i = ndim; i < HG_MAX_DIMS; ++i) {
        out->dims[i] = 0;
    }
    return HG_OK;
}

int64_t hg_shape_numel(const hg_shape_t* shape)
{
    int64_t n = 1;
    if (shape == NULL) {
        return 0;
    }
    for (int64_t i = 0; i < shape->ndim; ++i) {
        n *= shape->dims[i];
    }
    return n;
}

hg_status_t hg_shape_contiguous_strides(const hg_shape_t* shape,
                                        int64_t* strides)
{
    if (shape == NULL || strides == NULL) {
        return hg_fail(HG_ERR_INVALID_ARGUMENT,
                       "hg_shape_contiguous_strides: NULL argument");
    }
    int64_t stride = 1;
    for (int64_t i = shape->ndim - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape->dims[i];
    }
    for (int64_t i = shape->ndim; i < HG_MAX_DIMS; ++i) {
        strides[i] = 0;
    }
    return HG_OK;
}

int hg_shape_equal(const hg_shape_t* a, const hg_shape_t* b)
{
    if (a == NULL || b == NULL || a->ndim != b->ndim) {
        return 0;
    }
    for (int64_t i = 0; i < a->ndim; ++i) {
        if (a->dims[i] != b->dims[i]) {
            return 0;
        }
    }
    return 1;
}

hg_status_t hg_broadcast_shapes(const hg_shape_t* a, const hg_shape_t* b,
                                hg_shape_t* out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return hg_fail(HG_ERR_INVALID_ARGUMENT,
                       "hg_broadcast_shapes: NULL argument");
    }
    if (a->ndim > HG_MAX_DIMS || b->ndim > HG_MAX_DIMS) {
        return hg_fail(HG_ERR_INVALID_ARGUMENT,
                       "hg_broadcast_shapes: rank exceeds %d", HG_MAX_DIMS);
    }

    int64_t ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    hg_shape_t result;
    result.ndim = ndim;

    for (int64_t i = 0; i < ndim; ++i) {
        int64_t ad = 1;
        int64_t bd = 1;
        int64_t ia = i - (ndim - a->ndim);
        int64_t ib = i - (ndim - b->ndim);
        if (ia >= 0) {
            ad = a->dims[ia];
        }
        if (ib >= 0) {
            bd = b->dims[ib];
        }

        int64_t d;
        if (ad == bd) {
            d = ad;
        } else if (ad == 1) {
            d = bd;
        } else if (bd == 1) {
            d = ad;
        } else {
            return hg_fail(HG_ERR_SHAPE_MISMATCH,
                           "hg_broadcast_shapes: sizes %lld and %lld at dim %lld "
                           "are not broadcastable",
                           (long long)ad, (long long)bd, (long long)i);
        }
        result.dims[i] = d;
    }
    for (int64_t i = ndim; i < HG_MAX_DIMS; ++i) {
        result.dims[i] = 0;
    }

    *out = result;
    return HG_OK;
}

int hg_shape_is_contiguous(const hg_shape_t* shape, const int64_t* strides)
{
    if (shape == NULL || strides == NULL) {
        return 0;
    }
    if (shape->ndim <= 1) {
        return 1;
    }
    for (int64_t i = shape->ndim - 1; i >= 0; --i) {
        if (shape->dims[i] != 1 && strides[i] != 0) {
            int64_t expected = (i + 1 < shape->ndim) ? strides[i + 1] * shape->dims[i + 1] : 1;
            if (strides[i] != expected) {
                return 0;
            }
        }
    }
    return 1;
}
