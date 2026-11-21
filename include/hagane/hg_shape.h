#ifndef HAGANE_HG_SHAPE_H
#define HAGANE_HG_SHAPE_H

#include <stdint.h>
#include <hagane/hg_export.h>
#include <hagane/hg_error.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum tensor rank, matching PyTorch's practical limit. */
#define HG_MAX_DIMS 8

/** A fixed-capacity rank/dimension vector. `dims[0..ndim)` are significant. */
typedef struct hg_shape_t {
    int64_t ndim;
    int64_t dims[HG_MAX_DIMS];
} hg_shape_t;

/** Build a shape from `ndim` dimensions. Scalars have ndim == 0. */
HG_API hg_status_t hg_shape_make(const int64_t* dims, int64_t ndim, hg_shape_t* out);

/** Number of elements; 1 for a scalar, 0 when any dimension is zero. */
HG_API int64_t hg_shape_numel(const hg_shape_t* shape);

/** Write C-contiguous strides for `shape` into `strides[HG_MAX_DIMS]`. */
HG_API hg_status_t hg_shape_contiguous_strides(const hg_shape_t* shape, int64_t* strides);

/** Non-zero when two shapes have the same rank and dimensions. */
HG_API int hg_shape_equal(const hg_shape_t* a, const hg_shape_t* b);

/** PyTorch broadcasting: align right, each dim equal or 1, result is max. */
HG_API hg_status_t hg_broadcast_shapes(const hg_shape_t* a, const hg_shape_t* b, hg_shape_t* out);

/** Non-zero when `strides` describe a contiguous layout for `shape`. */
HG_API int hg_shape_is_contiguous(const hg_shape_t* shape, const int64_t* strides);

#ifdef __cplusplus
}
#endif

#endif /* HAGANE_HG_SHAPE_H */
