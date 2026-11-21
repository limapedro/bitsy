#ifndef BITSY_BT_SHAPE_H
#define BITSY_BT_SHAPE_H

#include <stdint.h>
#include <bitsy/bt_export.h>
#include <bitsy/bt_error.h>

#ifdef __cplusplus
extern "C" {
#endif

/* PyTorch's practical rank ceiling. */
#define BT_MAX_DIMS 8

/* Fixed-capacity rank/dims; only [0,ndim) is significant. */
typedef struct bt_shape_t {
  int64_t ndim;
  int64_t dims[BT_MAX_DIMS];
} bt_shape_t;

BT_API bt_status_t bt_shape_make(const int64_t* dims, int64_t ndim, bt_shape_t* out);

/* 1 for a scalar, 0 when any dimension is zero. */
BT_API int64_t bt_shape_numel(const bt_shape_t* shape);

/* Write C-contiguous strides in elements. */
BT_API bt_status_t bt_shape_contiguous_strides(const bt_shape_t* shape, int64_t* strides);

/* Non-zero when rank and every dimension match. */
BT_API int bt_shape_equal(const bt_shape_t* a, const bt_shape_t* b);

/* Right-align, each dim equal or 1, result is the max. */
BT_API bt_status_t bt_broadcast_shapes(const bt_shape_t* a, const bt_shape_t* b, bt_shape_t* out);

/* Non-zero when `strides` describe a contiguous layout. */
BT_API int bt_shape_is_contiguous(const bt_shape_t* shape, const int64_t* strides);

#ifdef __cplusplus
}
#endif

#endif /* BITSY_BT_SHAPE_H */
