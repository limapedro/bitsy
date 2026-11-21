#include <bitsy/bt_shape.h>

#include "error_internal.h"

/* *** shape math *** */

bt_status_t bt_shape_make(const int64_t* dims, int64_t ndim, bt_shape_t* out) {
  if (!out || ndim < 0 || ndim > BT_MAX_DIMS || (ndim > 0 && !dims))
    return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_shape_make: ndim must be in [0, %d]", BT_MAX_DIMS);
  out->ndim = ndim;
  for (int64_t i = 0; i < ndim; ++i) {
    if (dims[i] < 0)
      return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_shape_make: negative dim at %lld", (long long)i);
    out->dims[i] = dims[i];
  }
  for (int64_t i = ndim; i < BT_MAX_DIMS; ++i) out->dims[i] = 0;  // no stale dims
  return BT_OK;
}

int64_t bt_shape_numel(const bt_shape_t* shape) {
  if (!shape) return 0;
  int64_t n = 1;
  for (int64_t i = 0; i < shape->ndim; ++i) n *= shape->dims[i];
  return n;
}

bt_status_t bt_shape_contiguous_strides(const bt_shape_t* shape, int64_t* strides) {
  if (!shape || !strides) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_shape_contiguous_strides: NULL");
  int64_t stride = 1;
  for (int64_t i = shape->ndim - 1; i >= 0; --i) { strides[i] = stride; stride *= shape->dims[i]; }
  for (int64_t i = shape->ndim; i < BT_MAX_DIMS; ++i) strides[i] = 0;
  return BT_OK;
}

int bt_shape_equal(const bt_shape_t* a, const bt_shape_t* b) {
  if (!a || !b || a->ndim != b->ndim) return 0;
  for (int64_t i = 0; i < a->ndim; ++i) if (a->dims[i] != b->dims[i]) return 0;
  return 1;
}

bt_status_t bt_broadcast_shapes(const bt_shape_t* a, const bt_shape_t* b, bt_shape_t* out) {
  if (!a || !b || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_broadcast_shapes: NULL");
  if (a->ndim > BT_MAX_DIMS || b->ndim > BT_MAX_DIMS)
    return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_broadcast_shapes: rank exceeds %d", BT_MAX_DIMS);

  int64_t ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  bt_shape_t r = { .ndim = ndim };
  for (int64_t i = 0; i < ndim; ++i) {
    // right-align: a missing leading dimension is a size-1 dimension
    int64_t ia = i - (ndim - a->ndim), ib = i - (ndim - b->ndim);
    int64_t ad = ia >= 0 ? a->dims[ia] : 1, bd = ib >= 0 ? b->dims[ib] : 1;
    int64_t d;
    if (ad == bd) d = ad;
    else if (ad == 1) d = bd;
    else if (bd == 1) d = ad;
    else return bt_fail(BT_ERR_SHAPE_MISMATCH,
                        "bt_broadcast_shapes: %lld and %lld at dim %lld are not broadcastable",
                        (long long)ad, (long long)bd, (long long)i);
    r.dims[i] = d;
  }
  for (int64_t i = ndim; i < BT_MAX_DIMS; ++i) r.dims[i] = 0;
  *out = r;
  return BT_OK;
}

int bt_shape_is_contiguous(const bt_shape_t* shape, const int64_t* strides) {
  if (!shape || !strides || shape->ndim <= 1) return shape && strides ? 1 : 0;
  for (int64_t i = shape->ndim - 1; i >= 0; --i) {
    // size-1 dims can carry any stride in PyTorch and are still contiguous
    if (shape->dims[i] == 1 || !strides[i]) continue;
    int64_t expected = i + 1 < shape->ndim ? strides[i + 1] * shape->dims[i + 1] : 1;
    if (strides[i] != expected) return 0;
  }
  return 1;
}
