#include "cpu_ref.h"

#include <stdint.h>

#include "../../core/error_internal.h"
#include "../../core/tensor_internal.h"

/* An input dim added by broadcasting, or of size 1, must not move the pointer. */
static int64_t stride_at(const bt_tensor_t* t, int64_t out_dim, int64_t out_ndim) {
  int64_t in_dim = out_dim - (out_ndim - t->shape.ndim);
  if (in_dim < 0 || in_dim >= t->shape.ndim || t->shape.dims[in_dim] == 1) return 0;
  return t->strides[in_dim];
}

/* Naive elementwise add. It is the oracle, not the fast path. */
bt_status_t bt_cpu_ref_add_float32(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t* out) {
  if (!a || !b || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_cpu_ref_add_float32: NULL");
  if (a->dtype != BT_FLOAT32 || b->dtype != BT_FLOAT32 || out->dtype != BT_FLOAT32)
    return bt_fail(BT_ERR_UNSUPPORTED_DTYPE, "bt_cpu_ref_add_float32: float32 only");

  const bt_shape_t* shape = &out->shape;
  int64_t ndim = shape->ndim, total = bt_shape_numel(shape);
  if (!total) return BT_OK; // empty tensors do no work

  const float* adata = (const float*)bt_storage_data(a->storage) + a->offset;
  const float* bdata = (const float*)bt_storage_data(b->storage) + b->offset;
  float* odata = (float*)bt_storage_data(out->storage) + out->offset;

  int64_t sa[BT_MAX_DIMS], sb[BT_MAX_DIMS], coord[BT_MAX_DIMS];
  for (int64_t d = 0; d < ndim; ++d) {
    sa[d] = stride_at(a, d, ndim);
    sb[d] = stride_at(b, d, ndim);
    coord[d] = 0;
  }

  for (int64_t n = 0; n < total; ++n) {
    int64_t ia = 0, ib = 0, io = 0;
    for (int64_t d = 0; d < ndim; ++d) {
      ia += coord[d] * sa[d];
      ib += coord[d] * sb[d];
      io += coord[d] * out->strides[d];
    }
    odata[io] = adata[ia] + bdata[ib]; // correctly-rounded IEEE add, same as PyTorch
    for (int64_t d = ndim - 1; d >= 0; --d) { // odometer, least significant dim last
      if (++coord[d] < shape->dims[d]) break;
      coord[d] = 0;
    }
  }
  bt_storage_bump_version(out->storage);
  return BT_OK;
}
