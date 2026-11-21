#include "cpu_ref.h"

#include <stdint.h>

#include "../../core/error_internal.h"
#include "../../core/tensor_internal.h"

/* Naive i-j-k dot products; this is the oracle, not the fast path. */
bt_status_t bt_cpu_ref_matmul(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t* out) {
  if (!a || !b || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "cpu_ref matmul: NULL");
  int64_t M = a->shape.dims[0], K = a->shape.dims[1], N = b->shape.dims[1];
  const float* ad = (const float*)bt_storage_data(a->storage) + a->offset;
  const float* bd = (const float*)bt_storage_data(b->storage) + b->offset;
  float* cd = (float*)bt_storage_data(out->storage) + out->offset;
  int64_t sa0 = a->strides[0], sa1 = a->strides[1];
  int64_t sb0 = b->strides[0], sb1 = b->strides[1];
  int64_t sc0 = out->strides[0], sc1 = out->strides[1];
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float sum = 0.f;
      for (int64_t k = 0; k < K; ++k) sum += ad[i * sa0 + k * sa1] * bd[k * sb0 + j * sb1];
      cd[i * sc0 + j * sc1] = sum;
    }
  }
  bt_storage_bump_version(out->storage);
  return BT_OK;
}

/* b is unused: the dispatcher passes the input twice for unary kernels. */
bt_status_t bt_cpu_ref_transpose(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t* out) {
  (void)b;
  int64_t M = a->shape.dims[0], N = a->shape.dims[1];
  const float* ad = (const float*)bt_storage_data(a->storage) + a->offset;
  float* cd = (float*)bt_storage_data(out->storage) + out->offset;
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j)
      cd[j * out->strides[0] + i * out->strides[1]] = ad[i * a->strides[0] + j * a->strides[1]];
  bt_storage_bump_version(out->storage);
  return BT_OK;
}
