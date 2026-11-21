#ifndef BITSY_BT_TENSOR_H
#define BITSY_BT_TENSOR_H

#include <stddef.h>
#include <stdint.h>
#include <bitsy/bt_export.h>
#include <bitsy/bt_dtype.h>
#include <bitsy/bt_device.h>
#include <bitsy/bt_error.h>
#include <bitsy/bt_shape.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A view: (storage, offset, sizes, strides, dtype, device). Opaque. */
typedef struct bt_tensor bt_tensor_t;

/* Copy host_data (numel * dtype_size bytes) when non-NULL; NULL + error on failure. */
BT_API bt_tensor_t* bt_tensor_create(const int64_t* dims, int64_t ndim,
                                     bt_dtype_t dtype, bt_device_t device,
                                     const void* host_data);

BT_API bt_tensor_t* bt_tensor_empty(const int64_t* dims, int64_t ndim,
                                    bt_dtype_t dtype, bt_device_t device);
BT_API bt_tensor_t* bt_tensor_zeros(const int64_t* dims, int64_t ndim,
                                    bt_dtype_t dtype, bt_device_t device);

/* Increase the view refcount. */
BT_API bt_tensor_t* bt_tensor_retain(bt_tensor_t* tensor);

/* Drop a view ref; NULL is a no-op. */
BT_API void bt_tensor_release(bt_tensor_t* tensor);

/* All returned views are valid for the lifetime of `tensor`. */

/* Shape of the view. */
BT_API const bt_shape_t* bt_tensor_shape(const bt_tensor_t* tensor);

/* Strides in elements. */
BT_API const int64_t* bt_tensor_strides(const bt_tensor_t* tensor);

/* Element type. */
BT_API bt_dtype_t bt_tensor_dtype(const bt_tensor_t* tensor);

/* Device holding the storage. */
BT_API bt_device_t bt_tensor_device(const bt_tensor_t* tensor);

/* Product of the dimensions. */
BT_API int64_t bt_tensor_numel(const bt_tensor_t* tensor);

/* Non-zero when the view is C-contiguous. */
BT_API int bt_tensor_is_contiguous(const bt_tensor_t* tensor);

/* Version of the underlying storage. */
BT_API uint64_t bt_tensor_version(const bt_tensor_t* tensor);

/* Copy numel * dtype_size bytes out. Contiguous views only for M0. */
BT_API bt_status_t bt_tensor_copy_to_host(const bt_tensor_t* tensor, void* dst,
                                          size_t dst_bytes);

/* Contiguous reshape: shares storage, no copy, no Vulkan dispatch. */
BT_API bt_status_t bt_tensor_view(const bt_tensor_t* tensor, const int64_t* dims,
                                  int64_t ndim, bt_tensor_t** out);

/* exp_017: 2D transpose as a strided view: shares storage, no copy, no Vulkan
 * dispatch. GEMM consumes tensor strides directly, so backward can use this
 * instead of materialising a transposed buffer. */
BT_API bt_status_t bt_tensor_transpose_view(const bt_tensor_t* tensor,
                                            bt_tensor_t** out);

#ifdef __cplusplus
}
#endif

#endif /* BITSY_BT_TENSOR_H */
