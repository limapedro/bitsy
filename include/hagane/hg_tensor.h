#ifndef HAGANE_HG_TENSOR_H
#define HAGANE_HG_TENSOR_H

#include <stddef.h>
#include <stdint.h>
#include <hagane/hg_export.h>
#include <hagane/hg_dtype.h>
#include <hagane/hg_device.h>
#include <hagane/hg_error.h>
#include <hagane/hg_shape.h>

#ifdef __cplusplus
extern "C" {
#endif

/** A tensor view: (storage, offset, sizes, strides, dtype, device). Opaque. */
typedef struct hg_tensor hg_tensor_t;

/**
 * Create a contiguous tensor. When `host_data` is non-NULL, exactly
 * numel * dtype_size bytes are copied into new storage.
 * Returns NULL and sets the thread-local error on failure.
 */
HG_API hg_tensor_t* hg_tensor_create(const int64_t* dims, int64_t ndim,
                                     hg_dtype_t dtype, hg_device_t device,
                                     const void* host_data);

/** Uninitialized contiguous tensor. */
HG_API hg_tensor_t* hg_tensor_empty(const int64_t* dims, int64_t ndim,
                                    hg_dtype_t dtype, hg_device_t device);

/** Zero-filled contiguous tensor. */
HG_API hg_tensor_t* hg_tensor_zeros(const int64_t* dims, int64_t ndim,
                                    hg_dtype_t dtype, hg_device_t device);

/** Increase the refcount and return the same pointer. */
HG_API hg_tensor_t* hg_tensor_retain(hg_tensor_t* tensor);

/** Decrease the refcount, freeing the view (and possibly storage) at zero. */
HG_API void hg_tensor_release(hg_tensor_t* tensor);

/** Shape of the view. Valid for the lifetime of `tensor`. */
HG_API const hg_shape_t* hg_tensor_shape(const hg_tensor_t* tensor);

/** Strides of the view in elements. Valid for the lifetime of `tensor`. */
HG_API const int64_t* hg_tensor_strides(const hg_tensor_t* tensor);

HG_API hg_dtype_t hg_tensor_dtype(const hg_tensor_t* tensor);
HG_API hg_device_t hg_tensor_device(const hg_tensor_t* tensor);
HG_API int64_t hg_tensor_numel(const hg_tensor_t* tensor);
HG_API int hg_tensor_is_contiguous(const hg_tensor_t* tensor);
HG_API uint64_t hg_tensor_version(const hg_tensor_t* tensor);

/** Copy exactly `numel * dtype_size` bytes out of the tensor into `dst`. */
HG_API hg_status_t hg_tensor_copy_to_host(const hg_tensor_t* tensor, void* dst,
                                          size_t dst_bytes);

#ifdef __cplusplus
}
#endif

#endif /* HAGANE_HG_TENSOR_H */
