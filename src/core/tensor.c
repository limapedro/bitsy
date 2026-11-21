#include <bitsy/bt_tensor.h>

#include <stdlib.h>
#include <string.h>

#include "error_internal.h"
#include "storage_internal.h"
#include "tensor_internal.h"

#ifdef BT_ENABLE_VULKAN
bt_status_t bt_vk_upload(void* handle, const void* host, size_t nbytes);
bt_status_t bt_vk_readback(void* handle, void* host, size_t nbytes);
#endif

/* A tensor is just a view over refcounted storage. No hidden owners. */
static bt_tensor_t* alloc(const int64_t* dims, int64_t ndim, bt_dtype_t dtype,
                          bt_device_t device, const void* host_data, int zero) {
  bt_clear_error();
  bt_shape_t shape;
  if (bt_shape_make(dims, ndim, &shape) != BT_OK) return NULL;
  if (dtype < 0 || dtype >= BT_DTYPE_COUNT || !bt_dtype_size(dtype))
    return bt_fail(BT_ERR_UNSUPPORTED_DTYPE, "bt_tensor_create: invalid dtype %d", (int)dtype), NULL;

  int64_t numel = bt_shape_numel(&shape);
  size_t nbytes = (size_t)numel * bt_dtype_size(dtype);
  bt_storage_t* storage = bt_storage_alloc(nbytes, device);
  if (!storage) return NULL;

  if (device.type == BT_DEVICE_TYPE_VULKAN) {
#ifdef BT_ENABLE_VULKAN
    if (host_data && nbytes) {
      bt_status_t status = bt_vk_upload(storage->gpu, host_data, nbytes);
      if (status != BT_OK) { bt_storage_release(storage); return NULL; }
    }
#else
    bt_storage_release(storage);
    return bt_fail(BT_ERR_NOT_IMPLEMENTED, "vulkan backend not compiled"), NULL;
#endif
  } else {
    void* data = bt_storage_data(storage);
    if (host_data && numel) memcpy(data, host_data, nbytes);
    else if (zero && nbytes) memset(data, 0, nbytes);
  }

  bt_tensor_t* t = calloc(1, sizeof(*t));
  if (!t) {
    bt_storage_release(storage);
    return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_tensor_create: out of memory"), NULL;
  }
  t->storage = storage;
  t->shape = shape;
  (void)bt_shape_contiguous_strides(&shape, t->strides);
  t->dtype = dtype;
  t->device = device;
  t->refcount = 1;
  return t;
}

/* *** creation *** */

bt_tensor_t* bt_tensor_create(const int64_t* dims, int64_t ndim, bt_dtype_t dtype,
                              bt_device_t device, const void* host_data) {
  return alloc(dims, ndim, dtype, device, host_data, 0);
}
bt_tensor_t* bt_tensor_empty(const int64_t* dims, int64_t ndim, bt_dtype_t dtype, bt_device_t device) {
  return alloc(dims, ndim, dtype, device, NULL, 0);
}
bt_tensor_t* bt_tensor_zeros(const int64_t* dims, int64_t ndim, bt_dtype_t dtype, bt_device_t device) {
  return alloc(dims, ndim, dtype, device, NULL, 1);
}

/* *** ownership and views *** */

bt_tensor_t* bt_tensor_retain(bt_tensor_t* tensor) { if (tensor) ++tensor->refcount; return tensor; }

void bt_tensor_release(bt_tensor_t* tensor) {
  if (tensor && --tensor->refcount == 0) { bt_storage_release(tensor->storage); free(tensor); }
}

const bt_shape_t* bt_tensor_shape(const bt_tensor_t* tensor) { return tensor ? &tensor->shape : NULL; }
const int64_t* bt_tensor_strides(const bt_tensor_t* tensor) { return tensor ? tensor->strides : NULL; }
bt_dtype_t bt_tensor_dtype(const bt_tensor_t* tensor) { return tensor ? tensor->dtype : BT_DTYPE_INVALID; }
bt_device_t bt_tensor_device(const bt_tensor_t* tensor) { return tensor ? tensor->device : bt_device_cpu(); }
int64_t bt_tensor_numel(const bt_tensor_t* tensor) { return tensor ? bt_shape_numel(&tensor->shape) : 0; }
int bt_tensor_is_contiguous(const bt_tensor_t* tensor) {
  return tensor ? bt_shape_is_contiguous(&tensor->shape, tensor->strides) : 0;
}
uint64_t bt_tensor_version(const bt_tensor_t* tensor) {
  return tensor ? bt_storage_version(tensor->storage) : 0;
}

/* *** host transfer *** */

bt_status_t bt_tensor_view(const bt_tensor_t* tensor, const int64_t* dims,
                           int64_t ndim, bt_tensor_t** out) {
  if (!tensor || !dims || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_tensor_view: NULL");
  if (!bt_tensor_is_contiguous(tensor))
    return bt_fail(BT_ERR_NOT_IMPLEMENTED, "bt_tensor_view: only contiguous views yet");
  bt_shape_t shape;
  bt_status_t status = bt_shape_make(dims, ndim, &shape);
  if (status != BT_OK) return status;
  if (bt_shape_numel(&shape) != bt_tensor_numel(tensor))
    return bt_fail(BT_ERR_SHAPE_MISMATCH, "bt_tensor_view: element count changed");
  bt_tensor_t* view = calloc(1, sizeof(*view));
  if (!view) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_tensor_view: OOM");
  view->storage = bt_storage_retain(tensor->storage);
  view->offset = tensor->offset;
  view->shape = shape;
  (void)bt_shape_contiguous_strides(&shape, view->strides);
  view->dtype = tensor->dtype;
  view->device = tensor->device;
  view->refcount = 1;
  *out = view;
  return BT_OK;
}

bt_status_t bt_tensor_transpose_view(const bt_tensor_t* tensor, bt_tensor_t** out) {
  if (!tensor || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_tensor_transpose_view: NULL");
  if (tensor->shape.ndim != 2)
    return bt_fail(BT_ERR_SHAPE_MISMATCH, "bt_tensor_transpose_view: expects 2D input");
  int64_t dims[2] = { tensor->shape.dims[1], tensor->shape.dims[0] };
  bt_shape_t shape;
  bt_status_t status = bt_shape_make(dims, 2, &shape);
  if (status != BT_OK) return status;
  bt_tensor_t* view = calloc(1, sizeof(*view));
  if (!view) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_tensor_transpose_view: OOM");
  view->storage = bt_storage_retain(tensor->storage);
  view->offset = tensor->offset;
  view->shape = shape;
  view->strides[0] = tensor->strides[1];
  view->strides[1] = tensor->strides[0];
  view->dtype = tensor->dtype;
  view->device = tensor->device;
  view->refcount = 1;
  *out = view;
  return BT_OK;
}

bt_status_t bt_tensor_copy_to_host(const bt_tensor_t* tensor, void* dst, size_t dst_bytes) {
  if (!tensor || !dst) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_tensor_copy_to_host: NULL");
  // Strided copy is a separate op; claiming contiguous-only here avoids wrong data.
  if (!bt_tensor_is_contiguous(tensor))
    return bt_fail(BT_ERR_NOT_IMPLEMENTED, "bt_tensor_copy_to_host: strided views not supported yet");
  size_t needed = (size_t)bt_tensor_numel(tensor) * bt_dtype_size(tensor->dtype);
  if (dst_bytes < needed)
    return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_tensor_copy_to_host: dst too small (%zu < %zu)",
                   dst_bytes, needed);
  if (needed) {
    if (tensor->device.type == BT_DEVICE_TYPE_VULKAN) {
#ifdef BT_ENABLE_VULKAN
      return bt_vk_readback(tensor->storage->gpu, dst, needed);
#else
      return bt_fail(BT_ERR_NOT_IMPLEMENTED, "vulkan backend not compiled");
#endif
    }
    const unsigned char* src = bt_storage_data(tensor->storage);
    memcpy(dst, src + tensor->offset * bt_dtype_size(tensor->dtype), needed);
  }
  return BT_OK;
}
