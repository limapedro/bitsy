#include <hagane/hg_tensor.h>

#include <stdlib.h>
#include <string.h>

#include "error_internal.h"
#include "tensor_internal.h"

static hg_tensor_t* tensor_alloc(const int64_t* dims, int64_t ndim,
                                 hg_dtype_t dtype, hg_device_t device,
                                 const void* host_data, int zero_fill)
{
    hg_clear_error();

    hg_shape_t shape;
    hg_status_t status = hg_shape_make(dims, ndim, &shape);
    if (status != HG_OK) {
        return NULL;
    }
    if (dtype < 0 || dtype >= HG_DTYPE_COUNT || hg_dtype_size(dtype) == 0) {
        hg_fail(HG_ERR_UNSUPPORTED_DTYPE, "hg_tensor_create: invalid dtype %d",
                (int)dtype);
        return NULL;
    }

    int64_t numel = hg_shape_numel(&shape);
    size_t nbytes = (size_t)numel * hg_dtype_size(dtype);
    hg_storage_t* storage = hg_storage_alloc(nbytes, device);
    if (storage == NULL) {
        return NULL;
    }

    void* data = hg_storage_data(storage);
    if (host_data != NULL) {
        if (numel > 0) {
            memcpy(data, host_data, nbytes);
        }
    } else if (zero_fill && nbytes > 0) {
        memset(data, 0, nbytes);
    }

    hg_tensor_t* tensor = (hg_tensor_t*)calloc(1, sizeof(*tensor));
    if (tensor == NULL) {
        hg_storage_release(storage);
        hg_fail(HG_ERR_OUT_OF_MEMORY, "hg_tensor_create: out of memory");
        return NULL;
    }
    tensor->storage = storage;
    tensor->offset = 0;
    tensor->shape = shape;
    (void)hg_shape_contiguous_strides(&shape, tensor->strides);
    tensor->dtype = dtype;
    tensor->device = device;
    tensor->refcount = 1;
    return tensor;
}

hg_tensor_t* hg_tensor_create(const int64_t* dims, int64_t ndim,
                              hg_dtype_t dtype, hg_device_t device,
                              const void* host_data)
{
    return tensor_alloc(dims, ndim, dtype, device, host_data, 0);
}

hg_tensor_t* hg_tensor_empty(const int64_t* dims, int64_t ndim,
                             hg_dtype_t dtype, hg_device_t device)
{
    return tensor_alloc(dims, ndim, dtype, device, NULL, 0);
}

hg_tensor_t* hg_tensor_zeros(const int64_t* dims, int64_t ndim,
                             hg_dtype_t dtype, hg_device_t device)
{
    return tensor_alloc(dims, ndim, dtype, device, NULL, 1);
}

hg_tensor_t* hg_tensor_retain(hg_tensor_t* tensor)
{
    if (tensor != NULL) {
        ++tensor->refcount;
    }
    return tensor;
}

void hg_tensor_release(hg_tensor_t* tensor)
{
    if (tensor == NULL) {
        return;
    }
    if (--tensor->refcount == 0) {
        hg_storage_release(tensor->storage);
        free(tensor);
    }
}

const hg_shape_t* hg_tensor_shape(const hg_tensor_t* tensor)
{
    return tensor == NULL ? NULL : &tensor->shape;
}

const int64_t* hg_tensor_strides(const hg_tensor_t* tensor)
{
    return tensor == NULL ? NULL : tensor->strides;
}

hg_dtype_t hg_tensor_dtype(const hg_tensor_t* tensor)
{
    return tensor == NULL ? HG_DTYPE_INVALID : tensor->dtype;
}

hg_device_t hg_tensor_device(const hg_tensor_t* tensor)
{
    return tensor == NULL ? hg_device_cpu() : tensor->device;
}

int64_t hg_tensor_numel(const hg_tensor_t* tensor)
{
    return tensor == NULL ? 0 : hg_shape_numel(&tensor->shape);
}

int hg_tensor_is_contiguous(const hg_tensor_t* tensor)
{
    if (tensor == NULL) {
        return 0;
    }
    return hg_shape_is_contiguous(&tensor->shape, tensor->strides);
}

uint64_t hg_tensor_version(const hg_tensor_t* tensor)
{
    return tensor == NULL ? 0 : hg_storage_version(tensor->storage);
}

hg_status_t hg_tensor_copy_to_host(const hg_tensor_t* tensor, void* dst,
                                   size_t dst_bytes)
{
    if (tensor == NULL || dst == NULL) {
        return hg_fail(HG_ERR_INVALID_ARGUMENT,
                       "hg_tensor_copy_to_host: NULL argument");
    }
    if (!hg_tensor_is_contiguous(tensor)) {
        return hg_fail(HG_ERR_NOT_IMPLEMENTED,
                       "hg_tensor_copy_to_host: non-contiguous views are not "
                       "supported yet");
    }
    size_t needed = (size_t)hg_tensor_numel(tensor) * hg_dtype_size(tensor->dtype);
    if (dst_bytes < needed) {
        return hg_fail(HG_ERR_INVALID_ARGUMENT,
                       "hg_tensor_copy_to_host: destination too small "
                       "(%zu < %zu bytes)", dst_bytes, needed);
    }
    if (needed > 0) {
        const unsigned char* src = (const unsigned char*)hg_storage_data(tensor->storage);
        memcpy(dst, src + tensor->offset * hg_dtype_size(tensor->dtype), needed);
    }
    return HG_OK;
}
