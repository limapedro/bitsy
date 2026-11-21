#ifndef HAGANE_SRC_CORE_TENSOR_INTERNAL_H
#define HAGANE_SRC_CORE_TENSOR_INTERNAL_H

#include <stdint.h>
#include <hagane/hg_dtype.h>
#include <hagane/hg_device.h>
#include <hagane/hg_shape.h>
#include <hagane/hg_storage.h>
#include <hagane/hg_tensor.h>

/* Definition of the opaque public type. Views share refcounted storage. */
struct hg_tensor {
    hg_storage_t* storage;
    int64_t offset;                 /* element offset into storage */
    hg_shape_t shape;
    int64_t strides[HG_MAX_DIMS];   /* elements, not bytes */
    hg_dtype_t dtype;
    hg_device_t device;
    int requires_grad;
    int refcount;
};

#endif /* HAGANE_SRC_CORE_TENSOR_INTERNAL_H */
