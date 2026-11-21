#ifndef BITSY_SRC_CORE_TENSOR_INTERNAL_H
#define BITSY_SRC_CORE_TENSOR_INTERNAL_H

#include <stdint.h>
#include <bitsy/bt_dtype.h>
#include <bitsy/bt_device.h>
#include <bitsy/bt_shape.h>
#include <bitsy/bt_storage.h>
#include <bitsy/bt_tensor.h>

/* Definition of the opaque type. A view borrows storage; it owns nothing. */
struct bt_tensor {
  bt_storage_t* storage;
  int64_t offset;                 /* elements, not bytes */
  bt_shape_t shape;
  int64_t strides[BT_MAX_DIMS];   /* elements */
  bt_dtype_t dtype;
  bt_device_t device;
  int requires_grad;
  int refcount;
};

#endif /* BITSY_SRC_CORE_TENSOR_INTERNAL_H */
