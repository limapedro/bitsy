#ifndef BITSY_SRC_CORE_STORAGE_INTERNAL_H
#define BITSY_SRC_CORE_STORAGE_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include <bitsy/bt_storage.h>

/* Bytes plus one opaque backend handle. Views borrow this, never own it. */
struct bt_storage {
  void* data;
  size_t nbytes;
  bt_device_t device;
  uint64_t version;
  size_t refcount;
  void* gpu; /* Vulkan buffer bundle when device.type == BT_DEVICE_TYPE_VULKAN */
};

#endif /* BITSY_SRC_CORE_STORAGE_INTERNAL_H */
