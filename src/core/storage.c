#include <bitsy/bt_storage.h>

#include <stdlib.h>

#include "error_internal.h"
#include "storage_internal.h"

/* Backend hooks. Missing Vulkan build is an honest error, not a wrong answer. */
#ifdef BT_ENABLE_VULKAN
void* bt_vk_alloc(size_t nbytes);
void bt_vk_free(void* handle);
bt_status_t bt_vk_upload(void* handle, const void* host, size_t nbytes);
bt_status_t bt_vk_readback(void* handle, void* host, size_t nbytes);
#endif

bt_storage_t* bt_storage_alloc(size_t nbytes, bt_device_t device) {
  if (device.type == BT_DEVICE_TYPE_VULKAN) {
#ifdef BT_ENABLE_VULKAN
    bt_storage_t* s = calloc(1, sizeof(*s));
    if (!s) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_storage_alloc: OOM"), NULL;
    if (!(s->gpu = bt_vk_alloc(nbytes))) { free(s); return NULL; }
    s->nbytes = nbytes;
    s->device = device;
    s->refcount = 1;
    return s;
#else
    return bt_fail(BT_ERR_NOT_IMPLEMENTED,
                   "vulkan backend not compiled; configure with -DBT_ENABLE_VULKAN=ON"), NULL;
#endif
  }

  bt_storage_t* s = calloc(1, sizeof(*s));
  if (!s) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_storage_alloc: OOM"), NULL;
  // malloc(0) is implementation-defined; keep a valid (if unusable) pointer.
  if (!(s->data = malloc(nbytes ? nbytes : 1))) {
    free(s);
    return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_storage_alloc: OOM"), NULL;
  }
  s->nbytes = nbytes;
  s->device = device;
  s->refcount = 1;
  return s;
}

bt_storage_t* bt_storage_retain(bt_storage_t* storage) { if (storage) ++storage->refcount; return storage; }

void bt_storage_release(bt_storage_t* storage) {
  if (!storage || --storage->refcount) return;
#ifdef BT_ENABLE_VULKAN
  if (storage->device.type == BT_DEVICE_TYPE_VULKAN) bt_vk_free(storage->gpu);
#endif
  free(storage->data);
  free(storage);
}

void* bt_storage_data(bt_storage_t* storage) { return storage ? storage->data : NULL; }
size_t bt_storage_nbytes(const bt_storage_t* storage) { return storage ? storage->nbytes : 0; }
bt_device_t bt_storage_device(const bt_storage_t* storage) { return storage ? storage->device : bt_device_cpu(); }
uint64_t bt_storage_version(const bt_storage_t* storage) { return storage ? storage->version : 0; }
void bt_storage_bump_version(bt_storage_t* storage) { if (storage) ++storage->version; }
