#include <bitsy/bt_storage.h>

#include <stdlib.h>

#include "error_internal.h"

/* Refcounted bytes; views hold a pointer but never own it. */
struct bt_storage {
  void* data;
  size_t nbytes;
  bt_device_t device;
  uint64_t version;
  size_t refcount;
};

/* *** storage *** */

bt_storage_t* bt_storage_alloc(size_t nbytes, bt_device_t device) {
  if (device.type != BT_DEVICE_TYPE_CPU)
    return bt_fail(BT_ERR_NOT_IMPLEMENTED, "bt_storage_alloc: cpu_ref only before M2"), NULL;

  bt_storage_t* s = calloc(1, sizeof(*s));
  if (!s) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_storage_alloc: out of memory"), NULL;
  // malloc(0) is implementation-defined; keep a valid (if unusable) pointer.
  if (!(s->data = malloc(nbytes ? nbytes : 1))) {
    free(s);
    return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_storage_alloc: out of memory"), NULL;
  }
  s->nbytes = nbytes;
  s->device = device;
  s->refcount = 1; // the caller owns the first reference
  return s;
}

bt_storage_t* bt_storage_retain(bt_storage_t* storage) {
  if (storage) ++storage->refcount;
  return storage;
}

void bt_storage_release(bt_storage_t* storage) {
  if (storage && --storage->refcount == 0) { free(storage->data); free(storage); }
}

void* bt_storage_data(bt_storage_t* storage) { return storage ? storage->data : NULL; }
size_t bt_storage_nbytes(const bt_storage_t* storage) { return storage ? storage->nbytes : 0; }
bt_device_t bt_storage_device(const bt_storage_t* storage) {
  return storage ? storage->device : bt_device_cpu();
}
uint64_t bt_storage_version(const bt_storage_t* storage) { return storage ? storage->version : 0; }
void bt_storage_bump_version(bt_storage_t* storage) { if (storage) ++storage->version; }
