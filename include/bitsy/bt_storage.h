#ifndef BITSY_BT_STORAGE_H
#define BITSY_BT_STORAGE_H

#include <stddef.h>
#include <stdint.h>
#include <bitsy/bt_export.h>
#include <bitsy/bt_device.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Refcounted bytes. Views borrow it; only the refcount owns it. */
typedef struct bt_storage bt_storage_t;

/* NULL + error on failure. */
BT_API bt_storage_t* bt_storage_alloc(size_t nbytes, bt_device_t device);

/* Increase the refcount and return the same pointer. */
BT_API bt_storage_t* bt_storage_retain(bt_storage_t* storage);

/* Decrease the refcount, freeing at zero. NULL is a no-op. */
BT_API void bt_storage_release(bt_storage_t* storage);

/* Mutable pointer to the bytes. */
BT_API void* bt_storage_data(bt_storage_t* storage);

/* Size of the buffer in bytes. */
BT_API size_t bt_storage_nbytes(const bt_storage_t* storage);

/* Device the buffer lives on. */
BT_API bt_device_t bt_storage_device(const bt_storage_t* storage);

/* Bumped on every in-place write; autograd uses it to catch invalid views. */
/* Current version counter. */
BT_API uint64_t bt_storage_version(const bt_storage_t* storage);

/* Increment the counter after an in-place write. */
BT_API void bt_storage_bump_version(bt_storage_t* storage);

#ifdef __cplusplus
}
#endif

#endif /* BITSY_BT_STORAGE_H */
