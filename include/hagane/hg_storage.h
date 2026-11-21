#ifndef HAGANE_HG_STORAGE_H
#define HAGANE_HG_STORAGE_H

#include <stddef.h>
#include <stdint.h>
#include <hagane/hg_export.h>
#include <hagane/hg_device.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Refcounted byte buffer. Opaque; use retain/release to manage ownership. */
typedef struct hg_storage hg_storage_t;

/** Allocate `nbytes` on `device`. Returns NULL and sets the error on failure. */
HG_API hg_storage_t* hg_storage_alloc(size_t nbytes, hg_device_t device);

/** Increase the refcount and return the same pointer. */
HG_API hg_storage_t* hg_storage_retain(hg_storage_t* storage);

/** Decrease the refcount, freeing at zero. NULL is a no-op. */
HG_API void hg_storage_release(hg_storage_t* storage);

/** Mutable pointer to the buffer. */
HG_API void* hg_storage_data(hg_storage_t* storage);

/** Immutable size of the buffer in bytes. */
HG_API size_t hg_storage_nbytes(const hg_storage_t* storage);

/** Device the buffer lives on. */
HG_API hg_device_t hg_storage_device(const hg_storage_t* storage);

/** Monotonic version counter, incremented on every in-place mutation. */
HG_API uint64_t hg_storage_version(const hg_storage_t* storage);

/** Increment the version counter. Called by mutating operators. */
HG_API void hg_storage_bump_version(hg_storage_t* storage);

#ifdef __cplusplus
}
#endif

#endif /* HAGANE_HG_STORAGE_H */
