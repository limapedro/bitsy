#ifndef HAGANE_HG_DEVICE_H
#define HAGANE_HG_DEVICE_H

#include <stddef.h>
#include <hagane/hg_export.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum hg_device_type_t {
    HG_DEVICE_TYPE_CPU = 0,
    HG_DEVICE_TYPE_VULKAN = 1
} hg_device_type_t;

/** A device: a type plus an ordinal index, mirroring torch.device. */
typedef struct hg_device_t {
    hg_device_type_t type;
    int index;
} hg_device_t;

/** The default CPU device. */
HG_API hg_device_t hg_device_cpu(void);

/** Vulkan device `index` (normally 0). */
HG_API hg_device_t hg_device_vulkan(int index);

/** Parse "cpu", "vulkan", "vulkan:0", etc. Returns HG_ERR_INVALID_ARGUMENT. */
HG_API int hg_device_from_string(const char* text, hg_device_t* out);

/** Non-zero when two device values are identical. */
HG_API int hg_device_equal(hg_device_t a, hg_device_t b);

/** Write a canonical device string ("cpu", "vulkan:1") into `buf`. */
HG_API void hg_device_name(hg_device_t device, char* buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif /* HAGANE_HG_DEVICE_H */
