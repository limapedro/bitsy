#ifndef BITSY_BT_DEVICE_H
#define BITSY_BT_DEVICE_H

#include <stddef.h>
#include <bitsy/bt_export.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum bt_device_type_t { BT_DEVICE_TYPE_CPU = 0, BT_DEVICE_TYPE_VULKAN = 1 } bt_device_type_t;

/* torch.device-like: a type plus an ordinal. */
typedef struct bt_device_t {
  bt_device_type_t type;
  int index;
} bt_device_t;

/* The default CPU device. */
BT_API bt_device_t bt_device_cpu(void);

/* Vulkan device `index` (normally 0). */
BT_API bt_device_t bt_device_vulkan(int index);

/* Parse "cpu", "vulkan", "vulkan:0"; returns 0 on invalid input. */
BT_API int bt_device_from_string(const char* text, bt_device_t* out);

/* Non-zero when two device values are identical. */
BT_API int bt_device_equal(bt_device_t a, bt_device_t b);

/* Write "cpu" or "vulkan:N" into buf. */
BT_API void bt_device_name(bt_device_t device, char* buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif /* BITSY_BT_DEVICE_H */
