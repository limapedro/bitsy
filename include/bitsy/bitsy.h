#ifndef BITSY_BITSY_H
#define BITSY_BITSY_H

/* Umbrella header for the public Bitsy C API. */

#include <bitsy/bt_export.h>
#include <bitsy/bt_error.h>
#include <bitsy/bt_dtype.h>
#include <bitsy/bt_device.h>
#include <bitsy/bt_shape.h>
#include <bitsy/bt_storage.h>
#include <bitsy/bt_tensor.h>
#include <bitsy/bt_ops.h>
#include <bitsy/bt_conv.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BT_VERSION_MAJOR 0
#define BT_VERSION_MINOR 2
#define BT_VERSION_PATCH 0
#define BT_VERSION_STRING "0.2.0-bitsy"

/* Initialize process-wide state (currently just the dispatcher table). */
BT_API bt_status_t bt_init(void);

/* Release process-wide state. */
BT_API bt_status_t bt_shutdown(void);

/* Compiled version string. */
BT_API const char* bt_version(void);

/* Non-zero when a Vulkan compute device was initialized. */
BT_API int bt_has_vulkan(void);

/* Submit all recorded Vulkan work and wait for it. Readback also flushes.
 * Call this before comparing wall time against a synchronized PyTorch op. */
BT_API bt_status_t bt_vulkan_flush(void);

/* GPU time of the last flushed batch, measured with timestamp queries. */
BT_API double bt_vulkan_device_time_ms(void);

#ifdef __cplusplus
}
#endif

#endif /* BITSY_BITSY_H */
