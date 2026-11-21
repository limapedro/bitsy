#ifndef HAGANE_HAGANE_H
#define HAGANE_HAGANE_H

/** Umbrella header for the public Hagane C API. */

#include <hagane/hg_export.h>
#include <hagane/hg_error.h>
#include <hagane/hg_dtype.h>
#include <hagane/hg_device.h>
#include <hagane/hg_shape.h>
#include <hagane/hg_storage.h>
#include <hagane/hg_tensor.h>
#include <hagane/hg_ops.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HG_VERSION_MAJOR 0
#define HG_VERSION_MINOR 1
#define HG_VERSION_PATCH 0
#define HG_VERSION_STRING "0.1.0-m0"

/** Initialize process-wide state (currently only the dispatcher table). */
HG_API hg_status_t hg_init(void);

/** Release process-wide state. Safe to call once per successful hg_init. */
HG_API hg_status_t hg_shutdown(void);

/** Compile-time version string. */
HG_API const char* hg_version(void);

#ifdef __cplusplus
}
#endif

#endif /* HAGANE_HAGANE_H */
