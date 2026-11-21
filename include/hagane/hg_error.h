#ifndef HAGANE_HG_ERROR_H
#define HAGANE_HG_ERROR_H

#include <hagane/hg_export.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Status codes returned by every fallible Hagane API function. */
typedef enum hg_status_t {
    HG_OK = 0,
    HG_ERR_INVALID_ARGUMENT,
    HG_ERR_OUT_OF_MEMORY,
    HG_ERR_NOT_IMPLEMENTED,
    HG_ERR_UNSUPPORTED_DTYPE,
    HG_ERR_UNSUPPORTED_DEVICE,
    HG_ERR_SHAPE_MISMATCH,
    HG_ERR_BACKEND,
    HG_ERR_INTERNAL,
    HG_STATUS_COUNT
} hg_status_t;

/** Human-readable name of a status code, e.g. "HG_ERR_NOT_IMPLEMENTED". */
HG_API const char* hg_status_name(hg_status_t status);

/** Thread-local description of the most recent error, or "" when none. */
HG_API const char* hg_get_last_error(void);

/** Clear the thread-local error state. */
HG_API void hg_clear_error(void);

#ifdef __cplusplus
}
#endif

#endif /* HAGANE_HG_ERROR_H */
