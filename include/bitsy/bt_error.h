#ifndef BITSY_BT_ERROR_H
#define BITSY_BT_ERROR_H

#include <bitsy/bt_export.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Every fallible call returns one of these; no ordinary error escapes as an
 * exception or abort. */
typedef enum bt_status_t {
  BT_OK = 0,
  BT_ERR_INVALID_ARGUMENT,
  BT_ERR_OUT_OF_MEMORY,
  BT_ERR_NOT_IMPLEMENTED,
  BT_ERR_UNSUPPORTED_DTYPE,
  BT_ERR_UNSUPPORTED_DEVICE,
  BT_ERR_SHAPE_MISMATCH,
  BT_ERR_BACKEND,
  BT_ERR_INTERNAL,
  BT_STATUS_COUNT
} bt_status_t;

/* Name of a status code, e.g. "BT_ERR_NOT_IMPLEMENTED". */
BT_API const char* bt_status_name(bt_status_t status);

/* Thread-local description of the last error, or "" when there was none. */
BT_API const char* bt_get_last_error(void);

/* Clear the thread-local error state. */
BT_API void bt_clear_error(void);

#ifdef __cplusplus
}
#endif

#endif /* BITSY_BT_ERROR_H */
