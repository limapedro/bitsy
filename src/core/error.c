#include "error_internal.h"

#include <stdarg.h>
#include <stdio.h>

/* Two threads can fail independently, so the last error must be thread-local. */
#if defined(_MSC_VER)
#define BT_TLS __declspec(thread)
#else
#define BT_TLS _Thread_local
#endif

#define BT_ERROR_SIZE 512

static BT_TLS char g_last_error[BT_ERROR_SIZE];
static BT_TLS bt_status_t g_last_status;

/* *** errors *** */

bt_status_t bt_fail(bt_status_t status, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  (void)vsnprintf(g_last_error, sizeof(g_last_error), fmt, args);
  va_end(args);
  g_last_error[BT_ERROR_SIZE - 1] = '\0';
  g_last_status = status;
  return status; /* return value lets callers `return bt_fail(...)` inline */
}

const char* bt_get_last_error(void) { return g_last_error; }

void bt_clear_error(void) {
  g_last_error[0] = '\0';
  g_last_status = BT_OK;
}

const char* bt_status_name(bt_status_t status) {
  switch (status) {
    case BT_OK: return "BT_OK";
    case BT_ERR_INVALID_ARGUMENT: return "BT_ERR_INVALID_ARGUMENT";
    case BT_ERR_OUT_OF_MEMORY: return "BT_ERR_OUT_OF_MEMORY";
    case BT_ERR_NOT_IMPLEMENTED: return "BT_ERR_NOT_IMPLEMENTED";
    case BT_ERR_UNSUPPORTED_DTYPE: return "BT_ERR_UNSUPPORTED_DTYPE";
    case BT_ERR_UNSUPPORTED_DEVICE: return "BT_ERR_UNSUPPORTED_DEVICE";
    case BT_ERR_SHAPE_MISMATCH: return "BT_ERR_SHAPE_MISMATCH";
    case BT_ERR_BACKEND: return "BT_ERR_BACKEND";
    case BT_ERR_INTERNAL: return "BT_ERR_INTERNAL";
    default: return "BT_UNKNOWN_STATUS";
  }
}
