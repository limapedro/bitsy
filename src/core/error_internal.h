#ifndef BITSY_SRC_CORE_ERROR_INTERNAL_H
#define BITSY_SRC_CORE_ERROR_INTERNAL_H

#include <bitsy/bt_error.h>

/* Record a formatted thread-local error and return `status` for convenience. */
bt_status_t bt_fail(bt_status_t status, const char* fmt, ...);

#endif /* BITSY_SRC_CORE_ERROR_INTERNAL_H */
