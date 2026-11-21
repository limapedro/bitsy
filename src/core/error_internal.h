#ifndef HAGANE_SRC_CORE_ERROR_INTERNAL_H
#define HAGANE_SRC_CORE_ERROR_INTERNAL_H

#include <hagane/hg_error.h>

/* Record a formatted thread-local error and return `status` for convenience. */
hg_status_t hg_fail(hg_status_t status, const char* fmt, ...);

#endif /* HAGANE_SRC_CORE_ERROR_INTERNAL_H */
