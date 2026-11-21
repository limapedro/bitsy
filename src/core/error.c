#include "error_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#if defined(_MSC_VER)
#  define HG_THREAD_LOCAL __declspec(thread)
#else
#  define HG_THREAD_LOCAL _Thread_local
#endif

#define HG_ERROR_BUFFER_SIZE 512

static HG_THREAD_LOCAL char g_last_error[HG_ERROR_BUFFER_SIZE];
static HG_THREAD_LOCAL hg_status_t g_last_status;

hg_status_t hg_fail(hg_status_t status, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    (void)vsnprintf(g_last_error, sizeof(g_last_error), fmt, args);
    va_end(args);
    g_last_error[HG_ERROR_BUFFER_SIZE - 1] = '\0';
    g_last_status = status;
    return status;
}

const char* hg_get_last_error(void)
{
    return g_last_error;
}

void hg_clear_error(void)
{
    g_last_error[0] = '\0';
    g_last_status = HG_OK;
}

const char* hg_status_name(hg_status_t status)
{
    switch (status) {
        case HG_OK: return "HG_OK";
        case HG_ERR_INVALID_ARGUMENT: return "HG_ERR_INVALID_ARGUMENT";
        case HG_ERR_OUT_OF_MEMORY: return "HG_ERR_OUT_OF_MEMORY";
        case HG_ERR_NOT_IMPLEMENTED: return "HG_ERR_NOT_IMPLEMENTED";
        case HG_ERR_UNSUPPORTED_DTYPE: return "HG_ERR_UNSUPPORTED_DTYPE";
        case HG_ERR_UNSUPPORTED_DEVICE: return "HG_ERR_UNSUPPORTED_DEVICE";
        case HG_ERR_SHAPE_MISMATCH: return "HG_ERR_SHAPE_MISMATCH";
        case HG_ERR_BACKEND: return "HG_ERR_BACKEND";
        case HG_ERR_INTERNAL: return "HG_ERR_INTERNAL";
        default: return "HG_UNKNOWN_STATUS";
    }
}
