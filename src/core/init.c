#include <hagane/hagane.h>

#include "dispatcher_internal.h"
#include "error_internal.h"

/* Implemented by src/backends/cpu_ref/register.c. */
hg_status_t hg_cpu_ref_register(void);

hg_status_t hg_init(void)
{
    hg_clear_error();
    hg_dispatcher_reset();
    return hg_cpu_ref_register();
}

hg_status_t hg_shutdown(void)
{
    hg_clear_error();
    hg_dispatcher_reset();
    return HG_OK;
}

const char* hg_version(void)
{
    return HG_VERSION_STRING;
}
