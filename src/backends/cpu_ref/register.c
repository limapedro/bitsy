#include "cpu_ref.h"

#include "../../core/dispatcher_internal.h"

hg_status_t hg_cpu_ref_register(void)
{
    hg_status_t status = hg_register_binary(HG_OP_ADD, HG_BACKEND_CPU_REF,
                                            HG_FLOAT32, hg_cpu_ref_add_float32);
    return status;
}
