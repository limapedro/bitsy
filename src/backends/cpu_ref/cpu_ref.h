#ifndef HAGANE_SRC_BACKENDS_CPU_REF_CPU_REF_H
#define HAGANE_SRC_BACKENDS_CPU_REF_CPU_REF_H

#include <hagane/hg_error.h>
#include <hagane/hg_tensor.h>

/** Register the cpu_ref kernel implementations with the dispatcher. */
hg_status_t hg_cpu_ref_register(void);

/** Naive broadcasting FP32 addition. Output is allocated by the caller. */
hg_status_t hg_cpu_ref_add_float32(const hg_tensor_t* a, const hg_tensor_t* b,
                                   hg_tensor_t* out);

#endif /* HAGANE_SRC_BACKENDS_CPU_REF_CPU_REF_H */
