#ifndef HAGANE_HG_OPS_H
#define HAGANE_HG_OPS_H

#include <hagane/hg_export.h>
#include <hagane/hg_error.h>
#include <hagane/hg_tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Elementwise addition with PyTorch broadcasting. Allocates a new output.
 * Supported at M0: float32 tensors on cpu_ref. Other combinations return
 * HG_ERR_NOT_IMPLEMENTED; no kernel ever writes a wrong answer.
 */
HG_API hg_status_t hg_add(const hg_tensor_t* a, const hg_tensor_t* b,
                          hg_tensor_t** out);

#ifdef __cplusplus
}
#endif

#endif /* HAGANE_HG_OPS_H */
