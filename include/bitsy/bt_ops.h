#ifndef BITSY_BT_OPS_H
#define BITSY_BT_OPS_H

#include <bitsy/bt_export.h>
#include <bitsy/bt_error.h>
#include <bitsy/bt_tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Elementwise add with PyTorch broadcasting; allocates the output. M0
 * supports float32 on cpu_ref; everything else returns BT_ERR_NOT_IMPLEMENTED
 * rather than writing a wrong answer. */
BT_API bt_status_t bt_add(const bt_tensor_t* a, const bt_tensor_t* b,
                          bt_tensor_t** out);

#ifdef __cplusplus
}
#endif

#endif /* BITSY_BT_OPS_H */
