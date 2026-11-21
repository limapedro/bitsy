#ifndef BITSY_BT_DTYPE_H
#define BITSY_BT_DTYPE_H

#include <stddef.h>
#include <bitsy/bt_export.h>

#ifdef __cplusplus
extern "C" {
#endif

/* BF16 is stored as a packed uint16: Turing has no native BF16. */
typedef enum bt_dtype_t {
  BT_FLOAT32 = 0,
  BT_FLOAT16,
  BT_BFLOAT16,
  BT_FLOAT64,
  BT_INT32,
  BT_INT64,
  BT_BOOL,
  BT_UINT8,
  BT_DTYPE_COUNT,
  BT_DTYPE_INVALID = -1
} bt_dtype_t;

/* Bytes per element, or 0 for an invalid dtype. */
BT_API size_t bt_dtype_size(bt_dtype_t dtype);

/* PyTorch-like name, e.g. "float32". */
BT_API const char* bt_dtype_name(bt_dtype_t dtype);

/* Non-zero for floating-point dtypes. */
BT_API int bt_dtype_is_floating(bt_dtype_t dtype);

/* Non-zero for integer dtypes (BOOL and UINT8 included). */
BT_API int bt_dtype_is_integer(bt_dtype_t dtype);

/* PyTorch-style promotion: the common dtype of two inputs. */
BT_API bt_dtype_t bt_promote_types(bt_dtype_t a, bt_dtype_t b);

#ifdef __cplusplus
}
#endif

#endif /* BITSY_BT_DTYPE_H */
