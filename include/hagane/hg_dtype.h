#ifndef HAGANE_HG_DTYPE_H
#define HAGANE_HG_DTYPE_H

#include <stddef.h>
#include <hagane/hg_export.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Supported element types. BF16 is stored as a packed uint16. */
typedef enum hg_dtype_t {
    HG_FLOAT32 = 0,
    HG_FLOAT16,
    HG_BFLOAT16,
    HG_FLOAT64,
    HG_INT32,
    HG_INT64,
    HG_BOOL,
    HG_UINT8,
    HG_DTYPE_COUNT,
    HG_DTYPE_INVALID = -1
} hg_dtype_t;

/** Size in bytes of one element of `dtype`, or 0 for an invalid value. */
HG_API size_t hg_dtype_size(hg_dtype_t dtype);

/** Canonical PyTorch-like name, e.g. "float32". */
HG_API const char* hg_dtype_name(hg_dtype_t dtype);

/** Non-zero for floating-point dtypes. */
HG_API int hg_dtype_is_floating(hg_dtype_t dtype);

/** Non-zero for integer dtypes (BOOL and UINT8 included). */
HG_API int hg_dtype_is_integer(hg_dtype_t dtype);

/** PyTorch-style type promotion: returns the common dtype of `a` and `b`. */
HG_API hg_dtype_t hg_promote_types(hg_dtype_t a, hg_dtype_t b);

#ifdef __cplusplus
}
#endif

#endif /* HAGANE_HG_DTYPE_H */
