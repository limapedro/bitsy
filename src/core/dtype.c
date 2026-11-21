#include <hagane/hg_dtype.h>

#include <string.h>

size_t hg_dtype_size(hg_dtype_t dtype)
{
    switch (dtype) {
        case HG_FLOAT32: return 4;
        case HG_FLOAT16: return 2;
        case HG_BFLOAT16: return 2;
        case HG_FLOAT64: return 8;
        case HG_INT32: return 4;
        case HG_INT64: return 8;
        case HG_BOOL: return 1;
        case HG_UINT8: return 1;
        default: return 0;
    }
}

const char* hg_dtype_name(hg_dtype_t dtype)
{
    switch (dtype) {
        case HG_FLOAT32: return "float32";
        case HG_FLOAT16: return "float16";
        case HG_BFLOAT16: return "bfloat16";
        case HG_FLOAT64: return "float64";
        case HG_INT32: return "int32";
        case HG_INT64: return "int64";
        case HG_BOOL: return "bool";
        case HG_UINT8: return "uint8";
        default: return "invalid";
    }
}

int hg_dtype_is_floating(hg_dtype_t dtype)
{
    return dtype == HG_FLOAT32 || dtype == HG_FLOAT16 ||
           dtype == HG_BFLOAT16 || dtype == HG_FLOAT64;
}

int hg_dtype_is_integer(hg_dtype_t dtype)
{
    return dtype == HG_INT32 || dtype == HG_INT64 ||
           dtype == HG_BOOL || dtype == HG_UINT8;
}

/* PyTorch puts bool below the integer types and floating types above them;
 * mixed int/float promotes to the floating type. This rank table reproduces
 * those cases for the dtypes Hagane supports today. */
static int dtype_rank(hg_dtype_t dtype)
{
    switch (dtype) {
        case HG_BOOL: return 0;
        case HG_UINT8: return 1;
        case HG_INT32: return 2;
        case HG_INT64: return 3;
        case HG_FLOAT16: return 4;
        case HG_BFLOAT16: return 5;
        case HG_FLOAT32: return 6;
        case HG_FLOAT64: return 7;
        default: return -1;
    }
}

hg_dtype_t hg_promote_types(hg_dtype_t a, hg_dtype_t b)
{
    if (a == HG_DTYPE_INVALID || b == HG_DTYPE_INVALID ||
        a >= HG_DTYPE_COUNT || b >= HG_DTYPE_COUNT) {
        return HG_DTYPE_INVALID;
    }
    return dtype_rank(a) >= dtype_rank(b) ? a : b;
}
