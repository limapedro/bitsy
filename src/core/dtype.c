#include <bitsy/bt_dtype.h>

/* *** sizes and names *** */

size_t bt_dtype_size(bt_dtype_t dtype) {
  switch (dtype) {
    case BT_FLOAT32: case BT_INT32: return 4;
    case BT_FLOAT16: case BT_BFLOAT16: return 2;
    case BT_FLOAT64: case BT_INT64: return 8;
    case BT_BOOL: case BT_UINT8: return 1;
    default: return 0;
  }
}

const char* bt_dtype_name(bt_dtype_t dtype) {
  switch (dtype) {
    case BT_FLOAT32: return "float32";
    case BT_FLOAT16: return "float16";
    case BT_BFLOAT16: return "bfloat16";
    case BT_FLOAT64: return "float64";
    case BT_INT32: return "int32";
    case BT_INT64: return "int64";
    case BT_BOOL: return "bool";
    case BT_UINT8: return "uint8";
    default: return "invalid";
  }
}

int bt_dtype_is_floating(bt_dtype_t dtype) {
  return dtype == BT_FLOAT32 || dtype == BT_FLOAT16 ||
         dtype == BT_BFLOAT16 || dtype == BT_FLOAT64;
}

int bt_dtype_is_integer(bt_dtype_t dtype) {
  return dtype == BT_INT32 || dtype == BT_INT64 ||
         dtype == BT_BOOL || dtype == BT_UINT8;
}

/* *** promotion *** */

/* PyTorch ranks floats above ints for mixed promotion; bool is the bottom. */
static int rank(bt_dtype_t dtype) {
  switch (dtype) {
    case BT_BOOL: return 0;
    case BT_UINT8: return 1;
    case BT_INT32: return 2;
    case BT_INT64: return 3;
    case BT_FLOAT16: return 4;
    case BT_BFLOAT16: return 5;
    case BT_FLOAT32: return 6;
    case BT_FLOAT64: return 7;
    default: return -1;
  }
}

bt_dtype_t bt_promote_types(bt_dtype_t a, bt_dtype_t b) {
  if (a < 0 || b < 0 || a >= BT_DTYPE_COUNT || b >= BT_DTYPE_COUNT) return BT_DTYPE_INVALID;
  return rank(a) >= rank(b) ? a : b;
}
