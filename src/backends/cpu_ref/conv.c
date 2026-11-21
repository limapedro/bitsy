#include "cpu_ref.h"

#include <stdint.h>

#include "../../core/conv_dispatch_internal.h"
#include "../../core/error_internal.h"
#include "../../core/tensor_internal.h"

#define AT4(ptr, s, n, c, y, x) ptr[(n) * s[0] + (c) * s[1] + (y) * s[2] + (x) * s[3]]

static bt_status_t check(const bt_tensor_t* out) {
  return out && bt_tensor_dtype(out) == BT_FLOAT32 ? BT_OK
    : bt_fail(BT_ERR_INVALID_ARGUMENT, "cpu_ref conv: bad output");
}

static bt_status_t conv_forward(const bt_tensor_t* x, const bt_tensor_t* w,
                                const bt_tensor_t* b, bt_conv_params_t p, bt_tensor_t* out) {
  const bt_shape_t* xs = &x->shape;
  const bt_shape_t* ws = &w->shape;
  int64_t N = xs->dims[0], H = xs->dims[2], Wd = xs->dims[3];
  int64_t O = ws->dims[0], kH = ws->dims[2], kW = ws->dims[3];
  int64_t OH = out->shape.dims[2], OW = out->shape.dims[3];
  const float* xd = (const float*)bt_storage_data(x->storage) + x->offset;
  const float* wd = (const float*)bt_storage_data(w->storage) + w->offset;
  const float* bd = (const float*)bt_storage_data(b->storage) + b->offset;
  float* od = (float*)bt_storage_data(out->storage) + out->offset;
  for (int64_t n = 0; n < N; ++n)
    for (int64_t o = 0; o < O; ++o)
      for (int64_t y = 0; y < OH; ++y)
        for (int64_t x2 = 0; x2 < OW; ++x2) {
          float sum = bd[o];
          for (int64_t c = 0; c < xs->dims[1]; ++c)
            for (int64_t ky = 0; ky < kH; ++ky)
              for (int64_t kx = 0; kx < kW; ++kx) {
                int64_t yy = y * p.stride - p.pad + ky, xx = x2 * p.stride - p.pad + kx;
                if (yy < 0 || xx < 0 || yy >= H || xx >= Wd) continue;
                sum += AT4(xd, x->strides, n, c, yy, xx) * AT4(wd, w->strides, o, c, ky, kx);
              }
          AT4(od, out->strides, n, o, y, x2) = sum;
        }
  return BT_OK;
}

static bt_status_t conv_dx(const bt_tensor_t* dy, const bt_tensor_t* w,
                           bt_conv_params_t p, bt_tensor_t* out) {
  const bt_shape_t* ds = &dy->shape;
  const bt_shape_t* ws = &w->shape;
  int64_t N = ds->dims[0], O = ds->dims[1], OH = ds->dims[2], OW = ds->dims[3];
  int64_t C = ws->dims[1], kH = ws->dims[2], kW = ws->dims[3];
  int64_t H = out->shape.dims[2], Wd = out->shape.dims[3];
  const float* dd = (const float*)bt_storage_data(dy->storage) + dy->offset;
  const float* wd = (const float*)bt_storage_data(w->storage) + w->offset;
  float* xd = (float*)bt_storage_data(out->storage) + out->offset;
  for (int64_t n = 0; n < N; ++n)
    for (int64_t c = 0; c < C; ++c)
      for (int64_t y = 0; y < H; ++y)
        for (int64_t x = 0; x < Wd; ++x) {
          float sum = 0.f;
          for (int64_t o = 0; o < O; ++o)
            for (int64_t ky = 0; ky < kH; ++ky)
              for (int64_t kx = 0; kx < kW; ++kx) {
                int64_t ty = y + p.pad - ky, tx = x + p.pad - kx;
                if (ty < 0 || tx < 0 || ty % p.stride || tx % p.stride) continue;
                int64_t oy = ty / p.stride, ox = tx / p.stride;
                if (oy >= OH || ox >= OW) continue;
                sum += AT4(dd, dy->strides, n, o, oy, ox) * AT4(wd, w->strides, o, c, ky, kx);
              }
          AT4(xd, out->strides, n, c, y, x) = sum;
        }
  return BT_OK;
}

static bt_status_t conv_dw(const bt_tensor_t* x, const bt_tensor_t* dy,
                           bt_conv_params_t p, bt_tensor_t* out) {
  const bt_shape_t* xs = &x->shape;
  const bt_shape_t* ds = &dy->shape;
  int64_t N = xs->dims[0], H = xs->dims[2], Wd = xs->dims[3];
  int64_t O = ds->dims[1], OH = ds->dims[2], OW = ds->dims[3];
  int64_t C = xs->dims[1], kH = out->shape.dims[2], kW = out->shape.dims[3];
  const float* xd = (const float*)bt_storage_data(x->storage) + x->offset;
  const float* dd = (const float*)bt_storage_data(dy->storage) + dy->offset;
  float* wd = (float*)bt_storage_data(out->storage) + out->offset;
  for (int64_t o = 0; o < O; ++o)
    for (int64_t c = 0; c < C; ++c)
      for (int64_t ky = 0; ky < kH; ++ky)
        for (int64_t kx = 0; kx < kW; ++kx) {
          float sum = 0.f;
          for (int64_t n = 0; n < N; ++n)
            for (int64_t y = 0; y < OH; ++y)
              for (int64_t xx = 0; xx < OW; ++xx) {
                int64_t yy = y * p.stride - p.pad + ky, xin = xx * p.stride - p.pad + kx;
                if (yy < 0 || xin < 0 || yy >= H || xin >= Wd) continue;
                sum += AT4(dd, dy->strides, n, o, y, xx) * AT4(xd, x->strides, n, c, yy, xin);
              }
          AT4(wd, out->strides, o, c, ky, kx) = sum;
        }
  return BT_OK;
}

static bt_status_t conv_db(const bt_tensor_t* dy, int64_t O, bt_tensor_t* out) {
  const bt_shape_t* ds = &dy->shape;
  const float* dd = (const float*)bt_storage_data(dy->storage) + dy->offset;
  float* bd = (float*)bt_storage_data(out->storage) + out->offset;
  for (int64_t o = 0; o < O; ++o) {
    float sum = 0.f;
    for (int64_t n = 0; n < ds->dims[0]; ++n)
      for (int64_t y = 0; y < ds->dims[2]; ++y)
        for (int64_t x = 0; x < ds->dims[3]; ++x)
          sum += AT4(dd, dy->strides, n, o, y, x);
    bd[o] = sum;
  }
  return BT_OK;
}

bt_status_t bt_cpu_ref_conv(const bt_tensor_t* a, const bt_tensor_t* b,
                            const bt_tensor_t* c, bt_conv_params_t params,
                            bt_conv_kind_t kind, bt_tensor_t* out) {
  bt_status_t status = check(out);
  if (status != BT_OK) return status;
  switch (kind) {
    case BT_CONV_FORWARD: return conv_forward(a, b, c, params, out);
    case BT_CONV_DX: return conv_dx(a, b, params, out);
    case BT_CONV_DW: return conv_dw(a, b, params, out);
    case BT_CONV_DB: return conv_db(a, out->shape.dims[0], out);
    default: return bt_fail(BT_ERR_INVALID_ARGUMENT, "cpu_ref conv: unknown kind");
  }
}
