/* C unit tests for core mechanics and the first cpu_ref operator. */

#include <bitsy/bitsy.h>

#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(c) do { \
  if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++failures; } \
} while (0)

static void check_status(bt_status_t got, bt_status_t want, const char* what) {
  if (got == want) return;
  fprintf(stderr, "FAIL %s: want %s, got %s (%s)\n",
          what, bt_status_name(want), bt_status_name(got), bt_get_last_error());
  ++failures;
}

/* *** dtypes and devices *** */

static void dtypes(void) {
  CHECK(bt_dtype_size(BT_FLOAT32) == 4);
  CHECK(bt_dtype_size(BT_FLOAT16) == 2);
  CHECK(bt_dtype_size(BT_BFLOAT16) == 2);
  CHECK(bt_dtype_size(BT_INT64) == 8);
  CHECK(!strcmp(bt_dtype_name(BT_FLOAT32), "float32"));
  CHECK(bt_promote_types(BT_FLOAT32, BT_INT32) == BT_FLOAT32);
  CHECK(bt_promote_types(BT_INT64, BT_FLOAT32) == BT_FLOAT32);
  CHECK(bt_promote_types(BT_BOOL, BT_BOOL) == BT_BOOL);
}

static void devices(void) {
  bt_device_t d;
  CHECK(bt_device_from_string("cpu", &d) && bt_device_equal(d, bt_device_cpu()));
  CHECK(bt_device_from_string("Vulkan:0", &d) && d.type == BT_DEVICE_TYPE_VULKAN && d.index == 0);
  CHECK(!bt_device_from_string("cuda", &d) && strstr(bt_get_last_error(), "expected"));
}

/* *** shape math *** */

static void shapes(void) {
  int64_t dims[2] = { 2, 3 }, bad[1] = { 4 }, bdims[1] = { 3 };
  bt_shape_t a, b, c, d, e, out;
  CHECK(bt_shape_make(dims, 2, &a) == BT_OK);
  CHECK(bt_shape_numel(&a) == 6 && bt_shape_equal(&a, &a));

  int64_t strides[BT_MAX_DIMS];
  CHECK(bt_shape_contiguous_strides(&a, strides) == BT_OK);
  CHECK(strides[0] == 3 && strides[1] == 1);

  CHECK(bt_shape_make(bdims, 1, &b) == BT_OK);
  CHECK(bt_broadcast_shapes(&a, &b, &out) == BT_OK && out.ndim == 2);
  CHECK(out.dims[0] == 2 && out.dims[1] == 3);

  int64_t cd[2] = { 3, 1 }, dd[2] = { 1, 4 };
  CHECK(bt_shape_make(cd, 2, &c) == BT_OK && bt_shape_make(dd, 2, &d) == BT_OK);
  CHECK(bt_broadcast_shapes(&c, &d, &out) == BT_OK && out.dims[0] == 3 && out.dims[1] == 4);

  CHECK(bt_shape_make(bad, 1, &e) == BT_OK);
  check_status(bt_broadcast_shapes(&a, &e, &out), BT_ERR_SHAPE_MISMATCH, "broadcast error");

  bt_shape_t scalar;
  CHECK(bt_shape_make(NULL, 0, &scalar) == BT_OK && bt_shape_numel(&scalar) == 1);
  CHECK(bt_broadcast_shapes(&scalar, &a, &out) == BT_OK && bt_shape_equal(&out, &a));
}

/* *** storage and tensor views *** */

static void storage(void) {
  bt_storage_t* s = bt_storage_alloc(16, bt_device_cpu());
  CHECK(s && bt_storage_nbytes(s) == 16 && bt_storage_data(s));
  CHECK(bt_storage_version(s) == 0);
  bt_storage_bump_version(s);
  CHECK(bt_storage_version(s) == 1);
  bt_storage_retain(s);
  bt_storage_release(s);
  CHECK(bt_storage_nbytes(s) == 16);
  bt_storage_release(s);
}

static void tensor_creation(void) {
  int64_t dims[2] = { 2, 2 };
  float data[4] = { 1.f, 2.f, 3.f, 4.f }, out[4] = { 0 };
  bt_tensor_t* t = bt_tensor_create(dims, 2, BT_FLOAT32, bt_device_cpu(), data);
  CHECK(t && bt_tensor_numel(t) == 4 && bt_tensor_dtype(t) == BT_FLOAT32);
  CHECK(bt_tensor_is_contiguous(t));
  CHECK(bt_tensor_strides(t)[0] == 2 && bt_tensor_strides(t)[1] == 1);
  check_status(bt_tensor_copy_to_host(t, out, sizeof(out)), BT_OK, "copy out");
  CHECK(!memcmp(out, data, sizeof(out)));
  bt_tensor_retain(t);
  bt_tensor_release(t);
  CHECK(bt_tensor_numel(t) == 4);
  bt_tensor_release(t);

  bt_tensor_t* z = bt_tensor_zeros(dims, 2, BT_FLOAT32, bt_device_cpu());
  CHECK(bt_tensor_copy_to_host(z, out, sizeof(out)) == BT_OK && out[0] == 0.f && out[3] == 0.f);
  bt_tensor_release(z);
}

/* *** first operator *** */

static void add_basic(void) {
  int64_t dims[1] = { 3 };
  float av[3] = { 1.f, 2.f, 3.f }, bv[3] = { 4.f, 5.f, 6.f }, out[3];
  bt_tensor_t* a = bt_tensor_create(dims, 1, BT_FLOAT32, bt_device_cpu(), av);
  bt_tensor_t* b = bt_tensor_create(dims, 1, BT_FLOAT32, bt_device_cpu(), bv);
  bt_tensor_t* r = NULL;
  check_status(bt_add(a, b, &r), BT_OK, "add basic");
  CHECK(r && bt_tensor_copy_to_host(r, out, sizeof(out)) == BT_OK);
  CHECK(out[0] == 5.f && out[1] == 7.f && out[2] == 9.f);
  bt_tensor_release(r); bt_tensor_release(a); bt_tensor_release(b);
}

static void add_broadcast(void) {
  int64_t ad[2] = { 2, 3 }, bd[1] = { 3 };
  float av[6] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f }, bv[3] = { 10.f, 20.f, 30.f }, out[6];
  bt_tensor_t* a = bt_tensor_create(ad, 2, BT_FLOAT32, bt_device_cpu(), av);
  bt_tensor_t* b = bt_tensor_create(bd, 1, BT_FLOAT32, bt_device_cpu(), bv);
  bt_tensor_t* r = NULL;
  check_status(bt_add(a, b, &r), BT_OK, "add broadcast");
  CHECK(r && bt_tensor_shape(r)->ndim == 2);
  CHECK(bt_tensor_copy_to_host(r, out, sizeof(out)) == BT_OK);
  CHECK(out[0] == 11.f && out[2] == 33.f && out[5] == 36.f);
  bt_tensor_release(r); bt_tensor_release(a); bt_tensor_release(b);
}

static void add_scalar(void) {
  int64_t dims[2] = { 2, 2 };
  float av[4] = { 1.f, 2.f, 3.f, 4.f }, scalar = 100.f, out[4];
  bt_tensor_t* a = bt_tensor_create(dims, 2, BT_FLOAT32, bt_device_cpu(), av);
  bt_tensor_t* s = bt_tensor_create(NULL, 0, BT_FLOAT32, bt_device_cpu(), &scalar);
  bt_tensor_t* r = NULL;
  check_status(bt_add(a, s, &r), BT_OK, "add scalar");
  CHECK(bt_tensor_copy_to_host(r, out, sizeof(out)) == BT_OK);
  CHECK(out[0] == 101.f && out[3] == 104.f);
  bt_tensor_release(r); bt_tensor_release(a); bt_tensor_release(s);
}

static void add_empty(void) {
  int64_t ad[2] = { 0, 2 }, bd[2] = { 1, 2 };
  float bv[2] = { 1.f, 2.f };
  bt_tensor_t* a = bt_tensor_create(ad, 2, BT_FLOAT32, bt_device_cpu(), NULL);
  bt_tensor_t* b = bt_tensor_create(bd, 2, BT_FLOAT32, bt_device_cpu(), bv);
  bt_tensor_t* r = NULL;
  check_status(bt_add(a, b, &r), BT_OK, "add empty");
  CHECK(r && bt_tensor_numel(r) == 0);
  bt_tensor_release(r); bt_tensor_release(a); bt_tensor_release(b);
}

static void add_errors(void) {
  int64_t dims[1] = { 2 }, bad[1] = { 3 };
  float av[2] = { 1.f, 2.f };
  bt_tensor_t* a = bt_tensor_create(dims, 1, BT_FLOAT16, bt_device_cpu(), NULL);
  bt_tensor_t* b = bt_tensor_create(dims, 1, BT_FLOAT16, bt_device_cpu(), NULL);
  bt_tensor_t* r = NULL;
  check_status(bt_add(a, b, &r), BT_ERR_UNSUPPORTED_DTYPE, "add float16");
  CHECK(strstr(bt_get_last_error(), "float32"));
  bt_tensor_release(a); bt_tensor_release(b);

  a = bt_tensor_create(dims, 1, BT_FLOAT32, bt_device_cpu(), av);
  b = bt_tensor_create(dims, 1, BT_FLOAT32, bt_device_cpu(), av);
  bt_tensor_t* c = bt_tensor_create(bad, 1, BT_FLOAT32, bt_device_cpu(), NULL);
  check_status(bt_add(a, c, &r), BT_ERR_SHAPE_MISMATCH, "add shape error");
  bt_tensor_release(a); bt_tensor_release(b); bt_tensor_release(c);

  bt_tensor_t* vk = bt_tensor_empty(dims, 1, BT_FLOAT32, bt_device_vulkan(0));
  if (bt_has_vulkan()) {
    CHECK(vk && bt_tensor_device(vk).type == BT_DEVICE_TYPE_VULKAN);
    bt_tensor_release(vk);
  } else {
    CHECK(!vk && strstr(bt_get_last_error(), "vulkan backend not compiled"));
  }
}

/* GPU smoke test: upload, dispatch the shader, read back. */
static void vulkan_add(void) {
  if (!bt_has_vulkan()) return;
  int64_t dims[1] = { 3 };
  float av[3] = { 1.f, 2.f, 3.f }, bv[3] = { 10.f, 20.f, 30.f }, out[3];
  bt_device_t gpu = bt_device_vulkan(0);
  bt_tensor_t* a = bt_tensor_create(dims, 1, BT_FLOAT32, gpu, av);
  bt_tensor_t* b = bt_tensor_create(dims, 1, BT_FLOAT32, gpu, bv);
  bt_tensor_t* r = NULL;
  check_status(bt_add(a, b, &r), BT_OK, "vulkan add");
  CHECK(r && bt_tensor_copy_to_host(r, out, sizeof(out)) == BT_OK);
  CHECK(out[0] == 11.f && out[1] == 22.f && out[2] == 33.f);
  bt_tensor_release(r); bt_tensor_release(a); bt_tensor_release(b);
}

/* GEMM smoke test on both backends. */
static void matmul_test(void) {
  int64_t ad[2] = { 2, 3 }, bd[2] = { 3, 2 };
  float av[6] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f }, bv[6] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
  float out[4];
  bt_device_t devices[2] = { bt_device_cpu(), bt_device_vulkan(0) };
  for (int i = 0; i < 2; ++i) {
    if (devices[i].type == BT_DEVICE_TYPE_VULKAN && !bt_has_vulkan()) continue;
    bt_tensor_t* a = bt_tensor_create(ad, 2, BT_FLOAT32, devices[i], av);
    bt_tensor_t* b = bt_tensor_create(bd, 2, BT_FLOAT32, devices[i], bv);
    bt_tensor_t* r = NULL;
    check_status(bt_matmul(a, b, &r), BT_OK, "matmul");
    CHECK(r && bt_tensor_copy_to_host(r, out, sizeof(out)) == BT_OK);
    CHECK(out[0] == 22.f && out[1] == 28.f && out[2] == 49.f && out[3] == 64.f);
    bt_tensor_release(r); bt_tensor_release(a); bt_tensor_release(b);
  }
}

/* Contiguous reshape shares storage; 2D transpose matches PyTorch's .t(). */
static void view_transpose(void) {
  int64_t ad[2] = { 2, 3 }, flat[1] = { 6 }, td[2] = { 3, 2 };
  float av[6] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f }, out[6], trans[6];
  bt_tensor_t* a = bt_tensor_create(ad, 2, BT_FLOAT32, bt_device_cpu(), av);
  bt_tensor_t* v = NULL;
  check_status(bt_tensor_view(a, flat, 1, &v), BT_OK, "view");
  CHECK(v && bt_tensor_numel(v) == 6 && bt_tensor_shape(v)->ndim == 1);
  CHECK(bt_tensor_copy_to_host(v, out, sizeof(out)) == BT_OK && !memcmp(out, av, sizeof(out)));
  bt_tensor_release(v);

  check_status(bt_transpose(a, &v), BT_OK, "transpose");
  CHECK(v && bt_tensor_shape(v)->ndim == 2);
  CHECK(bt_tensor_shape(v)->dims[0] == td[0] && bt_tensor_shape(v)->dims[1] == td[1]);
  CHECK(bt_tensor_copy_to_host(v, trans, sizeof(trans)) == BT_OK);
  CHECK(trans[0] == 1.f && trans[1] == 4.f && trans[5] == 6.f);
  bt_tensor_release(v);
  bt_tensor_release(a);
}

/* *** suite *** */

typedef void (*test_fn)(void);

int main(void) {
  check_status(bt_init(), BT_OK, "bt_init");
  const test_fn tests[] = {
    dtypes, devices, shapes, storage, tensor_creation,
    add_basic, add_broadcast, add_scalar, add_empty, add_errors, vulkan_add,
    matmul_test, view_transpose
  };
  const size_t count = sizeof(tests) / sizeof(tests[0]);
  for (size_t i = 0; i < count; ++i) tests[i]();
  check_status(bt_shutdown(), BT_OK, "bt_shutdown");
  if (!failures) { printf("All %zu C unit tests passed.\n", count); return 0; }
  fprintf(stderr, "%d check(s) failed.\n", failures);
  return 1;
}
