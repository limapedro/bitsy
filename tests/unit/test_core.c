/* C unit tests for core mechanics and the first cpu_ref operator. */

#include <hagane/hagane.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

static void check_status(hg_status_t actual, hg_status_t expected, const char* what)
{
    if (actual != expected) {
        fprintf(stderr, "FAIL %s: expected %s, got %s (%s)\n", what,
                hg_status_name(expected), hg_status_name(actual),
                hg_get_last_error());
        ++g_failures;
    }
}

static void test_dtypes(void)
{
    CHECK(hg_dtype_size(HG_FLOAT32) == 4);
    CHECK(hg_dtype_size(HG_FLOAT16) == 2);
    CHECK(hg_dtype_size(HG_BFLOAT16) == 2);
    CHECK(hg_dtype_size(HG_FLOAT64) == 8);
    CHECK(hg_dtype_size(HG_INT64) == 8);
    CHECK(strcmp(hg_dtype_name(HG_FLOAT32), "float32") == 0);
    CHECK(hg_promote_types(HG_FLOAT32, HG_INT32) == HG_FLOAT32);
    CHECK(hg_promote_types(HG_BOOL, HG_BOOL) == HG_BOOL);
    CHECK(hg_promote_types(HG_INT64, HG_FLOAT32) == HG_FLOAT32);
}

static void test_devices(void)
{
    hg_device_t d;
    CHECK(hg_device_from_string("cpu", &d) == 1);
    CHECK(hg_device_equal(d, hg_device_cpu()));
    CHECK(hg_device_from_string("Vulkan:0", &d) == 1);
    CHECK(d.type == HG_DEVICE_TYPE_VULKAN && d.index == 0);
    CHECK(hg_device_from_string("cuda", &d) == 0);
    CHECK(strstr(hg_get_last_error(), "expected") != NULL);
}

static void test_shapes(void)
{
    int64_t dims[2] = { 2, 3 };
    hg_shape_t a, b, out;
    CHECK(hg_shape_make(dims, 2, &a) == HG_OK);
    CHECK(hg_shape_numel(&a) == 6);
    CHECK(hg_shape_equal(&a, &a));

    int64_t strides[HG_MAX_DIMS];
    CHECK(hg_shape_contiguous_strides(&a, strides) == HG_OK);
    CHECK(strides[0] == 3 && strides[1] == 1);

    int64_t bdims[1] = { 3 };
    CHECK(hg_shape_make(bdims, 1, &b) == HG_OK);
    CHECK(hg_broadcast_shapes(&a, &b, &out) == HG_OK);
    CHECK(out.ndim == 2 && out.dims[0] == 2 && out.dims[1] == 3);

    int64_t cdims[2] = { 3, 1 };
    int64_t ddims[2] = { 1, 4 };
    hg_shape_t c, d;
    CHECK(hg_shape_make(cdims, 2, &c) == HG_OK);
    CHECK(hg_shape_make(ddims, 2, &d) == HG_OK);
    CHECK(hg_broadcast_shapes(&c, &d, &out) == HG_OK);
    CHECK(out.ndim == 2 && out.dims[0] == 3 && out.dims[1] == 4);

    int64_t bad[1] = { 4 };
    hg_shape_t e;
    CHECK(hg_shape_make(bad, 1, &e) == HG_OK);
    check_status(hg_broadcast_shapes(&a, &e, &out), HG_ERR_SHAPE_MISMATCH,
                 "incompatible broadcast");

    hg_shape_t scalar;
    CHECK(hg_shape_make(NULL, 0, &scalar) == HG_OK);
    CHECK(hg_shape_numel(&scalar) == 1);
    CHECK(hg_broadcast_shapes(&scalar, &a, &out) == HG_OK);
    CHECK(hg_shape_equal(&out, &a));
}

static void test_storage(void)
{
    hg_storage_t* s = hg_storage_alloc(16, hg_device_cpu());
    CHECK(s != NULL);
    CHECK(hg_storage_nbytes(s) == 16);
    CHECK(hg_storage_data(s) != NULL);
    CHECK(hg_storage_version(s) == 0);
    hg_storage_bump_version(s);
    CHECK(hg_storage_version(s) == 1);
    hg_storage_retain(s);
    hg_storage_release(s);
    CHECK(hg_storage_nbytes(s) == 16);
    hg_storage_release(s);
}

static void test_tensor_creation(void)
{
    int64_t dims[2] = { 2, 2 };
    float data[4] = { 1.f, 2.f, 3.f, 4.f };
    hg_tensor_t* t = hg_tensor_create(dims, 2, HG_FLOAT32, hg_device_cpu(), data);
    CHECK(t != NULL);
    CHECK(hg_tensor_numel(t) == 4);
    CHECK(hg_tensor_dtype(t) == HG_FLOAT32);
    CHECK(hg_tensor_is_contiguous(t));
    const int64_t* strides = hg_tensor_strides(t);
    CHECK(strides[0] == 2 && strides[1] == 1);

    float out[4] = { 0.f, 0.f, 0.f, 0.f };
    check_status(hg_tensor_copy_to_host(t, out, sizeof(out)), HG_OK, "copy out");
    CHECK(memcmp(out, data, sizeof(out)) == 0);

    hg_tensor_retain(t);
    hg_tensor_release(t);
    CHECK(hg_tensor_numel(t) == 4);
    hg_tensor_release(t);

    hg_tensor_t* z = hg_tensor_zeros(dims, 2, HG_FLOAT32, hg_device_cpu());
    CHECK(z != NULL);
    CHECK(hg_tensor_copy_to_host(z, out, sizeof(out)) == HG_OK);
    CHECK(out[0] == 0.f && out[3] == 0.f);
    hg_tensor_release(z);
}

static void test_add_basic(void)
{
    int64_t dims[1] = { 3 };
    float av[3] = { 1.f, 2.f, 3.f };
    float bv[3] = { 4.f, 5.f, 6.f };
    hg_tensor_t* a = hg_tensor_create(dims, 1, HG_FLOAT32, hg_device_cpu(), av);
    hg_tensor_t* b = hg_tensor_create(dims, 1, HG_FLOAT32, hg_device_cpu(), bv);
    hg_tensor_t* out = NULL;
    check_status(hg_add(a, b, &out), HG_OK, "add basic");
    CHECK(out != NULL);
    float result[3];
    CHECK(hg_tensor_copy_to_host(out, result, sizeof(result)) == HG_OK);
    CHECK(result[0] == 5.f && result[1] == 7.f && result[2] == 9.f);
    hg_tensor_release(out);
    hg_tensor_release(a);
    hg_tensor_release(b);
}

static void test_add_broadcast(void)
{
    int64_t adims[2] = { 2, 3 };
    int64_t bdims[1] = { 3 };
    float av[6] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
    float bv[3] = { 10.f, 20.f, 30.f };
    hg_tensor_t* a = hg_tensor_create(adims, 2, HG_FLOAT32, hg_device_cpu(), av);
    hg_tensor_t* b = hg_tensor_create(bdims, 1, HG_FLOAT32, hg_device_cpu(), bv);
    hg_tensor_t* out = NULL;
    check_status(hg_add(a, b, &out), HG_OK, "add broadcast");
    CHECK(out != NULL && hg_tensor_shape(out)->ndim == 2);
    float result[6];
    CHECK(hg_tensor_copy_to_host(out, result, sizeof(result)) == HG_OK);
    CHECK(result[0] == 11.f && result[2] == 33.f && result[5] == 36.f);
    hg_tensor_release(out);
    hg_tensor_release(a);
    hg_tensor_release(b);
}

static void test_add_scalar(void)
{
    int64_t adims[2] = { 2, 2 };
    float av[4] = { 1.f, 2.f, 3.f, 4.f };
    float scalar = 100.f;
    hg_tensor_t* a = hg_tensor_create(adims, 2, HG_FLOAT32, hg_device_cpu(), av);
    hg_tensor_t* s = hg_tensor_create(NULL, 0, HG_FLOAT32, hg_device_cpu(), &scalar);
    hg_tensor_t* out = NULL;
    check_status(hg_add(a, s, &out), HG_OK, "add scalar");
    float result[4];
    CHECK(hg_tensor_copy_to_host(out, result, sizeof(result)) == HG_OK);
    CHECK(result[0] == 101.f && result[3] == 104.f);
    hg_tensor_release(out);
    hg_tensor_release(a);
    hg_tensor_release(s);
}

static void test_add_empty(void)
{
    int64_t adims[2] = { 0, 2 };
    int64_t bdims[2] = { 1, 2 };
    float bv[2] = { 1.f, 2.f };
    hg_tensor_t* a = hg_tensor_create(adims, 2, HG_FLOAT32, hg_device_cpu(), NULL);
    hg_tensor_t* b = hg_tensor_create(bdims, 2, HG_FLOAT32, hg_device_cpu(), bv);
    hg_tensor_t* out = NULL;
    check_status(hg_add(a, b, &out), HG_OK, "add empty");
    CHECK(out != NULL && hg_tensor_numel(out) == 0);
    hg_tensor_release(out);
    hg_tensor_release(a);
    hg_tensor_release(b);
}

static void test_add_errors(void)
{
    int64_t dims[1] = { 2 };
    float av[2] = { 1.f, 2.f };
    hg_tensor_t* a = hg_tensor_create(dims, 1, HG_FLOAT16, hg_device_cpu(), NULL);
    hg_tensor_t* b = hg_tensor_create(dims, 1, HG_FLOAT16, hg_device_cpu(), NULL);
    hg_tensor_t* out = NULL;
    check_status(hg_add(a, b, &out), HG_ERR_UNSUPPORTED_DTYPE, "add float16");
    CHECK(strstr(hg_get_last_error(), "float32") != NULL);
    hg_tensor_release(a);
    hg_tensor_release(b);

    a = hg_tensor_create(dims, 1, HG_FLOAT32, hg_device_cpu(), av);
    b = hg_tensor_create(dims, 1, HG_FLOAT32, hg_device_cpu(), av);
    int64_t bad[1] = { 3 };
    hg_tensor_t* c = hg_tensor_create(bad, 1, HG_FLOAT32, hg_device_cpu(), NULL);
    check_status(hg_add(a, c, &out), HG_ERR_SHAPE_MISMATCH, "add shape error");
    hg_tensor_release(a);
    hg_tensor_release(b);
    hg_tensor_release(c);

    hg_tensor_t* vk = hg_tensor_empty(dims, 1, HG_FLOAT32,
                                      hg_device_vulkan(0));
    CHECK(vk == NULL);
    CHECK(strstr(hg_get_last_error(), "cpu_ref") != NULL);
}

typedef void (*test_fn)(void);

int main(void)
{
    check_status(hg_init(), HG_OK, "hg_init");

    const test_fn tests[] = {
        test_dtypes, test_devices, test_shapes, test_storage,
        test_tensor_creation, test_add_basic, test_add_broadcast,
        test_add_scalar, test_add_empty, test_add_errors
    };
    const size_t count = sizeof(tests) / sizeof(tests[0]);
    for (size_t i = 0; i < count; ++i) {
        tests[i]();
    }

    check_status(hg_shutdown(), HG_OK, "hg_shutdown");

    if (g_failures == 0) {
        printf("All %zu C unit tests passed.\n", count);
        return 0;
    }
    fprintf(stderr, "%d check(s) failed.\n", g_failures);
    return 1;
}
