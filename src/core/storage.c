#include <hagane/hg_storage.h>

#include <stdlib.h>
#include <string.h>

#include "error_internal.h"

struct hg_storage {
    void* data;
    size_t nbytes;
    hg_device_t device;
    uint64_t version;
    size_t refcount;
};

hg_storage_t* hg_storage_alloc(size_t nbytes, hg_device_t device)
{
    if (device.type != HG_DEVICE_TYPE_CPU) {
        hg_fail(HG_ERR_NOT_IMPLEMENTED,
                "hg_storage_alloc: only cpu_ref storage is available before M2");
        return NULL;
    }

    hg_storage_t* s = (hg_storage_t*)calloc(1, sizeof(*s));
    if (s == NULL) {
        hg_fail(HG_ERR_OUT_OF_MEMORY, "hg_storage_alloc: out of memory");
        return NULL;
    }
    /* malloc(0) is implementation-defined; allocate one byte for empty data. */
    s->data = malloc(nbytes == 0 ? 1 : nbytes);
    if (s->data == NULL) {
        free(s);
        hg_fail(HG_ERR_OUT_OF_MEMORY, "hg_storage_alloc: out of memory");
        return NULL;
    }
    s->nbytes = nbytes;
    s->device = device;
    s->version = 0;
    s->refcount = 1;
    return s;
}

hg_storage_t* hg_storage_retain(hg_storage_t* storage)
{
    if (storage != NULL) {
        ++storage->refcount;
    }
    return storage;
}

void hg_storage_release(hg_storage_t* storage)
{
    if (storage == NULL) {
        return;
    }
    if (--storage->refcount == 0) {
        free(storage->data);
        free(storage);
    }
}

void* hg_storage_data(hg_storage_t* storage)
{
    return storage == NULL ? NULL : storage->data;
}

size_t hg_storage_nbytes(const hg_storage_t* storage)
{
    return storage == NULL ? 0 : storage->nbytes;
}

hg_device_t hg_storage_device(const hg_storage_t* storage)
{
    if (storage == NULL) {
        return hg_device_cpu();
    }
    return storage->device;
}

uint64_t hg_storage_version(const hg_storage_t* storage)
{
    return storage == NULL ? 0 : storage->version;
}

void hg_storage_bump_version(hg_storage_t* storage)
{
    if (storage != NULL) {
        ++storage->version;
    }
}
