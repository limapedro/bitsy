#include <hagane/hg_device.h>

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "error_internal.h"

hg_device_t hg_device_cpu(void)
{
    hg_device_t d = { HG_DEVICE_TYPE_CPU, -1 };
    return d;
}

hg_device_t hg_device_vulkan(int index)
{
    hg_device_t d = { HG_DEVICE_TYPE_VULKAN, index };
    return d;
}

int hg_device_from_string(const char* text, hg_device_t* out)
{
    char name[64];
    size_t n = 0;
    int index = -1;

    if (text == NULL || out == NULL) {
        hg_fail(HG_ERR_INVALID_ARGUMENT, "hg_device_from_string: NULL argument");
        return 0;
    }

    while (text[n] != '\0' && n + 1 < sizeof(name)) {
        name[n] = (char)tolower((unsigned char)text[n]);
        ++n;
    }
    name[n] = '\0';

    const char* colon = strchr(name, ':');
    if (colon != NULL) {
        char* end = NULL;
        long parsed = strtol(colon + 1, &end, 10);
        if (end == colon + 1 || *end != '\0' || parsed < 0 || parsed > 1024) {
            hg_fail(HG_ERR_INVALID_ARGUMENT,
                    "hg_device_from_string: invalid device index in '%s'", text);
            return 0;
        }
        index = (int)parsed;
    }

    if (strncmp(name, "cpu", 3) == 0 && (colon == NULL || colon == name + 3)) {
        *out = hg_device_cpu();
        return 1;
    }
    if (strncmp(name, "vulkan", 6) == 0 &&
        (colon == NULL || colon == name + 6)) {
        *out = hg_device_vulkan(index < 0 ? 0 : index);
        return 1;
    }

    hg_fail(HG_ERR_INVALID_ARGUMENT,
            "hg_device_from_string: expected 'cpu' or 'vulkan[:N]', got '%s'", text);
    return 0;
}

int hg_device_equal(hg_device_t a, hg_device_t b)
{
    return a.type == b.type && a.index == b.index;
}

void hg_device_name(hg_device_t device, char* buf, size_t buf_size)
{
    if (buf == NULL || buf_size == 0) {
        return;
    }
    if (device.type == HG_DEVICE_TYPE_CPU) {
        (void)snprintf(buf, buf_size, "cpu");
    } else {
        (void)snprintf(buf, buf_size, "vulkan:%d", device.index);
    }
}
