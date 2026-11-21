#include <bitsy/bt_device.h>

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "error_internal.h"

/* *** device *** */

bt_device_t bt_device_cpu(void) { bt_device_t d = { BT_DEVICE_TYPE_CPU, -1 }; return d; }

bt_device_t bt_device_vulkan(int index) { bt_device_t d = { BT_DEVICE_TYPE_VULKAN, index }; return d; }

int bt_device_from_string(const char* text, bt_device_t* out) {
  if (!text || !out) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_device_from_string: NULL argument"), 0;

  char name[64];
  size_t n = 0;
  while (text[n] && n + 1 < sizeof(name)) { name[n] = (char)tolower((unsigned char)text[n]); ++n; }
  name[n] = '\0';

  int index = -1;
  const char* colon = strchr(name, ':');
  if (colon) {  // "vulkan:0" — reject trailing garbage and silly indices
    char* end = NULL;
    long parsed = strtol(colon + 1, &end, 10);
    if (end == colon + 1 || *end || parsed < 0 || parsed > 1024)
      return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_device_from_string: bad index in '%s'", text), 0;
    index = (int)parsed;
  }

  if (!strncmp(name, "cpu", 3) && (!colon || colon == name + 3)) { *out = bt_device_cpu(); return 1; }
  if (!strncmp(name, "vulkan", 6) && (!colon || colon == name + 6)) {
    *out = bt_device_vulkan(index < 0 ? 0 : index);
    return 1;
  }
  return bt_fail(BT_ERR_INVALID_ARGUMENT,
                 "bt_device_from_string: expected 'cpu' or 'vulkan[:N]', got '%s'", text), 0;
}

int bt_device_equal(bt_device_t a, bt_device_t b) { return a.type == b.type && a.index == b.index; }

void bt_device_name(bt_device_t device, char* buf, size_t buf_size) {
  if (!buf || !buf_size) return;
  if (device.type == BT_DEVICE_TYPE_CPU) (void)snprintf(buf, buf_size, "cpu");
  else (void)snprintf(buf, buf_size, "vulkan:%d", device.index);
}
