#include <bitsy/bitsy.h>

#include "dispatcher_internal.h"
#include "error_internal.h"

#ifdef BT_ENABLE_VULKAN
#include "../backends/vulkan/vk_internal.h"
#endif

/* Implemented by src/backends/cpu_ref/register.c. */
bt_status_t bt_cpu_ref_register(void);

/* *** process state *** */

bt_status_t bt_init(void) {
  bt_clear_error();
  bt_dispatcher_reset();
  bt_status_t status = bt_cpu_ref_register();
  if (status != BT_OK) return status;
#ifdef BT_ENABLE_VULKAN
  // A machine without a usable Vulkan device still gets a working cpu_ref.
  status = bt_vk_register();
  if (status != BT_OK) bt_clear_error();
#endif
  return BT_OK;
}

bt_status_t bt_shutdown(void) {
  bt_clear_error();
#ifdef BT_ENABLE_VULKAN
  bt_vk_shutdown();
#endif
  bt_dispatcher_reset();
  return BT_OK;
}

const char* bt_version(void) { return BT_VERSION_STRING; }

int bt_has_vulkan(void) {
#ifdef BT_ENABLE_VULKAN
  return bt_vk_available();
#else
  return 0;
#endif
}
