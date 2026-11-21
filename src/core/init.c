#include <bitsy/bitsy.h>

#include "dispatcher_internal.h"
#include "error_internal.h"

/* Implemented by src/backends/cpu_ref/register.c. */
bt_status_t bt_cpu_ref_register(void);

/* *** process state *** */

bt_status_t bt_init(void) {
  bt_clear_error();
  bt_dispatcher_reset();
  return bt_cpu_ref_register();
}

bt_status_t bt_shutdown(void) {
  bt_clear_error();
  bt_dispatcher_reset();
  return BT_OK;
}

const char* bt_version(void) { return BT_VERSION_STRING; }
