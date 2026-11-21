#ifndef BITSY_BT_PROFILER_H
#define BITSY_BT_PROFILER_H

#include <stdint.h>
#include <bitsy/bt_export.h>
#include <bitsy/bt_error.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opt-in instrumentation. Off by default; no cost on the normal path. */
typedef struct bt_profile_summary_t {
  uint64_t dispatches;
  uint64_t barriers;
  uint64_t submits;
  uint64_t allocations;    /* successful vkAllocateMemory calls */
  uint64_t allocation_hits;/* served from the size-class cache */
  double gpu_ms;           /* sum of timestamped dispatch times */
  double cpu_enqueue_ms;   /* CPU time spent recording dispatches */
  double alloc_ms;         /* CPU time spent allocating */
  double bytes;            /* estimated bytes moved by recorded ops */
  double flops;            /* estimated FLOPs of recorded ops */
} bt_profile_summary_t;

BT_API void bt_profiler_enable(int enabled);
BT_API void bt_profiler_reset(void);
BT_API int bt_profiler_enabled(void);

/* Called by the Vulkan backend; `query_index` maps to timestamp pair slots. */
int bt_profiler_record(const char* name, double cpu_start_us, double cpu_end_us,
                       double bytes, double flops, int query_index);
void bt_profiler_set_gpu_ms(int event_index, double ms);
void bt_profiler_barrier(void);
void bt_profiler_submit(void);
void bt_profiler_alloc(int hit, double cpu_ms);

BT_API bt_profile_summary_t bt_profiler_summary(void);

/* Chrome-trace JSON: one X event per dispatch plus a "summary" metadata event
 * containing counters and percentages of the calibrated ceilings. */
BT_API bt_status_t bt_profiler_dump_json(const char* path);

#ifdef __cplusplus
}
#endif

#endif /* BITSY_BT_PROFILER_H */
