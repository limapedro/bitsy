#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L  /* clock_gettime / CLOCK_MONOTONIC on Linux */
#endif
#include <bitsy/bt_profiler.h>

#include <stdio.h>
#include <string.h>

#include "error_internal.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#define PROF_MAX_EVENTS 2048

/* Calibrated ceilings from perf/HARDWARE_LEDGER.md (exp_006). */
#define PROF_DRAM_GBS 304.4
#define PROF_FP32_GFLOPS 5490.0

typedef struct {
  char name[48];
  double start_us;
  double dur_us;
  double gpu_ms;
  double bytes;
  double flops;
  int query_index;
  int valid;
} prof_event_t;

static prof_event_t g_events[PROF_MAX_EVENTS];
static int g_count;
static int g_enabled;
static bt_profile_summary_t g_summary;
static double g_origin_us;

static double now_us(void) {
#ifdef _WIN32
  static LARGE_INTEGER freq;
  if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
  LARGE_INTEGER t; QueryPerformanceCounter(&t);
  return (double)t.QuadPart * 1e6 / (double)freq.QuadPart;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
#endif
}

void bt_profiler_enable(int enabled) {
  g_enabled = enabled ? 1 : 0;
  if (g_enabled && g_origin_us == 0.0) g_origin_us = now_us();
}

int bt_profiler_enabled(void) { return g_enabled; }

void bt_profiler_reset(void) {
  memset(g_events, 0, sizeof(g_events));
  memset(&g_summary, 0, sizeof(g_summary));
  g_count = 0;
  g_origin_us = now_us();
}

int bt_profiler_record(const char* name, double cpu_start_us, double cpu_end_us,
                       double bytes, double flops, int query_index) {
  (void)query_index;
  if (!g_enabled || g_count >= PROF_MAX_EVENTS) return -1;
  prof_event_t* e = &g_events[g_count++];
  snprintf(e->name, sizeof(e->name), "%s", name ? name : "dispatch");
  e->start_us = cpu_start_us - g_origin_us;
  e->dur_us = cpu_end_us - cpu_start_us;
  e->bytes = bytes;
  e->flops = flops;
  e->query_index = query_index;
  e->valid = 1;
  g_summary.dispatches++;
  g_summary.cpu_enqueue_ms += e->dur_us / 1e3;
  g_summary.bytes += bytes;
  g_summary.flops += flops;
  return g_count - 1;
}

void bt_profiler_set_gpu_ms(int event_index, double ms) {
  if (!g_enabled || event_index < 0 || event_index >= g_count) return;
  g_events[event_index].gpu_ms = ms;
  g_summary.gpu_ms += ms;
}

void bt_profiler_barrier(void) { if (g_enabled) g_summary.barriers++; }
void bt_profiler_submit(void) { if (g_enabled) g_summary.submits++; }

void bt_profiler_alloc(int hit, double cpu_ms) {
  if (!g_enabled) return;
  if (hit) g_summary.allocation_hits++;
  else g_summary.allocations++;
  g_summary.alloc_ms += cpu_ms;
}

bt_profile_summary_t bt_profiler_summary(void) { return g_summary; }

bt_status_t bt_profiler_dump_json(const char* path) {
  if (!path) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_profiler_dump_json: NULL path");
  FILE* f = fopen(path, "w");
  if (!f) return bt_fail(BT_ERR_INVALID_ARGUMENT, "bt_profiler_dump_json: cannot open file");

  fprintf(f, "{\n  \"traceEvents\": [\n");
  for (int i = 0; i < g_count; ++i) {
    prof_event_t* e = &g_events[i];
    fprintf(f, "    {\"name\": \"%s\", \"cat\": \"dispatch\", \"ph\": \"X\", "
               "\"ts\": %.3f, \"dur\": %.3f, \"pid\": 1, \"tid\": 1, "
               "\"args\": {\"gpu_ms\": %.6f, \"bytes\": %.1f, \"flops\": %.1f}}%s\n",
            e->name, e->start_us, e->dur_us, e->gpu_ms, e->bytes, e->flops,
            i + 1 == g_count ? "" : ",");
  }
  double bw = g_summary.gpu_ms > 0
    ? g_summary.bytes / (g_summary.gpu_ms * 1e-3) / 1e9 : 0.0;
  double gf = g_summary.gpu_ms > 0
    ? g_summary.flops / (g_summary.gpu_ms * 1e-3) / 1e9 : 0.0;
  fprintf(f, "  ],\n  \"summary\": {\n"
               "    \"dispatches\": %llu, \"barriers\": %llu, \"submits\": %llu,\n"
               "    \"allocations\": %llu, \"allocation_hits\": %llu,\n"
               "    \"gpu_ms\": %.6f, \"cpu_enqueue_ms\": %.6f, \"alloc_ms\": %.6f,\n"
               "    \"bytes\": %.1f, \"flops\": %.1f,\n"
               "    \"achieved_GBs\": %.3f, \"pct_dram_ceiling\": %.3f,\n"
               "    \"achieved_GFLOPs\": %.3f, \"pct_fp32_ceiling\": %.3f\n"
               "  }\n}\n",
          (unsigned long long)g_summary.dispatches,
          (unsigned long long)g_summary.barriers,
          (unsigned long long)g_summary.submits,
          (unsigned long long)g_summary.allocations,
          (unsigned long long)g_summary.allocation_hits,
          g_summary.gpu_ms, g_summary.cpu_enqueue_ms, g_summary.alloc_ms,
          g_summary.bytes, g_summary.flops,
          bw, 100.0 * bw / PROF_DRAM_GBS,
          gf, 100.0 * gf / PROF_FP32_GFLOPS);
  fclose(f);
  return BT_OK;
}
