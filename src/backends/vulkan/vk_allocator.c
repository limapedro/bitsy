#include "vk_internal.h"

#ifdef _WIN32
#include <windows.h>
#endif
#include <stdlib.h>
#include <string.h>

#include "../../core/error_internal.h"
#include <bitsy/bt_profiler.h>
#include "vk_device_internal.h"

/* exp_001: reuse device-local buffers by power-of-two size class. This removes
 * vkAllocateMemory from the per-op hot path; a finer suballocator is M10. */
typedef struct vk_buffer_t {
  VkBuffer buffer;
  VkDeviceMemory memory;
  VkDeviceSize size;
  struct vk_buffer_t* next;
} vk_buffer_t;

#define CACHE_BUCKETS 24
static vk_buffer_t* g_free[CACHE_BUCKETS];
#define MAX_DEFERRED 64
static void* g_deferred[MAX_DEFERRED];
static int g_deferred_count;

void bt_vk_defer_free(void* handle) {
  if (!handle) return;
  if (g_deferred_count < MAX_DEFERRED) g_deferred[g_deferred_count++] = handle;
  else bt_vk_free(handle);
}
void bt_vk_deferred_release(void) {
  for (int i = 0; i < g_deferred_count; ++i) bt_vk_free(g_deferred[i]);
  g_deferred_count = 0;
}

static int bucket_of(size_t size) {
  int bucket = 0;
  while (size > 1 && bucket < CACHE_BUCKETS - 1) { size >>= 1; ++bucket; }
  return bucket;
}

static uint32_t find_memory_type(VkMemoryPropertyFlags wanted) {
  VkPhysicalDeviceMemoryProperties props;
  vkGetPhysicalDeviceMemoryProperties(g_vk.phys, &props);
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
    if ((props.memoryTypes[i].propertyFlags & wanted) == wanted) return i;
  return UINT32_MAX;
}

static bt_status_t make_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                               VkMemoryPropertyFlags props, vk_buffer_t* out) {
  VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                            .size = size, .usage = usage,
                            .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
  VKCHECK(vkCreateBuffer(g_vk.device, &bi, NULL, &out->buffer));
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(g_vk.device, out->buffer, &req);
  uint32_t type = find_memory_type(props);
  if (type == UINT32_MAX) return bt_fail(BT_ERR_BACKEND, "vulkan: no suitable memory type");
  VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                              .allocationSize = req.size, .memoryTypeIndex = type };
  VkResult result = vkAllocateMemory(g_vk.device, &ai, NULL, &out->memory);
  if (result != VK_SUCCESS) { vkDestroyBuffer(g_vk.device, out->buffer, NULL); return bt_fail(BT_ERR_BACKEND, "vulkan vkAllocateMemory failed (%d)", (int)result); }
  VKCHECK(vkBindBufferMemory(g_vk.device, out->buffer, out->memory, 0));
  out->size = size;
  return BT_OK;
}

static void destroy_buffer(vk_buffer_t* b) {
  if (!b) return;
  vkDestroyBuffer(g_vk.device, b->buffer, NULL);
  vkFreeMemory(g_vk.device, b->memory, NULL);
}

VkBuffer bt_vk_buffer_of(void* handle) { return handle ? ((vk_buffer_t*)handle)->buffer : VK_NULL_HANDLE; }

void bt_vk_begin_cmd(void) {
  if (g_vk.recording) return; // keep recording across dispatches until a flush
  VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                  .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
  (void)vkBeginCommandBuffer(g_vk.cmd, &bi);
  g_vk.recording = 1;
}

/* Conservative global barrier: transfers and shaders must see each other's writes. */
void bt_vk_barrier(void) {
  VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                         .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                         .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };
  vkCmdPipelineBarrier(g_vk.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
  bt_profiler_barrier();
}

static double alloc_now_ms(void) {
#ifdef _WIN32
  static LARGE_INTEGER freq;
  if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
  LARGE_INTEGER t; QueryPerformanceCounter(&t);
  return (double)t.QuadPart * 1e3 / (double)freq.QuadPart;
#else
  return 0.0;
#endif
}

void* bt_vk_alloc(size_t nbytes) {
  double t0 = alloc_now_ms();
  size_t size = 1;
  while (size < nbytes) size <<= 1; // size classes make buffers reusable
  int bucket = bucket_of(size);
  vk_buffer_t* b = g_free[bucket];
  if (b) {
    g_free[bucket] = b->next;
    b->next = NULL;
    bt_profiler_alloc(1, alloc_now_ms() - t0);
    return b;
  }
  b = calloc(1, sizeof(*b));
  if (!b) { bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_vk_alloc: OOM"); return NULL; }
  VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (make_buffer(size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b) != BT_OK) {
    free(b);
    return NULL;
  }
  bt_profiler_alloc(0, alloc_now_ms() - t0);
  return b;
}

void bt_vk_free(void* handle) {
  vk_buffer_t* b = handle;
  if (!b) return;
  b->next = g_free[bucket_of((size_t)b->size)];
  g_free[bucket_of((size_t)b->size)] = b;
}

/* exp_014: persistent host-visible staging pool for deferred transfers.
 * A staging slot moves used -> submitted (on flush) -> pooled (on wait), so it
 * is never reused while the GPU may still be reading/writing its copy. */
#define STAGE_MAX 64
static vk_buffer_t* g_stage_pool[STAGE_MAX]; static int g_stage_pool_n;
static vk_buffer_t* g_stage_used[STAGE_MAX]; static int g_stage_used_n;
static vk_buffer_t* g_stage_submitted[STAGE_MAX]; static int g_stage_submitted_n;
static unsigned long g_stage_allocs;

static void stage_push(vk_buffer_t* dst[], int* dn, vk_buffer_t* b) {
  if (*dn < STAGE_MAX) dst[(*dn)++] = b;
  else { destroy_buffer(b); free(b); }
}

void bt_vk_stage_on_flush(void) {
  for (int i = 0; i < g_stage_used_n; ++i)
    stage_push(g_stage_submitted, &g_stage_submitted_n, g_stage_used[i]);
  g_stage_used_n = 0;
}

void bt_vk_stage_on_wait(void) {
  for (int i = 0; i < g_stage_submitted_n; ++i)
    stage_push(g_stage_pool, &g_stage_pool_n, g_stage_submitted[i]);
  g_stage_submitted_n = 0;
}

unsigned long bt_vk_stage_alloc_count(void) { return g_stage_allocs; }

static vk_buffer_t* stage_acquire(size_t nbytes) {
  for (int i = 0; i < g_stage_pool_n; ++i) {
    if (g_stage_pool[i]->size >= nbytes) {
      vk_buffer_t* b = g_stage_pool[i];
      g_stage_pool[i] = g_stage_pool[--g_stage_pool_n]; // swap-remove
      return b;
    }
  }
  vk_buffer_t* b = calloc(1, sizeof(*b));
  if (!b) return NULL;
  size_t size = 1; while (size < nbytes) size <<= 1;
  if (make_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, b) != BT_OK) {
    free(b); return NULL;
  }
  ++g_stage_allocs;
  return b;
}

static int deferred_transfers(void) {
  static int val = -1;
  if (val < 0) {
    const char* env = getenv("BITSY_DEFERRED_UPLOAD");
    val = env && !atoi(env) ? 0 : 1; // exp_014: deferred is the default; BITSY_DEFERRED_UPLOAD=0 restores the old sync path
  }
  return val;
}

void bt_vk_allocator_shutdown(void) {
  for (int i = 0; i < CACHE_BUCKETS; ++i) {
    for (vk_buffer_t* b = g_free[i]; b;) {
      vk_buffer_t* next = b->next;
      destroy_buffer(b);
      free(b);
      b = next;
    }
    g_free[i] = NULL;
  }
  vk_buffer_t** pools[3] = { g_stage_pool, g_stage_used, g_stage_submitted };
  int counts[3] = { g_stage_pool_n, g_stage_used_n, g_stage_submitted_n };
  for (int p = 0; p < 3; ++p)
    for (int i = 0; i < counts[p]; ++i) { destroy_buffer(pools[p][i]); free(pools[p][i]); }
  g_stage_pool_n = g_stage_used_n = g_stage_submitted_n = 0;
}

bt_status_t bt_vk_upload(void* handle, const void* host, size_t nbytes) {
  if (!handle || !nbytes) return BT_OK;
  vk_buffer_t* dst = handle;
  if (!deferred_transfers()) {
    /* Old path (A/B baseline): fresh staging buffer + synchronous flush/wait. */
    vk_buffer_t stage = { 0 };
    bt_status_t status = make_buffer(nbytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stage);
    if (status != BT_OK) return status;
    void* mapped = NULL;
    VKCHECK(vkMapMemory(g_vk.device, stage.memory, 0, nbytes, 0, &mapped));
    memcpy(mapped, host, nbytes);
    vkUnmapMemory(g_vk.device, stage.memory);
    bt_vk_begin_cmd();
    VkBufferCopy copy = { .srcOffset = 0, .dstOffset = 0, .size = nbytes };
    vkCmdCopyBuffer(g_vk.cmd, stage.buffer, dst->buffer, 1, &copy);
    bt_vk_barrier();
    status = bt_vk_flush();
    if (status == BT_OK) status = bt_vk_wait(); // staging dies after this
    destroy_buffer(&stage);
    return status;
  }
  /* Deferred path: pooled staging, copy enqueued into the current command
   * buffer; the only host wait is the existing completion point. */
  if (g_stage_used_n >= STAGE_MAX) { bt_vk_flush(); bt_vk_wait(); } // safety: avoid in-use reuse
  vk_buffer_t* stage = stage_acquire(nbytes);
  if (!stage) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_vk_upload: staging OOM");
  void* mapped = NULL;
  VKCHECK(vkMapMemory(g_vk.device, stage->memory, 0, nbytes, 0, &mapped));
  memcpy(mapped, host, nbytes);
  vkUnmapMemory(g_vk.device, stage->memory);
  bt_vk_begin_cmd();
  VkBufferCopy copy = { .srcOffset = 0, .dstOffset = 0, .size = nbytes };
  vkCmdCopyBuffer(g_vk.cmd, stage->buffer, dst->buffer, 1, &copy);
  bt_vk_barrier();
  stage_push(g_stage_used, &g_stage_used_n, stage);
  return BT_OK;
}

bt_status_t bt_vk_readback(void* handle, void* host, size_t nbytes) {
  if (!handle || !nbytes) return BT_OK;
  vk_buffer_t* src = handle;
  if (!deferred_transfers()) {
    vk_buffer_t stage = { 0 };
    bt_status_t status = make_buffer(nbytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stage);
    if (status != BT_OK) return status;
    bt_vk_begin_cmd();
    VkBufferCopy copy = { .srcOffset = 0, .dstOffset = 0, .size = nbytes };
    vkCmdCopyBuffer(g_vk.cmd, src->buffer, stage.buffer, 1, &copy);
    bt_vk_barrier();
    status = bt_vk_flush();
    if (status == BT_OK) status = bt_vk_wait(); // host maps the staging buffer
    if (status != BT_OK) { destroy_buffer(&stage); return status; }
    void* mapped = NULL;
    VKCHECK(vkMapMemory(g_vk.device, stage.memory, 0, nbytes, 0, &mapped));
    memcpy(host, mapped, nbytes);
    vkUnmapMemory(g_vk.device, stage.memory);
    destroy_buffer(&stage);
    return BT_OK;
  }
  /* Deferred path: pooled staging. The host wait is inherent to readback, but
   * the staging allocation is amortised instead of paid every call. */
  vk_buffer_t* stage = stage_acquire(nbytes);
  if (!stage) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_vk_readback: staging OOM");
  bt_vk_begin_cmd();
  VkBufferCopy copy = { .srcOffset = 0, .dstOffset = 0, .size = nbytes };
  vkCmdCopyBuffer(g_vk.cmd, src->buffer, stage->buffer, 1, &copy);
  bt_vk_barrier();
  bt_status_t status = bt_vk_flush();
  if (status == BT_OK) status = bt_vk_wait();
  if (status != BT_OK) { stage_push(g_stage_pool, &g_stage_pool_n, stage); return status; }
  void* mapped = NULL;
  VKCHECK(vkMapMemory(g_vk.device, stage->memory, 0, nbytes, 0, &mapped));
  memcpy(host, mapped, nbytes);
  vkUnmapMemory(g_vk.device, stage->memory);
  stage_push(g_stage_pool, &g_stage_pool_n, stage);
  return BT_OK;
}
