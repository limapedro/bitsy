#ifndef BITSY_SRC_BACKENDS_VULKAN_VK_DEVICE_INTERNAL_H
#define BITSY_SRC_BACKENDS_VULKAN_VK_DEVICE_INTERNAL_H

#include <vulkan/vulkan.h>

#include <bitsy/bt_error.h>

/* Maximum dispatches per batched submission. One descriptor set per dispatch
 * is required because set contents are written on the host, not captured by
 * vkCmdBindDescriptorSets. */
#define BT_VK_MAX_DISPATCHES 1024
/* exp_015: a captured step is split into command-buffer chunks at host
 * readbacks (the interim autograd reduces bias gradients on the CPU). */
#define BT_VK_CAPTURE_MAX_CHUNKS 64

typedef struct {
  VkInstance instance;
  VkPhysicalDevice phys;
  VkDevice device;
  VkQueue queue;
  uint32_t qfam;
  VkCommandPool pool;
  VkCommandBuffer cmd;
  VkCommandBuffer main_cmd;
  VkCommandBuffer replay_cmd;
  VkCommandBuffer capture_chunks[BT_VK_CAPTURE_MAX_CHUNKS];
  int capture_chunk_count;
  int capturing;
  VkFence fence;
  VkSemaphore timeline;
  uint64_t timeline_value;
  int query_read_index;
  VkDescriptorSetLayout set_layout;
  VkDescriptorPool descriptor_pool;
  VkDescriptorSet sets[BT_VK_MAX_DISPATCHES];
  int set_index;
  int recording;
  VkPipelineLayout pipeline_layout;
  VkPipeline pipeline;
  VkPipeline elementwise_vec4_pipeline;
  VkPipeline matmul_pipeline;
  VkPipeline conv_pipeline;
  VkPipeline gemm_tiled_pipeline;
  VkPipeline gemm_rb_pipeline;
  VkPipeline gemm_rb_vec4_pipeline;
  VkPipeline gemm_rb_vec4_k16_pipeline;
  VkPipeline splitk_reduce_pipeline;
  VkPipeline fused_pipeline;
  VkPipelineCache pipeline_cache;
  VkQueryPool timestamp_pool;
  VkQueryPool profiler_pool;
  uint32_t profiler_query_count;
  int profiler_events[2048];
  int query_index;
  int have_timestamps;
  double device_time_ms;
  VkPhysicalDeviceProperties props;
  int ready;
} vk_ctx_t;

extern vk_ctx_t g_vk;

/* Ensure the command buffer is recording; no submit happens here. */
void bt_vk_begin_cmd(void);

/* Submit recorded work and signal the timeline; do not wait. */
bt_status_t bt_vk_flush(void);

/* Wait for the last submitted timeline value and collect timestamp results. */
bt_status_t bt_vk_wait(void);

/* Next descriptor set for one dispatch; flushes when the batch is full. */
VkDescriptorSet bt_vk_next_set(void);

/* Timestamp around one dispatch (no-op if the device lacks the feature). */
void bt_vk_timestamp_start(void);
void bt_vk_timestamp_end(void);

double bt_vk_device_time_ms(void);
void bt_vk_barrier(void);
VkBuffer bt_vk_buffer_of(void* handle);
void bt_vk_allocator_shutdown(void);
void bt_vk_defer_free(void* handle);        /* free after the next flush waits */
void bt_vk_deferred_release(void);
void bt_vk_hazard_reset(void);

#define VKCHECK(expr) do { \
  VkResult _r = (expr); \
  if (_r != VK_SUCCESS) return bt_fail(BT_ERR_BACKEND, "vulkan %s failed (%d)", #expr, (int)_r); \
} while (0)

#endif /* BITSY_SRC_BACKENDS_VULKAN_VK_DEVICE_INTERNAL_H */
