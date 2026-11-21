#ifndef BITSY_SRC_BACKENDS_VULKAN_VK_DEVICE_INTERNAL_H
#define BITSY_SRC_BACKENDS_VULKAN_VK_DEVICE_INTERNAL_H

#include <vulkan/vulkan.h>

#include <bitsy/bt_error.h>

typedef struct {
  VkInstance instance;
  VkPhysicalDevice phys;
  VkDevice device;
  VkQueue queue;
  uint32_t qfam;
  VkCommandPool pool;
  VkCommandBuffer cmd;
  VkFence fence;         /* one fence: dispatches are serialized in M2 */
  VkDescriptorSetLayout set_layout;
  VkDescriptorPool descriptor_pool;
  VkDescriptorSet set;
  VkPipelineLayout pipeline_layout;
  VkPipeline pipeline;
  VkPhysicalDeviceProperties props;
  int ready;
} vk_ctx_t;

extern vk_ctx_t g_vk;

/* End, submit and wait on the single fence. Serialized but deterministic. */
bt_status_t bt_vk_flush(void);
void bt_vk_begin_cmd(void);
void bt_vk_barrier(void);
VkBuffer bt_vk_buffer_of(void* handle);
void bt_vk_allocator_shutdown(void);

#define VKCHECK(expr) do { \
  VkResult _r = (expr); \
  if (_r != VK_SUCCESS) return bt_fail(BT_ERR_BACKEND, "vulkan %s failed (%d)", #expr, (int)_r); \
} while (0)

#endif /* BITSY_SRC_BACKENDS_VULKAN_VK_DEVICE_INTERNAL_H */
