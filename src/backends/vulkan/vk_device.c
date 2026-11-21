#include "vk_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../core/error_internal.h"
#include "vk_device_internal.h"
#include "elementwise.spv.h"
#include "matmul.spv.h"
#include "conv.spv.h"
#include "gemm_tiled.spv.h"

vk_ctx_t g_vk;

/* Pick a discrete GPU when there is one; otherwise the first compute device. */
static bt_status_t pick_physical(void) {
  uint32_t count = 0;
  VKCHECK(vkEnumeratePhysicalDevices(g_vk.instance, &count, NULL));
  if (!count) return bt_fail(BT_ERR_BACKEND, "vulkan: no physical device found");
  VkPhysicalDevice* devs = malloc(count * sizeof(VkPhysicalDevice));
  if (!devs) return bt_fail(BT_ERR_OUT_OF_MEMORY, "vulkan: OOM enumerating devices");
  VKCHECK(vkEnumeratePhysicalDevices(g_vk.instance, &count, devs));

  g_vk.phys = NULL;
  for (uint32_t i = 0; i < count; ++i) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(devs[i], &props);
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, NULL);
    VkQueueFamilyProperties* families = malloc(nq * sizeof(VkQueueFamilyProperties));
    if (!families) { free(devs); return bt_fail(BT_ERR_OUT_OF_MEMORY, "vulkan: OOM queue families"); }
    vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, families);
    for (uint32_t q = 0; q < nq; ++q) {
      VkQueueFamilyProperties qp = families[q];
      if (!(qp.queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
      if (!g_vk.phys || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        g_vk.phys = devs[i];
        g_vk.qfam = q;
        g_vk.props = props;
      }
    }
    free(families);
  }
  free(devs);
  return g_vk.phys ? BT_OK : bt_fail(BT_ERR_BACKEND, "vulkan: device has no compute queue");
}

static bt_status_t pipeline_from_spv(const uint32_t* code, size_t bytes, VkPipeline* pipeline) {
  VkShaderModuleCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = bytes, .pCode = code };
  VkShaderModule module;
  VKCHECK(vkCreateShaderModule(g_vk.device, &sci, NULL, &module));
  VkPipelineShaderStageCreateInfo stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                            .module = module, .pName = "main" };
  VkComputePipelineCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = stage, .layout = g_vk.pipeline_layout };
  VkResult result = vkCreateComputePipelines(g_vk.device, VK_NULL_HANDLE, 1, &cpi, NULL, pipeline);
  vkDestroyShaderModule(g_vk.device, module, NULL); // the pipeline owns the code
  return result == VK_SUCCESS ? BT_OK : bt_fail(BT_ERR_BACKEND, "vkCreateComputePipelines failed (%d)", (int)result);
}

static bt_status_t make_pipeline(void) {
  VkDescriptorSetLayoutBinding bindings[4];
  for (int i = 0; i < 4; ++i) {
    bindings[i] = (VkDescriptorSetLayoutBinding){
      .binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };
  }
  VkDescriptorSetLayoutCreateInfo dli = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .bindingCount = 4, .pBindings = bindings };
  VKCHECK(vkCreateDescriptorSetLayout(g_vk.device, &dli, NULL, &g_vk.set_layout));

  VkDescriptorPoolSize size = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                .descriptorCount = 4 * BT_VK_MAX_DISPATCHES };
  VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                     .maxSets = BT_VK_MAX_DISPATCHES,
                                     .poolSizeCount = 1, .pPoolSizes = &size };
  VKCHECK(vkCreateDescriptorPool(g_vk.device, &dpi, NULL, &g_vk.descriptor_pool));
  VkDescriptorSetLayout layouts[BT_VK_MAX_DISPATCHES];
  for (int i = 0; i < BT_VK_MAX_DISPATCHES; ++i) layouts[i] = g_vk.set_layout;
  VkDescriptorSetAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                     .descriptorPool = g_vk.descriptor_pool,
                                     .descriptorSetCount = BT_VK_MAX_DISPATCHES,
                                     .pSetLayouts = layouts };
  VKCHECK(vkAllocateDescriptorSets(g_vk.device, &ai, g_vk.sets));

  // 3 scalars + 4 int[8] arrays + 3 offsets = 152 bytes (matches vk_pc_t).
  VkPushConstantRange pc = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 152 };
  VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .setLayoutCount = 1, .pSetLayouts = &g_vk.set_layout,
                                     .pushConstantRangeCount = 1, .pPushConstantRanges = &pc };
  VKCHECK(vkCreatePipelineLayout(g_vk.device, &pli, NULL, &g_vk.pipeline_layout));

  bt_status_t status;
  if ((status = pipeline_from_spv(elementwise_spv, elementwise_spv_bytes, &g_vk.pipeline))) return status;
  if ((status = pipeline_from_spv(matmul_spv, matmul_spv_bytes, &g_vk.matmul_pipeline))) return status;
  if ((status = pipeline_from_spv(conv_spv, conv_spv_bytes, &g_vk.conv_pipeline))) return status;
  return pipeline_from_spv(gemm_tiled_spv, gemm_tiled_spv_bytes, &g_vk.gemm_tiled_pipeline);
}

bt_status_t bt_vk_init(void) {
  if (g_vk.ready) return BT_OK;
  bt_clear_error();
  VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "bitsy", .apiVersion = VK_API_VERSION_1_2 };
  VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
  VKCHECK(vkCreateInstance(&ici, NULL, &g_vk.instance));
  bt_status_t status = pick_physical();
  if (status != BT_OK) return status;

  float priority = 1.f;
  VkDeviceQueueCreateInfo qi = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                 .queueFamilyIndex = g_vk.qfam, .queueCount = 1, .pQueuePriorities = &priority };
  // Optional feature, queried not assumed: fp64 division keeps fp32 div exact.
  // Timestamp availability is a device limit, not a feature flag.
  VkPhysicalDeviceFeatures supported;
  vkGetPhysicalDeviceFeatures(g_vk.phys, &supported);
  VkPhysicalDeviceFeatures features = { 0 };
  features.shaderFloat64 = supported.shaderFloat64;
  g_vk.have_timestamps = g_vk.props.limits.timestampComputeAndGraphics ? 1 : 0;
  VkDeviceCreateInfo di = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                            .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
                            .pEnabledFeatures = &features };
  VKCHECK(vkCreateDevice(g_vk.phys, &di, NULL, &g_vk.device));
  vkGetDeviceQueue(g_vk.device, g_vk.qfam, 0, &g_vk.queue);

  VkCommandPoolCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                  .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                  .queueFamilyIndex = g_vk.qfam };
  VKCHECK(vkCreateCommandPool(g_vk.device, &cpi, NULL, &g_vk.pool));
  VkCommandBufferAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                     .commandPool = g_vk.pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                     .commandBufferCount = 1 };
  VKCHECK(vkAllocateCommandBuffers(g_vk.device, &ai, &g_vk.cmd));
  VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
  VKCHECK(vkCreateFence(g_vk.device, &fi, NULL, &g_vk.fence));
  if (g_vk.have_timestamps) {
    VkQueryPoolCreateInfo qi2 = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                  .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                  .queryCount = 2 * BT_VK_MAX_DISPATCHES,
                                  .flags = VK_QUERY_POOL_CREATE_RESET_BIT_KHR };
    VKCHECK(vkCreateQueryPool(g_vk.device, &qi2, NULL, &g_vk.timestamp_pool));
  }

  status = make_pipeline();
  if (status != BT_OK) return status;
  g_vk.ready = 1;
  return BT_OK;
}

void bt_vk_shutdown(void) {
  if (!g_vk.ready) return;
  (void)bt_vk_flush();
  bt_vk_allocator_shutdown();
  vkDeviceWaitIdle(g_vk.device);
  vkDestroyPipeline(g_vk.device, g_vk.pipeline, NULL);
  vkDestroyPipeline(g_vk.device, g_vk.matmul_pipeline, NULL);
  vkDestroyPipeline(g_vk.device, g_vk.conv_pipeline, NULL);
  vkDestroyPipeline(g_vk.device, g_vk.gemm_tiled_pipeline, NULL);
  vkDestroyPipelineLayout(g_vk.device, g_vk.pipeline_layout, NULL);
  if (g_vk.have_timestamps) vkDestroyQueryPool(g_vk.device, g_vk.timestamp_pool, NULL);
  vkDestroyDescriptorPool(g_vk.device, g_vk.descriptor_pool, NULL);
  vkDestroyDescriptorSetLayout(g_vk.device, g_vk.set_layout, NULL);
  vkDestroyFence(g_vk.device, g_vk.fence, NULL);
  vkFreeCommandBuffers(g_vk.device, g_vk.pool, 1, &g_vk.cmd);
  vkDestroyCommandPool(g_vk.device, g_vk.pool, NULL);
  vkDestroyDevice(g_vk.device, NULL);
  vkDestroyInstance(g_vk.instance, NULL);
  memset(&g_vk, 0, sizeof(g_vk));
}

int bt_vk_available(void) { return g_vk.ready; }
const char* bt_vk_device_name(void) { return g_vk.ready ? g_vk.props.deviceName : "none"; }

/* Batched submission: record many dispatches, submit once, wait once. */
bt_status_t bt_vk_flush(void) {
  if (!g_vk.ready || !g_vk.recording) return BT_OK;

  VKCHECK(vkEndCommandBuffer(g_vk.cmd));
  VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .commandBufferCount = 1, .pCommandBuffers = &g_vk.cmd };
  VKCHECK(vkResetFences(g_vk.device, 1, &g_vk.fence));
  VKCHECK(vkQueueSubmit(g_vk.queue, 1, &submit, g_vk.fence));
  VKCHECK(vkWaitForFences(g_vk.device, 1, &g_vk.fence, VK_TRUE, UINT64_MAX));

  if (g_vk.have_timestamps && g_vk.query_index > 0) {
    uint64_t stamps[2 * BT_VK_MAX_DISPATCHES];
    VkResult result = vkGetQueryPoolResults(
      g_vk.device, g_vk.timestamp_pool, 0, (uint32_t)g_vk.query_index,
      sizeof(stamps), stamps, sizeof(uint64_t),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (result == VK_SUCCESS) {
      double ns = 0.0;
      for (int i = 0; i < g_vk.query_index; i += 2)
        if (stamps[i + 1] > stamps[i]) ns += (double)(stamps[i + 1] - stamps[i]);
      g_vk.device_time_ms = ns * g_vk.props.limits.timestampPeriod / 1e6;
    }
    vkResetQueryPool(g_vk.device, g_vk.timestamp_pool, 0, (uint32_t)g_vk.query_index);
  }

  g_vk.query_index = 0;
  g_vk.set_index = 0;
  g_vk.recording = 0;
  return BT_OK;
}

VkDescriptorSet bt_vk_next_set(void) {
  if (g_vk.set_index >= BT_VK_MAX_DISPATCHES) (void)bt_vk_flush();
  return g_vk.sets[g_vk.set_index++];
}

void bt_vk_timestamp_start(void) {
  if (!g_vk.have_timestamps || g_vk.query_index + 1 >= 2 * BT_VK_MAX_DISPATCHES) return;
  vkCmdWriteTimestamp(g_vk.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      g_vk.timestamp_pool, (uint32_t)g_vk.query_index++);
}

void bt_vk_timestamp_end(void) {
  if (!g_vk.have_timestamps || g_vk.query_index + 1 >= 2 * BT_VK_MAX_DISPATCHES) return;
  vkCmdWriteTimestamp(g_vk.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      g_vk.timestamp_pool, (uint32_t)g_vk.query_index++);
}

double bt_vk_device_time_ms(void) { return g_vk.device_time_ms; }
