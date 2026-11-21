#include "vk_internal.h"

#include <stdlib.h>
#include <string.h>

#include "../../core/error_internal.h"
#include "vk_device_internal.h"
#include "elementwise.spv.h"

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

static bt_status_t make_pipeline(void) {
  VkDescriptorSetLayoutBinding bindings[3];
  for (int i = 0; i < 3; ++i) {
    bindings[i] = (VkDescriptorSetLayoutBinding){
      .binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };
  }
  VkDescriptorSetLayoutCreateInfo dli = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .bindingCount = 3, .pBindings = bindings };
  VKCHECK(vkCreateDescriptorSetLayout(g_vk.device, &dli, NULL, &g_vk.set_layout));

  VkDescriptorPoolSize size = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3 };
  VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                     .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &size };
  VKCHECK(vkCreateDescriptorPool(g_vk.device, &dpi, NULL, &g_vk.descriptor_pool));
  VkDescriptorSetAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                     .descriptorPool = g_vk.descriptor_pool,
                                     .descriptorSetCount = 1, .pSetLayouts = &g_vk.set_layout };
  VKCHECK(vkAllocateDescriptorSets(g_vk.device, &ai, &g_vk.set));

  // 3 scalars + 4 int[8] arrays + 3 offsets = 152 bytes (matches vk_pc_t).
  VkPushConstantRange pc = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 152 };
  VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .setLayoutCount = 1, .pSetLayouts = &g_vk.set_layout,
                                     .pushConstantRangeCount = 1, .pPushConstantRanges = &pc };
  VKCHECK(vkCreatePipelineLayout(g_vk.device, &pli, NULL, &g_vk.pipeline_layout));

  VkShaderModuleCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = elementwise_spv_bytes,
                                   .pCode = elementwise_spv };
  VkShaderModule module;
  VKCHECK(vkCreateShaderModule(g_vk.device, &sci, NULL, &module));

  VkPipelineShaderStageCreateInfo stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                            .module = module, .pName = "main" };
  VkComputePipelineCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = stage, .layout = g_vk.pipeline_layout };
  VKCHECK(vkCreateComputePipelines(g_vk.device, VK_NULL_HANDLE, 1, &cpi, NULL, &g_vk.pipeline));
  vkDestroyShaderModule(g_vk.device, module, NULL); // the pipeline owns the code
  return BT_OK;
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
  VkPhysicalDeviceFeatures supported;
  vkGetPhysicalDeviceFeatures(g_vk.phys, &supported);
  VkPhysicalDeviceFeatures features = { 0 };
  features.shaderFloat64 = supported.shaderFloat64;
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

  status = make_pipeline();
  if (status != BT_OK) return status;
  g_vk.ready = 1;
  return BT_OK;
}

void bt_vk_shutdown(void) {
  if (!g_vk.ready) return;
  bt_vk_allocator_shutdown();
  vkDeviceWaitIdle(g_vk.device);
  vkDestroyPipeline(g_vk.device, g_vk.pipeline, NULL);
  vkDestroyPipelineLayout(g_vk.device, g_vk.pipeline_layout, NULL);
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

/* Single in-flight command buffer: simple, correct, and TDR-safe for small ops. */
bt_status_t bt_vk_flush(void) {
  VKCHECK(vkEndCommandBuffer(g_vk.cmd));
  VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .commandBufferCount = 1, .pCommandBuffers = &g_vk.cmd };
  VKCHECK(vkResetFences(g_vk.device, 1, &g_vk.fence));
  VKCHECK(vkQueueSubmit(g_vk.queue, 1, &submit, g_vk.fence));
  VKCHECK(vkWaitForFences(g_vk.device, 1, &g_vk.fence, VK_TRUE, UINT64_MAX));
  return BT_OK;
}
