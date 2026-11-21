#include "vk_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../core/error_internal.h"
#include <bitsy/bt_profiler.h>
#include "vk_device_internal.h"
#include "elementwise.spv.h"
#include "matmul.spv.h"
#include "conv.spv.h"
#include "gemm_tiled.spv.h"
#include "gemm_rb.spv.h"
#include "gemm_rb_vec4.spv.h"
#include "gemm_rb_vec4_k16.spv.h"
#include "splitk_reduce.spv.h"
#include "fused.spv.h"

vk_ctx_t g_vk;
static const char *g_pipeline_cache_path = NULL;

/* Load an on-disk pipeline cache only when its device UUID matches. */
static void load_pipeline_cache(void) {
  g_pipeline_cache_path = getenv("BITSY_PIPELINE_CACHE");
  if (!g_pipeline_cache_path) g_pipeline_cache_path = "bitsy.pipeline_cache";
  void *data = NULL;
  size_t size = 0;
  FILE *f = fopen(g_pipeline_cache_path, "rb");
  if (f) {
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n > (long)sizeof(VkPipelineCacheHeaderVersionOne)) {
      data = malloc((size_t)n);
      if (data && fread(data, 1, (size_t)n, f) == (size_t)n) {
        VkPipelineCacheHeaderVersionOne *h = data;
        if (h->headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE ||
            h->vendorID != g_vk.props.vendorID || h->deviceID != g_vk.props.deviceID ||
            memcmp(h->pipelineCacheUUID, g_vk.props.pipelineCacheUUID, VK_UUID_SIZE) != 0) {
          free(data); data = NULL;
        } else {
          size = (size_t)n;
        }
      } else {
        free(data); data = NULL;
      }
    }
    fclose(f);
  }
  VkPipelineCacheCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
                                   .initialDataSize = size, .pInitialData = data };
  VkResult pr = vkCreatePipelineCache(g_vk.device, &ci, NULL, &g_vk.pipeline_cache);
  if (pr != VK_SUCCESS) bt_fail(BT_ERR_BACKEND, "vkCreatePipelineCache failed (%d)", (int)pr);
  free(data);
}


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
  VkResult result = vkCreateComputePipelines(g_vk.device, g_vk.pipeline_cache, 1, &cpi, NULL, pipeline);
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
  if ((status = pipeline_from_spv(gemm_tiled_spv, gemm_tiled_spv_bytes,
                                  &g_vk.gemm_tiled_pipeline))) return status;
  if ((status = pipeline_from_spv(fused_spv, fused_spv_bytes, &g_vk.fused_pipeline))) return status;
  if ((status = pipeline_from_spv(gemm_rb_spv, gemm_rb_spv_bytes, &g_vk.gemm_rb_pipeline))) return status;
  if ((status = pipeline_from_spv(gemm_rb_vec4_spv, gemm_rb_vec4_spv_bytes,
                                  &g_vk.gemm_rb_vec4_pipeline))) return status;
  if ((status = pipeline_from_spv(gemm_rb_vec4_k16_spv, gemm_rb_vec4_k16_spv_bytes,
                                  &g_vk.gemm_rb_vec4_k16_pipeline))) return status;
  return pipeline_from_spv(splitk_reduce_spv, splitk_reduce_spv_bytes,
                           &g_vk.splitk_reduce_pipeline);
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
  VkSemaphoreTypeCreateInfo sti = {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = 0 };
  VkSemaphoreCreateInfo sci2 = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                 .pNext = &sti };
  VKCHECK(vkCreateSemaphore(g_vk.device, &sci2, NULL, &g_vk.timeline));
  if (g_vk.have_timestamps) {
    VkQueryPoolCreateInfo qi2 = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                  .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                  .queryCount = 2 * BT_VK_MAX_DISPATCHES,
                                  .flags = VK_QUERY_POOL_CREATE_RESET_BIT_KHR };
    VKCHECK(vkCreateQueryPool(g_vk.device, &qi2, NULL, &g_vk.timestamp_pool));
    VkQueryPoolCreateInfo qi3 = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                  .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                  .queryCount = 2 * 2048,
                                  .flags = VK_QUERY_POOL_CREATE_RESET_BIT_KHR };
    VKCHECK(vkCreateQueryPool(g_vk.device, &qi3, NULL, &g_vk.profiler_pool));
  }

  load_pipeline_cache();
  status = make_pipeline();
  if (status != BT_OK) return status;
  g_vk.ready = 1;
  return BT_OK;
}

void bt_vk_shutdown(void) {
  if (!g_vk.ready) return;
  (void)bt_vk_flush();
  (void)bt_vk_wait();
  bt_vk_allocator_shutdown();
  vkDeviceWaitIdle(g_vk.device);
  vkDestroyPipeline(g_vk.device, g_vk.pipeline, NULL);
  vkDestroyPipeline(g_vk.device, g_vk.matmul_pipeline, NULL);
  vkDestroyPipeline(g_vk.device, g_vk.conv_pipeline, NULL);
  vkDestroyPipeline(g_vk.device, g_vk.gemm_tiled_pipeline, NULL);
  vkDestroyPipeline(g_vk.device, g_vk.fused_pipeline, NULL);
  vkDestroyPipeline(g_vk.device, g_vk.gemm_rb_pipeline, NULL);
  vkDestroyPipeline(g_vk.device, g_vk.gemm_rb_vec4_pipeline, NULL);
  vkDestroyPipeline(g_vk.device, g_vk.gemm_rb_vec4_k16_pipeline, NULL);
  vkDestroyPipeline(g_vk.device, g_vk.splitk_reduce_pipeline, NULL);
  vkDestroyPipelineLayout(g_vk.device, g_vk.pipeline_layout, NULL);
  vkDestroyPipelineCache(g_vk.device, g_vk.pipeline_cache, NULL);
  if (g_vk.have_timestamps) {
    vkDestroyQueryPool(g_vk.device, g_vk.timestamp_pool, NULL);
    vkDestroyQueryPool(g_vk.device, g_vk.profiler_pool, NULL);
  }
  vkDestroyDescriptorPool(g_vk.device, g_vk.descriptor_pool, NULL);
  vkDestroyDescriptorSetLayout(g_vk.device, g_vk.set_layout, NULL);
  vkDestroyFence(g_vk.device, g_vk.fence, NULL);
  vkDestroySemaphore(g_vk.device, g_vk.timeline, NULL);
  vkFreeCommandBuffers(g_vk.device, g_vk.pool, 1, &g_vk.cmd);
  vkDestroyCommandPool(g_vk.device, g_vk.pool, NULL);
  vkDestroyDevice(g_vk.device, NULL);
  vkDestroyInstance(g_vk.instance, NULL);
  memset(&g_vk, 0, sizeof(g_vk));
}

int bt_vk_available(void) { return g_vk.ready; }
const char* bt_vk_device_name(void) { return g_vk.ready ? g_vk.props.deviceName : "none"; }

/* Async submit: signal a timeline value, do not block the CPU. */
bt_status_t bt_vk_flush(void) {
  if (!g_vk.ready || !g_vk.recording) return BT_OK;

  VKCHECK(vkEndCommandBuffer(g_vk.cmd));
  uint64_t value = ++g_vk.timeline_value;
  VkTimelineSemaphoreSubmitInfo tsi = {
    .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
    .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &value };
  VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .pNext = &tsi, .commandBufferCount = 1,
                          .pCommandBuffers = &g_vk.cmd,
                          .signalSemaphoreCount = 1,
                          .pSignalSemaphores = &g_vk.timeline };
  VKCHECK(vkQueueSubmit(g_vk.queue, 1, &submit, VK_NULL_HANDLE));
  bt_profiler_submit();
  g_vk.recording = 0;
  return BT_OK;
}

/* Host completion point: wait for the last submitted value, collect timestamps,
 * and recycle descriptor sets / query slots. */
bt_status_t bt_vk_wait(void) {
  if (!g_vk.ready || g_vk.timeline_value == 0) return BT_OK;
  VkSemaphoreWaitInfo wait = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                               .semaphoreCount = 1, .pSemaphores = &g_vk.timeline,
                               .pValues = &g_vk.timeline_value };
  VKCHECK(vkWaitSemaphores(g_vk.device, &wait, UINT64_MAX));
  bt_vk_deferred_release(); // scratch buffers are safe once the GPU is idle
  bt_vk_hazard_reset();

  if (g_vk.have_timestamps && g_vk.query_index > g_vk.query_read_index) {
    int count = g_vk.query_index - g_vk.query_read_index;
    uint64_t stamps[2 * BT_VK_MAX_DISPATCHES];
    VkResult result = vkGetQueryPoolResults(
      g_vk.device, g_vk.timestamp_pool, (uint32_t)g_vk.query_read_index,
      (uint32_t)count, sizeof(stamps), stamps, sizeof(uint64_t),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (result == VK_SUCCESS) {
      double ns = 0.0;
      for (int i = 0; i < count; i += 2)
        if (stamps[i + 1] > stamps[i]) ns += (double)(stamps[i + 1] - stamps[i]);
      g_vk.device_time_ms = ns * g_vk.props.limits.timestampPeriod / 1e6;
    }
    vkResetQueryPool(g_vk.device, g_vk.timestamp_pool,
                     (uint32_t)g_vk.query_read_index, (uint32_t)count);
  }
  if (bt_profiler_enabled() && g_vk.have_timestamps && g_vk.profiler_query_count > 0) {
    uint32_t n = g_vk.profiler_query_count;
    if (n > 2048) n = 2048;
    uint64_t stamps[4096];
    VkResult pr = vkGetQueryPoolResults(g_vk.device, g_vk.profiler_pool, 0, 2 * n,
                                        sizeof(stamps), stamps, sizeof(uint64_t),
                                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (pr == VK_SUCCESS)
      for (uint32_t i = 0; i < n; ++i)
        bt_profiler_set_gpu_ms(g_vk.profiler_events[i],
          (double)(stamps[2 * i + 1] - stamps[2 * i]) * g_vk.props.limits.timestampPeriod / 1e6);
    vkResetQueryPool(g_vk.device, g_vk.profiler_pool, 0, 2 * n);
    g_vk.profiler_query_count = 0; // queries 0..n-1 are now available again
  }

  g_vk.query_read_index = 0;
  g_vk.query_index = 0;
  g_vk.set_index = 0;
  return BT_OK;
}

VkDescriptorSet bt_vk_next_set(void) {
  if (g_vk.set_index >= BT_VK_MAX_DISPATCHES) (void)bt_vk_wait();
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
