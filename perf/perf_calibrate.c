/* TOGISHI Phase 1 calibration: standalone C + Vulkan microbenchmarks.
 * This file does not include, link or call any Hagane code. */
#include <vulkan/vulkan.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
#define VK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "VkResult %d at %s:%d\n", (int)r_, __FILE__, __LINE__); exit(1); } } while (0)

typedef struct {
  VkInstance inst;
  VkPhysicalDevice phys;
  VkDevice dev;
  VkQueue q;
  uint32_t qfam;
  VkCommandPool pool;
  VkCommandBuffer cmd;
  VkCommandBuffer empty_cmd;
  VkFence fence;
  VkQueryPool qp;
  VkDescriptorPool dpool;
  VkDescriptorSetLayout dlayout;
  VkDescriptorSet set;
  VkPipelineLayout playout;
  VkPhysicalDeviceProperties props;
  VkPhysicalDeviceMemoryProperties memprops;
  int have_ts;
  double period_ns;
  float mem_budget_gb;
} ctx_t;

static ctx_t C;

static double now_ms(void) {
#ifdef _WIN32
  static LARGE_INTEGER freq, started;
  if (freq.QuadPart == 0) { QueryPerformanceFrequency(&freq); QueryPerformanceCounter(&started); }
  LARGE_INTEGER t; QueryPerformanceCounter(&t);
  return (double)(t.QuadPart - started.QuadPart) * 1000.0 / (double)freq.QuadPart;
#else
  return (double)clock() * 1000.0 / CLOCKS_PER_SEC;
#endif
}

static void *load_spv(const char *name, size_t *size) {
  char path[512];
  snprintf(path, sizeof(path), "shaders/%s.spv", name);
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  unsigned char *buf = malloc((size_t)n);
  CHECK(fread(buf, 1, (size_t)n, f) == (size_t)n);
  fclose(f);
  *size = (size_t)n;
  return buf;
}

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
  for (uint32_t i = 0; i < C.memprops.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (C.memprops.memoryTypes[i].propertyFlags & want) == want) return i;
  fprintf(stderr, "no memory type for flags %u\n", (unsigned)want); exit(1);
}

static void make_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                        VkMemoryPropertyFlags props, VkBuffer *buf, VkDeviceMemory *mem) {
  VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                            .size = size, .usage = usage,
                            .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
  VK(vkCreateBuffer(C.dev, &bi, NULL, buf));
  VkMemoryRequirements req; vkGetBufferMemoryRequirements(C.dev, *buf, &req);
  VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                              .allocationSize = req.size,
                              .memoryTypeIndex = mem_type(req.memoryTypeBits, props) };
  VK(vkAllocateMemory(C.dev, &ai, NULL, mem));
  VK(vkBindBufferMemory(C.dev, *buf, *mem, 0));
}

static void init_vulkan(void) {
  VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "togishi-calibrate",
                            .apiVersion = VK_API_VERSION_1_2 };
  VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
  VK(vkCreateInstance(&ici, NULL, &C.inst));

  uint32_t n = 0; vkEnumeratePhysicalDevices(C.inst, &n, NULL); CHECK(n > 0);
  VkPhysicalDevice *devs = malloc(n * sizeof(*devs));
  VK(vkEnumeratePhysicalDevices(C.inst, &n, devs));
  C.phys = NULL;
  for (uint32_t i = 0; i < n; ++i) {
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(devs[i], &p);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, NULL);
    VkQueueFamilyProperties *qs = malloc(nq * sizeof(*qs));
    vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, qs);
    for (uint32_t q = 0; q < nq; ++q)
      if ((qs[q].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
          (!C.phys || p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) {
        C.phys = devs[i]; C.qfam = q; C.props = p;
      }
    free(qs);
  }
  CHECK(C.phys);
  vkGetPhysicalDeviceMemoryProperties(C.phys, &C.memprops);
  C.have_ts = C.props.limits.timestampComputeAndGraphics ? 1 : 0;
  C.period_ns = C.have_ts ? C.props.limits.timestampPeriod : 0.0;

  VkPhysicalDeviceFeatures supported, features = { 0 };
  vkGetPhysicalDeviceFeatures(C.phys, &supported);
  features.shaderFloat64 = supported.shaderFloat64;
  float pri = 1.f;
  VkDeviceQueueCreateInfo qi = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                 .queueFamilyIndex = C.qfam, .queueCount = 1, .pQueuePriorities = &pri };
  VkDeviceCreateInfo di = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                            .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
                            .pEnabledFeatures = &features };
  VK(vkCreateDevice(C.phys, &di, NULL, &C.dev));
  vkGetDeviceQueue(C.dev, C.qfam, 0, &C.q);

  VkCommandPoolCreateInfo cp = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                 .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                 .queueFamilyIndex = C.qfam };
  VK(vkCreateCommandPool(C.dev, &cp, NULL, &C.pool));
  VkCommandBufferAllocateInfo ca = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                     .commandPool = C.pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                     .commandBufferCount = 1 };
  VkCommandBuffer bufs[2];
  ca.commandBufferCount = 2;
  VK(vkAllocateCommandBuffers(C.dev, &ca, bufs));
  C.cmd = bufs[0];
  C.empty_cmd = bufs[1];
  VkCommandBufferBeginInfo ebi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT };
  VK(vkBeginCommandBuffer(C.empty_cmd, &ebi));
  VK(vkEndCommandBuffer(C.empty_cmd));
  VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
  VK(vkCreateFence(C.dev, &fi, NULL, &C.fence));
  if (C.have_ts) {
    VkQueryPoolCreateInfo qi2 = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                  .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                  .queryCount = 2, .flags = VK_QUERY_POOL_CREATE_RESET_BIT_KHR };
    VK(vkCreateQueryPool(C.dev, &qi2, NULL, &C.qp));
  }

  VkDescriptorSetLayoutBinding bind[2];
  for (int i = 0; i < 2; ++i) bind[i] = (VkDescriptorSetLayoutBinding){
    .binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
  VkDescriptorSetLayoutCreateInfo dli = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .bindingCount = 2, .pBindings = bind };
  VK(vkCreateDescriptorSetLayout(C.dev, &dli, NULL, &C.dlayout));
  VkDescriptorPoolSize ps = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 };
  VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                     .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
  VK(vkCreateDescriptorPool(C.dev, &dpi, NULL, &C.dpool));
  VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                      .descriptorPool = C.dpool, .descriptorSetCount = 1,
                                      .pSetLayouts = &C.dlayout };
  VK(vkAllocateDescriptorSets(C.dev, &dai, &C.set));
  VkPushConstantRange pc = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 128 };
  VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .setLayoutCount = 1, .pSetLayouts = &C.dlayout,
                                     .pushConstantRangeCount = 1, .pPushConstantRanges = &pc };
  VK(vkCreatePipelineLayout(C.dev, &pli, NULL, &C.playout));
}

static VkPipeline pipeline(const char *name) {
  size_t size; void *code = load_spv(name, &size);
  VkShaderModuleCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = size, .pCode = code };
  VkShaderModule m; VK(vkCreateShaderModule(C.dev, &sci, NULL, &m));
  VkPipelineShaderStageCreateInfo st = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main" };
  VkComputePipelineCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = st, .layout = C.playout };
  VkPipeline p; VK(vkCreateComputePipelines(C.dev, VK_NULL_HANDLE, 1, &cpi, NULL, &p));
  vkDestroyShaderModule(C.dev, m, NULL);
  free(code);
  return p;
}

static void bind_buffers(VkBuffer a, VkBuffer b) {
  VkDescriptorBufferInfo info[2] = {
    { .buffer = a, .offset = 0, .range = VK_WHOLE_SIZE },
    { .buffer = b, .offset = 0, .range = VK_WHOLE_SIZE } };
  VkWriteDescriptorSet w[2];
  for (int i = 0; i < 2; ++i) w[i] = (VkWriteDescriptorSet){
    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = C.set,
    .dstBinding = (uint32_t)i, .descriptorCount = 1,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &info[i] };
  vkUpdateDescriptorSets(C.dev, 2, w, 0, NULL);
}

static void begin(VkPipeline p, const void *pc, uint32_t pc_size) {
  VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                  .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT };
  VK(vkBeginCommandBuffer(C.cmd, &bi));
  vkCmdBindPipeline(C.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p);
  vkCmdBindDescriptorSets(C.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, C.playout, 0, 1, &C.set, 0, NULL);
  vkCmdPushConstants(C.cmd, C.playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_size, pc);
  if (C.have_ts) vkCmdWriteTimestamp(C.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, C.qp, 0);
}

static void barrier(void) {
  VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                         .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                         .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };
  vkCmdPipelineBarrier(C.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

static double finish(void) {
  if (C.have_ts) vkCmdWriteTimestamp(C.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, C.qp, 1);
  VK(vkEndCommandBuffer(C.cmd));
  VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                      .commandBufferCount = 1, .pCommandBuffers = &C.cmd };
  VK(vkResetFences(C.dev, 1, &C.fence));
  VK(vkQueueSubmit(C.q, 1, &si, C.fence));
  VK(vkWaitForFences(C.dev, 1, &C.fence, VK_TRUE, UINT64_MAX));
  if (!C.have_ts) return 0.0;
  uint64_t ts[2];
  VK(vkGetQueryPoolResults(C.dev, C.qp, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
  vkResetQueryPool(C.dev, C.qp, 0, 2);
  return (double)(ts[1] - ts[0]) * C.period_ns / 1e6;
}

typedef struct { double ms; double gb_s; double flops; } result_t;

int main(int argc, char **argv) {
  const char *json_path = "calibration.json";
  for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], "--json")) json_path = argv[++i];

  init_vulkan();
  printf("GPU: %s\nVulkan API %u.%u.%u, driver %u\n",
         C.props.deviceName,
         VK_VERSION_MAJOR(C.props.apiVersion), VK_VERSION_MINOR(C.props.apiVersion),
         VK_VERSION_PATCH(C.props.apiVersion), C.props.driverVersion);
  printf("timestampPeriod=%.3f ns, timestamp=%d, shared=%u, push=%u, wg=%u\n",
         C.period_ns, C.have_ts, (unsigned)C.props.limits.maxComputeSharedMemorySize,
         (unsigned)C.props.limits.maxPushConstantsSize,
         (unsigned)C.props.limits.maxComputeWorkGroupInvocations);

  FILE *out = fopen(json_path, "w"); CHECK(out);
  fprintf(out, "{\n  \"device\": \"%s\",\n", C.props.deviceName);
  fprintf(out, "  \"timestampPeriodNs\": %.3f,\n", C.period_ns);
  fprintf(out, "  \"maxComputeSharedMemorySize\": %u,\n", (unsigned)C.props.limits.maxComputeSharedMemorySize);
  fprintf(out, "  \"maxPushConstantsSize\": %u,\n", (unsigned)C.props.limits.maxPushConstantsSize);
  fprintf(out, "  \"maxComputeWorkGroupInvocations\": %u,\n", (unsigned)C.props.limits.maxComputeWorkGroupInvocations);

  /* ---- vec4 copy ---- */
  {
    VkBuffer in, outb; VkDeviceMemory mi, mo;
    VkDeviceSize bytes = 1ull << 30;
    make_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &in, &mi);
    make_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &outb, &mo);
    bind_buffers(in, outb);
    VkPipeline pf = pipeline("fill");
    uint32_t pcf = (uint32_t)(bytes / 16);
    begin(pf, &pcf, sizeof(pcf));
    vkCmdDispatch(C.cmd, (pcf + 255) / 256, 1, 1);
    (void)finish();
    vkDestroyPipeline(C.dev, pf, NULL);

    VkPipeline p = pipeline("copy");
    uint32_t pc = (uint32_t)(bytes / 16);
    int reps = 1; // 1 GiB per pass: one pass is enough and avoids identical rewrites
    begin(p, &pc, sizeof(pc));
    for (int i = 0; i < reps; ++i) { vkCmdDispatch(C.cmd, (pc + 255) / 256, 1, 1); if (i + 1 < reps) barrier(); }
    double ms = finish();
    double per = ms / reps;
    double bw = (double)(2 * bytes) / (per * 1e-3) / 1e9;
    printf("copy vec4 1GiB: %.4f ms/dispatch, %.1f GB/s (%.1f%% of 336)\n",
           per, bw, 100.0 * bw / 336.0);
    fprintf(out, "  \"copy\": {\"ms\": %.6f, \"GBs\": %.3f, \"pctTheory\": %.3f},\n",
            per, bw, 100.0 * bw / 336.0);
    VkPipeline p4 = pipeline("copy4");
    uint32_t pc4 = (uint32_t)(bytes / 16 / 4);
    begin(p4, &pc4, sizeof(pc4));
    for (int i = 0; i < reps; ++i) { vkCmdDispatch(C.cmd, (pc4 + 255) / 256, 1, 1); if (i + 1 < reps) barrier(); }
    double total4 = finish();
    double per4 = total4 / reps;
    double bw4 = (double)(2 * bytes) / (per4 * 1e-3) / 1e9;
    printf("copy4 vec4x4 1GiB: %.4f ms/dispatch, %.1f GB/s (%.1f%% of 336)\n",
           per4, bw4, 100.0 * bw4 / 336.0);
    fprintf(out, "  \"copy4\": {\"ms\": %.6f, \"GBs\": %.3f, \"pctTheory\": %.3f},\n",
            per4, bw4, 100.0 * bw4 / 336.0);
    vkDestroyPipeline(C.dev, p4, NULL);
    vkDestroyPipeline(C.dev, p, NULL); vkDestroyBuffer(C.dev, in, NULL); vkFreeMemory(C.dev, mi, NULL);
    vkDestroyBuffer(C.dev, outb, NULL); vkFreeMemory(C.dev, mo, NULL);
  }

  /* ---- FP32 FMA ---- */
  {
    VkBuffer in, outb; VkDeviceMemory mi, mo;
    make_buffer(1024 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &in, &mi);
    make_buffer(524288 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &outb, &mo);
    bind_buffers(in, outb);
    VkPipeline pf2 = pipeline("fill");
    uint32_t pcf2 = 256;
    begin(pf2, &pcf2, sizeof(pcf2));
    vkCmdDispatch(C.cmd, 1, 1, 1);
    (void)finish();
    vkDestroyPipeline(C.dev, pf2, NULL);

    VkPipeline p = pipeline("fma");
    uint32_t pc[2] = { 65536, 1024 };
    int reps = 100;
    begin(p, pc, sizeof(pc));
    for (int i = 0; i < reps; ++i) { vkCmdDispatch(C.cmd, 65536 / 256, 1, 1); if (i + 1 < reps) barrier(); }
    double ms = finish() / reps;
    double flops = 65536.0 * 1024.0 * 2.0;
    double gf = flops / (ms * 1e-3) / 1e9;
    printf("fp32 FMA: %.4f ms/dispatch, %.1f GFLOP/s\n", ms, gf);
    fprintf(out, "  \"fma32\": {\"ms\": %.6f, \"GFLOPs\": %.3f},\n", ms, gf);
    VkPipeline p8 = pipeline("fma8");
    uint32_t pc8[2] = { 524288, 128 };
    begin(p8, pc8, sizeof(pc8));
    for (int i = 0; i < reps; ++i) { vkCmdDispatch(C.cmd, 524288 / 256, 1, 1); if (i + 1 < reps) barrier(); }
    double ms8 = finish() / reps;
    double gf8 = (524288.0 * 1024.0 * 2.0) / (ms8 * 1e-3) / 1e9;
    printf("fp32 FMA x8 accumulators: %.4f ms/dispatch, %.1f GFLOP/s\n", ms8, gf8);
    fprintf(out, "  \"fma32x8\": {\"ms\": %.6f, \"GFLOPs\": %.3f},\n", ms8, gf8);
    VkPipeline p16 = pipeline("fma16");
    uint32_t pc16[2] = { 65536, 64 };
    begin(p16, pc16, sizeof(pc16));
    for (int i = 0; i < reps; ++i) { vkCmdDispatch(C.cmd, 65536 / 256, 1, 1); if (i + 1 < reps) barrier(); }
    double ms16 = finish() / reps;
    double gf16 = flops / (ms16 * 1e-3) / 1e9;
    printf("fp32 FMA x16 accumulators: %.4f ms/dispatch, %.1f GFLOP/s\n", ms16, gf16);
    fprintf(out, "  \"fma32x16\": {\"ms\": %.6f, \"GFLOPs\": %.3f},\n", ms16, gf16);
    vkDestroyPipeline(C.dev, p16, NULL);
    vkDestroyPipeline(C.dev, p8, NULL);
    vkDestroyPipeline(C.dev, p, NULL); vkDestroyBuffer(C.dev, in, NULL); vkFreeMemory(C.dev, mi, NULL);
    vkDestroyBuffer(C.dev, outb, NULL); vkFreeMemory(C.dev, mo, NULL);
  }

  /* ---- shared-memory bank conflicts ---- */
  {
    VkBuffer outb; VkDeviceMemory mo;
    make_buffer(65536 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &outb, &mo);
    bind_buffers(outb, outb);
    VkPipeline p = pipeline("shared");
    fprintf(out, "  \"shared\": {");
    int strides[3] = { 1, 2, 32 };
    for (int k = 0; k < 3; ++k) {
      uint32_t pc = (uint32_t)strides[k];
      int reps = 200;
      begin(p, &pc, sizeof(pc));
      for (int i = 0; i < reps; ++i) { vkCmdDispatch(C.cmd, 256, 1, 1); if (i + 1 < reps) barrier(); }
      double ms = finish() / reps;
      printf("shared stride %2d: %.6f ms/dispatch\n", strides[k], ms);
      fprintf(out, "\"s%d\": %.6f%s", strides[k], ms, k == 2 ? "" : ", ");
    }
    fprintf(out, "},\n");
    vkDestroyPipeline(C.dev, p, NULL); vkDestroyBuffer(C.dev, outb, NULL); vkFreeMemory(C.dev, mo, NULL);
  }

  /* ---- L2 vs DRAM reads ---- */
  {
    VkPipeline p = pipeline("readlat");
    VkDeviceSize sizes[2] = { 1ull << 20, 512ull << 20 };
    uint32_t threads[2] = { 4096, 262144 };
    uint32_t strides[2] = { 4096, 262144 };
    const char *names[2] = { "l2read", "dramread" };
    fprintf(out, "  \"reads\": {");
    for (int k = 0; k < 2; ++k) {
      VkBuffer in, outb; VkDeviceMemory mi, mo;
      make_buffer(sizes[k], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &in, &mi);
      make_buffer(threads[k] * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &outb, &mo);
      bind_buffers(in, outb);
      VkPipeline pf3 = pipeline("fill");
      uint32_t pcf3 = (uint32_t)(sizes[k] / 16);
      begin(pf3, &pcf3, sizeof(pcf3));
      vkCmdDispatch(C.cmd, (pcf3 + 255) / 256, 1, 1);
      (void)finish();
      vkDestroyPipeline(C.dev, pf3, NULL);
      uint32_t pc[2] = { (uint32_t)(sizes[k] / 4), strides[k] };
      int reps = k == 0 ? 100 : 10;
      begin(p, pc, sizeof(pc));
      for (int i = 0; i < reps; ++i) { vkCmdDispatch(C.cmd, threads[k] / 256, 1, 1); if (i + 1 < reps) barrier(); }
      double ms = finish() / reps;
      double bw = threads[k] * 64.0 * 4.0 / (ms * 1e-3) / 1e9;
      printf("%-8s: %.4f ms, %.1f GB/s\n", names[k], ms, bw);
      fprintf(out, "\"%s\": {\"ms\": %.6f, \"GBs\": %.3f}%s",
              names[k], ms, bw, k == 1 ? "" : ", ");
      vkDestroyBuffer(C.dev, in, NULL); vkFreeMemory(C.dev, mi, NULL);
      vkDestroyBuffer(C.dev, outb, NULL); vkFreeMemory(C.dev, mo, NULL);
    }
    fprintf(out, "},\n");
    vkDestroyPipeline(C.dev, p, NULL);
  }

  /* ---- GPU empty dispatch time ---- */
  {
    VkBuffer outb; VkDeviceMemory mo;
    make_buffer(65536 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &outb, &mo);
    bind_buffers(outb, outb);
    VkPipeline p = pipeline("empty");
    int reps = 2000;
    begin(p, NULL, 0);
    for (int i = 0; i < reps; ++i) { vkCmdDispatch(C.cmd, 256, 1, 1); if (i + 1 < reps) barrier(); }
    double ms = finish() / reps;
    printf("empty dispatch+barrier: %.6f ms each\n", ms);
    fprintf(out, "  \"emptyDispatchMs\": %.6f,\n", ms);
    vkDestroyPipeline(C.dev, p, NULL); vkDestroyBuffer(C.dev, outb, NULL); vkFreeMemory(C.dev, mo, NULL);
  }

  /* ---- CPU-side costs ---- */
  {
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT };
    VK(vkBeginCommandBuffer(C.cmd, &bi));
    VkBuffer tiny; VkDeviceMemory mtiny;
    make_buffer(65536, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &tiny, &mtiny);
    bind_buffers(tiny, tiny);
    VkPipeline ep = pipeline("empty");
    vkCmdBindPipeline(C.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ep);
    vkCmdBindDescriptorSets(C.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, C.playout, 0, 1, &C.set, 0, NULL);
    int n = 20000;
    double t0 = now_ms();
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                           .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                           .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT };
    for (int i = 0; i < n; ++i)
      vkCmdPipelineBarrier(C.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    double barrier_us = (now_ms() - t0) * 1000.0 / n;
    t0 = now_ms();
    for (int i = 0; i < n; ++i) vkCmdDispatch(C.cmd, 1, 1, 1);
    double dispatch_us = (now_ms() - t0) * 1000.0 / n;
    VK(vkEndCommandBuffer(C.cmd));

    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1, .pCommandBuffers = &C.empty_cmd };
    int m = 2000;
    t0 = now_ms();
    for (int i = 0; i < m; ++i) {
      VK(vkResetFences(C.dev, 1, &C.fence));
      VK(vkQueueSubmit(C.q, 1, &si, C.fence));
      VK(vkWaitForFences(C.dev, 1, &C.fence, VK_TRUE, UINT64_MAX));
    }
    double submit_us = (now_ms() - t0) * 1000.0 / m;

    VkSemaphoreTypeCreateInfo sti = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = 0 };
    VkSemaphoreCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &sti };
    VkSemaphore sem; VK(vkCreateSemaphore(C.dev, &sci, NULL, &sem));
    t0 = now_ms();
    for (uint64_t i = 1; i <= (uint64_t)m; ++i) {
      uint64_t value = i;
      VkTimelineSemaphoreSubmitInfo tsi = { .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                                            .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &value };
      VkSubmitInfo s2 = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &tsi,
                          .commandBufferCount = 1, .pCommandBuffers = &C.empty_cmd,
                          .signalSemaphoreCount = 1, .pSignalSemaphores = &sem };
      VK(vkQueueSubmit(C.q, 1, &s2, VK_NULL_HANDLE));
      VkSemaphoreWaitInfo wi = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                 .semaphoreCount = 1, .pSemaphores = &sem, .pValues = &value };
      VK(vkWaitSemaphores(C.dev, &wi, UINT64_MAX));
    }
    double timeline_us = (now_ms() - t0) * 1000.0 / m;

    VkDeviceSize bytes = 4ull << 20;
    t0 = now_ms();
    for (int i = 0; i < 1000; ++i) {
      VkBuffer b; VkDeviceMemory mem;
      make_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &b, &mem);
      vkDestroyBuffer(C.dev, b, NULL); vkFreeMemory(C.dev, mem, NULL);
    }
    double alloc_us = (now_ms() - t0) * 1000.0 / 1000.0;

    printf("CPU barrier %.3f us, dispatch %.3f us, submit+fence %.3f us, timeline %.3f us, alloc+free %.3f us\n",
           barrier_us, dispatch_us, submit_us, timeline_us, alloc_us);
    fprintf(out, "  \"cpu\": {\"barrierUs\": %.4f, \"dispatchUs\": %.4f, \"submitFenceUs\": %.4f, "
                 "\"timelineUs\": %.4f, \"allocFreeUs\": %.4f},\n",
            barrier_us, dispatch_us, submit_us, timeline_us, alloc_us);
    vkDestroySemaphore(C.dev, sem, NULL);
    vkDestroyPipeline(C.dev, ep, NULL);
    vkDestroyBuffer(C.dev, tiny, NULL); vkFreeMemory(C.dev, mtiny, NULL);
  }

  /* ---- PCIe staging copies ---- */
  {
    VkDeviceSize bytes = 64ull << 20;
    VkBuffer host, devbuf; VkDeviceMemory mh, md;
    make_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &host, &mh);
    make_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &devbuf, &md);
    bind_buffers(host, devbuf);
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT };
    VK(vkBeginCommandBuffer(C.cmd, &bi));
    VkBufferCopy cpy = { .size = bytes };
    int reps = 20;
    if (C.have_ts) vkCmdWriteTimestamp(C.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, C.qp, 0);
    for (int i = 0; i < reps; ++i) {
      vkCmdCopyBuffer(C.cmd, host, devbuf, 1, &cpy);
      vkCmdCopyBuffer(C.cmd, devbuf, host, 1, &cpy);
    }
    if (C.have_ts) vkCmdWriteTimestamp(C.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, C.qp, 1);
    VK(vkEndCommandBuffer(C.cmd));
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &C.cmd };
    VK(vkResetFences(C.dev, 1, &C.fence));
    VK(vkQueueSubmit(C.q, 1, &si, C.fence));
    VK(vkWaitForFences(C.dev, 1, &C.fence, VK_TRUE, UINT64_MAX));
    double ms = 0;
    if (C.have_ts) {
      uint64_t ts[2];
      VK(vkGetQueryPoolResults(C.dev, C.qp, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
      vkResetQueryPool(C.dev, C.qp, 0, 2);
      ms = (double)(ts[1] - ts[0]) * C.period_ns / 1e6 / reps;
    }
    double bw = (double)(2 * bytes) / (ms * 1e-3) / 1e9;
    printf("staging H2D+D2H 64MiB: %.4f ms, %.1f GB/s combined\n", ms, bw);
    fprintf(out, "  \"pcie\": {\"ms\": %.6f, \"GBs\": %.3f}\n", ms, bw);
    vkDestroyBuffer(C.dev, host, NULL); vkFreeMemory(C.dev, mh, NULL);
    vkDestroyBuffer(C.dev, devbuf, NULL); vkFreeMemory(C.dev, md, NULL);
  }

  fprintf(out, "}\n");
  fclose(out);
  printf("wrote %s\n", json_path);
  return 0;
}
