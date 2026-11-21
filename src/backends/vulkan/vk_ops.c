#include "vk_internal.h"

#ifdef _WIN32
#include <windows.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../core/conv_dispatch_internal.h"
#include "../../core/dispatcher_internal.h"
#include <bitsy/bt_profiler.h>
#include "../../core/error_internal.h"
#include "../../core/storage_internal.h"
#include "../../core/tensor_internal.h"
#include "vk_device_internal.h"

static double prof_now_us(void) {
#ifdef _WIN32
  static LARGE_INTEGER freq;
  if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
  LARGE_INTEGER t; QueryPerformanceCounter(&t);
  return (double)t.QuadPart * 1e6 / (double)freq.QuadPart;
#else
  return 0.0;
#endif
}

/* Push constants mirror the GLSL PC block exactly (120 bytes). */
typedef struct {
  uint32_t n, ndim, op;
  int32_t dims[BT_MAX_DIMS];
  int32_t sa[BT_MAX_DIMS], sb[BT_MAX_DIMS], so[BT_MAX_DIMS];
  int32_t offa, offb, offo;
} vk_pc_t;

static int32_t stride_at(const bt_tensor_t* t, int64_t out_dim, int64_t out_ndim) {
  int64_t in_dim = out_dim - (out_ndim - t->shape.ndim);
  if (in_dim < 0 || in_dim >= t->shape.ndim || t->shape.dims[in_dim] == 1) return 0;
  return (int32_t)t->strides[in_dim];
}

/* Hazard tracking: emit a barrier only when a buffer was written/read by an
 * earlier dispatch in this batch. Replaces the unconditional global barrier. */
#define HAZ_MAX 2048
static VkBuffer haz_buf[HAZ_MAX];
static unsigned char haz_written[HAZ_MAX];
static unsigned char haz_read[HAZ_MAX];
static int haz_count;

void bt_vk_hazard_reset(void) {
  for (int i = 0; i < haz_count; ++i) { haz_buf[i] = VK_NULL_HANDLE; haz_written[i] = 0; haz_read[i] = 0; }
  haz_count = 0;
}

static int haz_slot(VkBuffer buffer) {
  for (int i = 0; i < haz_count; ++i) if (haz_buf[i] == buffer) return i;
  if (haz_count >= HAZ_MAX) return -1;
  haz_buf[haz_count] = buffer;
  haz_written[haz_count] = 0;
  haz_read[haz_count] = 0;
  return haz_count++;
}

static void hazard_barrier(VkBuffer buffers[4]) {
  VkBufferMemoryBarrier mb[3];
  int n = 0;
  for (int i = 0; i < 3; ++i) { // binding 3 (program) is read-only host-written
    int slot = haz_slot(buffers[i]);
    if (slot < 0) continue;
    VkAccessFlags src = 0, dst = 0;
    if (i < 2 && haz_written[slot]) { src = VK_ACCESS_SHADER_WRITE_BIT; dst = VK_ACCESS_SHADER_READ_BIT; }
    if (i == 2 && (haz_written[slot] || haz_read[slot])) {
      src = haz_written[slot] ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
      dst = VK_ACCESS_SHADER_WRITE_BIT;
    }
    if (!src) continue;
    mb[n++] = (VkBufferMemoryBarrier){
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, .buffer = buffers[i],
      .offset = 0, .size = VK_WHOLE_SIZE, .srcAccessMask = src, .dstAccessMask = dst };
  }
  if (n) {
    vkCmdPipelineBarrier(g_vk.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, NULL, (uint32_t)n, mb, 0, NULL);
    bt_profiler_barrier();
  }
}

static void hazard_mark(VkBuffer buffers[4]) {
  for (int i = 0; i < 2; ++i) {
    int slot = haz_slot(buffers[i]);
    if (slot >= 0) haz_read[slot] = 1;
  }
  int slot = haz_slot(buffers[2]);
  if (slot >= 0) { haz_written[slot] = 1; haz_read[slot] = 0; }
}

/* Record a dispatch into the current batch. Submission happens at flush time. */
static bt_status_t dispatch(VkPipeline pipeline, VkBuffer buffers[4],
                            const void* pc, size_t pc_size, uint32_t gx, uint32_t gy,
                            const char* name, double bytes, double flops) {
  double t0 = prof_now_us();
  VkDescriptorSet set = bt_vk_next_set(); // may flush when the batch is full
  VkDescriptorBufferInfo info[4];
  VkWriteDescriptorSet writes[4];
  for (int i = 0; i < 4; ++i) {
    info[i] = (VkDescriptorBufferInfo){ .buffer = buffers[i], .offset = 0, .range = VK_WHOLE_SIZE };
    writes[i] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
      .dstBinding = (uint32_t)i, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &info[i],
    };
  }
  vkUpdateDescriptorSets(g_vk.device, 4, writes, 0, NULL);

  bt_vk_begin_cmd();
  vkCmdBindPipeline(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          g_vk.pipeline_layout, 0, 1, &set, 0, NULL);
  vkCmdPushConstants(g_vk.cmd, g_vk.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                     0, (uint32_t)pc_size, pc);
  hazard_barrier(buffers);
  bt_vk_timestamp_start();
  int qi = -1;
  if (bt_profiler_enabled() && g_vk.have_timestamps &&
      g_vk.profiler_query_count < 2048) {
    qi = (int)g_vk.profiler_query_count++;
    vkCmdWriteTimestamp(g_vk.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        g_vk.profiler_pool, (uint32_t)(2 * qi));
  }
  vkCmdDispatch(g_vk.cmd, gx, gy, 1);
  if (qi >= 0)
    vkCmdWriteTimestamp(g_vk.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        g_vk.profiler_pool, (uint32_t)(2 * qi + 1));
  bt_vk_timestamp_end();
  hazard_mark(buffers);
  int event_index = bt_profiler_record(name, t0, prof_now_us(), bytes, flops, qi);
  if (qi >= 0) g_vk.profiler_events[qi] = event_index;
  // Experiment toggle: BT_VK_NO_BATCH=1 restores the old submit-per-op path.
  static int no_batch = -1;
  if (no_batch < 0) {
    const char* env = getenv("BT_VK_NO_BATCH");
    no_batch = env && atoi(env) ? 1 : 0;
  }
  return no_batch ? bt_vk_flush() : BT_OK;
}

bt_status_t bt_vk_elementwise(const bt_tensor_t* a, const bt_tensor_t* b,
                              bt_tensor_t* out, int op) {
  if (!a || !b || !out || !g_vk.ready)
    return bt_fail(BT_ERR_BACKEND, "bt_vk_elementwise: NULL tensor or Vulkan not initialized");
  int64_t total = bt_shape_numel(&out->shape);
  if (!total) return BT_OK; // no dispatch for empty tensors

  vk_pc_t pc = { .n = (uint32_t)total, .ndim = (uint32_t)out->shape.ndim, .op = (uint32_t)op,
                 .offa = (int32_t)a->offset, .offb = (int32_t)b->offset,
                 .offo = (int32_t)out->offset };
  for (int64_t d = 0; d < out->shape.ndim; ++d) {
    pc.dims[d] = (int32_t)out->shape.dims[d];
    pc.sa[d] = stride_at(a, d, out->shape.ndim);
    pc.sb[d] = stride_at(b, d, out->shape.ndim);
    pc.so[d] = (int32_t)out->strides[d];
  }

  VkBuffer buffers[4] = {
    bt_vk_buffer_of(a->storage->gpu),
    bt_vk_buffer_of(b->storage->gpu),
    bt_vk_buffer_of(out->storage->gpu),
    bt_vk_buffer_of(a->storage->gpu),
  };
  int64_t last = out->shape.ndim - 1;
  int vec4_ok = last >= 0 && out->shape.dims[last] % 4 == 0 &&
                a->offset % 4 == 0 && b->offset % 4 == 0 && out->offset % 4 == 0 &&
                bt_tensor_is_contiguous(a) && bt_tensor_is_contiguous(b) &&
                bt_tensor_is_contiguous(out) &&
                bt_shape_equal(&a->shape, &out->shape) &&
                bt_shape_equal(&b->shape, &out->shape);
  VkPipeline pipeline = g_vk.pipeline;
  uint32_t dispatch_n = (uint32_t)total;
  const char* name = "elementwise";
  if (vec4_ok) {
    pipeline = g_vk.elementwise_vec4_pipeline;
    dispatch_n = (uint32_t)(total / 4); // one lane handles four floats
    pc.n = dispatch_n;
    name = "elementwise_vec4";
  }
  // 256 lanes per workgroup; a single bounded dispatch keeps TDR out of the picture.
  bt_status_t status = dispatch(pipeline, buffers, &pc, sizeof(pc),
                                (dispatch_n + 255) / 256, 1,
                                name, 3.0 * total * 4.0, 2.0 * total);
  if (status == BT_OK) bt_storage_bump_version(out->storage);
  return status;
}

/* Fused pointwise: reuse the elementwise push-constant layout and odometer. */
static bt_status_t add_relu_kernel(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t* out) {
  int64_t total = bt_shape_numel(&out->shape);
  if (!total) return BT_OK;
  vk_pc_t pc = { .n = (uint32_t)total, .ndim = (uint32_t)out->shape.ndim,
                 .op = BT_OP_FUSED_ADD_RELU, .offo = (int32_t)out->offset,
                 .offa = (int32_t)a->offset, .offb = (int32_t)b->offset };
  for (int64_t d = 0; d < out->shape.ndim; ++d) {
    pc.dims[d] = (int32_t)out->shape.dims[d];
    pc.sa[d] = stride_at(a, d, out->shape.ndim);
    pc.sb[d] = stride_at(b, d, out->shape.ndim);
    pc.so[d] = (int32_t)out->strides[d];
  }
  VkBuffer buffers[4] = {
    bt_vk_buffer_of(a->storage->gpu), bt_vk_buffer_of(b->storage->gpu),
    bt_vk_buffer_of(out->storage->gpu), bt_vk_buffer_of(a->storage->gpu),
  };
  bt_status_t status = dispatch(g_vk.fused_pipeline, buffers, &pc, sizeof(pc),
                                (uint32_t)((total + 255) / 256), 1,
                                "add_relu", 3.0 * total * 4.0, 2.0 * total);
  if (status == BT_OK) bt_storage_bump_version(out->storage);
  return status;
}

typedef struct {
  uint32_t M, N, K;
  int32_t sa0, sa1, sb0, sb1, sc0, sc1;
  int32_t offa, offb, offc;
  int32_t k_start, k_len;
} vk_matmul_pc_t;

/* On-disk autotune cache written by benchmarks/ops/gemm_tune.py.
 * Format: "M N K variant" (one shape per line). Returns -1 on a miss. */
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996) /* fscanf: this probe uses a fixed local cache format */
#endif
static int cached_gemm_variant(uint32_t m, uint32_t n, uint32_t k) {
  const char* path = getenv("BITSY_GEMM_CACHE");
  if (!path) path = "benchmarks/results/gemm_autotune.txt";
  FILE* f = fopen(path, "r");
  if (!f) return -1;
  uint32_t cm, cn, ck;
  int variant, found = -1;
  while (fscanf(f, "%u %u %u %d", &cm, &cn, &ck, &variant) == 4) {
    if (cm == m && cn == n && ck == k) { found = variant; break; }
  }
  fclose(f);
  return found;
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

static bt_status_t matmul_kernel(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t* out) {
  if (!g_vk.ready) return bt_fail(BT_ERR_BACKEND, "bt_vk_matmul: Vulkan not initialized");
  vk_matmul_pc_t pc = {
    .M = (uint32_t)a->shape.dims[0], .K = (uint32_t)a->shape.dims[1],
    .N = (uint32_t)b->shape.dims[1],
    .sa0 = (int32_t)a->strides[0], .sa1 = (int32_t)a->strides[1],
    .sb0 = (int32_t)b->strides[0], .sb1 = (int32_t)b->strides[1],
    .sc0 = (int32_t)out->strides[0], .sc1 = (int32_t)out->strides[1],
    .offa = (int32_t)a->offset, .offb = (int32_t)b->offset,
    .offc = (int32_t)out->offset,
    .k_start = 0, .k_len = (int32_t)pc.K,
  };
  VkBuffer buffers[4] = {
    bt_vk_buffer_of(a->storage->gpu), bt_vk_buffer_of(b->storage->gpu),
    bt_vk_buffer_of(out->storage->gpu), bt_vk_buffer_of(a->storage->gpu),
  };
  const char* env = getenv("BT_GEMM_VARIANT");
  int variant = env ? atoi(env) : -1; // read every call so tuning loops work
  int vec4_ok = a->strides[1] == 1 && b->strides[1] == 1 &&
                a->strides[0] % 4 == 0 && b->strides[0] % 4 == 0;
  int chosen = variant;
  // Measured autotune (exp_008/009): tiled at small sizes, vec4 register-blocked
  // at large sizes when strides are 4-float aligned, scalar rb otherwise.
  if (chosen < 0) {
    int cached = cached_gemm_variant(pc.M, pc.N, pc.K);
    chosen = cached >= 0 ? cached
                         : ((pc.M <= 512 && pc.N <= 512) ? 1 : (vec4_ok ? 4 : 2));
  }
  if ((chosen == 4 || chosen == 5 || chosen == 6 || chosen == 7 || chosen == 8) && !vec4_ok) chosen = 2;

  double m = pc.M, n = pc.N, k = pc.K;
  double gemm_bytes = (m * k + k * n + m * n) * 4.0;
  double gemm_flops = 2.0 * m * n * k;
  bt_status_t status = BT_OK;

  if (chosen == 0 || chosen == 1) {
    uint32_t tile = chosen ? 16 : 8;
    VkPipeline pipeline = chosen ? g_vk.gemm_tiled_pipeline : g_vk.matmul_pipeline;
    status = dispatch(pipeline, buffers, &pc, sizeof(pc),
                      (pc.N + tile - 1) / tile, (pc.M + tile - 1) / tile,
                      chosen ? "gemm_tiled" : "gemm_naive", gemm_bytes, gemm_flops);
  } else if (chosen == 2) {
    status = dispatch(g_vk.gemm_rb_pipeline, buffers, &pc, sizeof(pc),
                      (pc.N + 127) / 128, (pc.M + 127) / 128,
                      "gemm_rb", gemm_bytes, gemm_flops);
  } else if (chosen == 4) {
    status = dispatch(g_vk.gemm_rb_vec4_pipeline, buffers, &pc, sizeof(pc),
                      (pc.N + 127) / 128, (pc.M + 127) / 128,
                      "gemm_rb_vec4", gemm_bytes, gemm_flops);
  } else if (chosen == 5) {
    status = dispatch(g_vk.gemm_rb_vec4_k16_pipeline, buffers, &pc, sizeof(pc),
                      (pc.N + 127) / 128, (pc.M + 127) / 128,
                      "gemm_rb_vec4_k16", gemm_bytes, gemm_flops);
  } else if (chosen == 6) {
    uint32_t tx = (pc.N + 127) / 128, ty = (pc.M + 127) / 128;
    status = dispatch(g_vk.gemm_rb_vec4_swz_pipeline, buffers, &pc, sizeof(pc),
                      tx * ty, 1, "gemm_rb_vec4_swz", gemm_bytes, gemm_flops);
  } else if (chosen == 7) {
    status = dispatch(g_vk.gemm_rb_vec4_128x64_pipeline, buffers, &pc, sizeof(pc),
                      (pc.N + 63) / 64, (pc.M + 127) / 128,
                      "gemm_rb_vec4_128x64", gemm_bytes, gemm_flops);
  } else if (chosen == 8) {
    status = dispatch(g_vk.gemm_rb_vec4_64x128_pipeline, buffers, &pc, sizeof(pc),
                      (pc.N + 127) / 128, (pc.M + 63) / 64,
                      "gemm_rb_vec4_64x128", gemm_bytes, gemm_flops);
  } else {
    const uint32_t splits = 4;
    size_t partial_bytes = (size_t)splits * pc.M * pc.N * sizeof(float);
    void* partial = bt_vk_alloc(partial_bytes);
    if (!partial) return bt_fail(BT_ERR_OUT_OF_MEMORY, "bt_vk_matmul: split-K scratch");
    for (uint32_t s = 0; s < splits; ++s) {
      vk_matmul_pc_t spc = pc;
      spc.k_start = (int32_t)(s * pc.K / splits);
      spc.k_len = (int32_t)((s + 1) * pc.K / splits - s * pc.K / splits);
      spc.offc = (int32_t)(s * pc.M * pc.N);
      VkBuffer sb[4] = { buffers[0], buffers[1], bt_vk_buffer_of(partial), buffers[0] };
      status = dispatch(g_vk.gemm_rb_pipeline, sb, &spc, sizeof(spc),
                        (pc.N + 127) / 128, (pc.M + 127) / 128,
                        "gemm_rb_splitk", gemm_bytes, gemm_flops / splits);
      if (status != BT_OK) break;
    }
    if (status == BT_OK) {
      uint32_t reduce[2] = { pc.M * pc.N, splits };
      VkBuffer rb[4] = { bt_vk_buffer_of(partial), bt_vk_buffer_of(partial),
                         buffers[2], bt_vk_buffer_of(partial) };
      uint32_t total = pc.M * pc.N;
      status = dispatch(g_vk.splitk_reduce_pipeline, rb, reduce, sizeof(reduce),
                        (total + 255) / 256, 1, "splitk_reduce",
                        (double)total * (splits + 1) * 4.0, (double)total * splits);
    }
    bt_vk_defer_free(partial);
  }
  if (status == BT_OK) bt_storage_bump_version(out->storage);
  return status;
}

/* Reuse the elementwise shader with op 8 and swapped input strides. */
static bt_status_t transpose_kernel(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t* out) {
  (void)b;
  vk_pc_t pc = { .n = (uint32_t)(a->shape.dims[0] * a->shape.dims[1]),
                 .ndim = 2, .op = 8, .offo = (int32_t)out->offset,
                 .offa = (int32_t)a->offset, .offb = (int32_t)a->offset };
  pc.dims[0] = (int32_t)a->shape.dims[1];
  pc.dims[1] = (int32_t)a->shape.dims[0];
  pc.sa[0] = (int32_t)a->strides[1];
  pc.sa[1] = (int32_t)a->strides[0];
  pc.so[0] = (int32_t)out->strides[0];
  pc.so[1] = (int32_t)out->strides[1];
  VkBuffer buffers[4] = { bt_vk_buffer_of(a->storage->gpu),
                          bt_vk_buffer_of(a->storage->gpu),
                          bt_vk_buffer_of(out->storage->gpu),
                          bt_vk_buffer_of(a->storage->gpu) };
  bt_status_t status = dispatch(g_vk.pipeline, buffers, &pc, sizeof(pc),
                                (uint32_t)((pc.n + 255) / 256), 1,
                                "transpose", 2.0 * pc.n * 4.0, 0.0);
  if (status == BT_OK) bt_storage_bump_version(out->storage);
  return status;
}

/* exp_017: fused W - lr*grad in one dispatch; lr is a push constant. */
typedef struct { uint32_t n, M, N; float lr;
                 int32_t sa0, sa1, sb0, sb1, sc0, sc1;
                 int32_t offw, offg, offo; } vk_sgd_pc_t;

bt_status_t bt_vk_sgd_step(const bt_tensor_t* w, const bt_tensor_t* grad,
                           float lr, bt_tensor_t* out) {
  if (!g_vk.ready) return bt_fail(BT_ERR_BACKEND, "bt_vk_sgd_step: Vulkan not ready");
  int64_t n = bt_shape_numel(&out->shape);
  if (!n) return BT_OK;
  /* 2D uses real strides (broadcast gradients have stride 0); other ranks keep
   * the original linear indexing. */
  int two_d = w->shape.ndim == 2;
  vk_sgd_pc_t pc = { .n = (uint32_t)n, .lr = lr,
                     .M = two_d ? (uint32_t)w->shape.dims[0] : 1,
                     .N = two_d ? (uint32_t)w->shape.dims[1] : (uint32_t)n,
                     .sa0 = two_d ? (int32_t)w->strides[0] : 0,
                     .sa1 = two_d ? (int32_t)w->strides[1] : 1,
                     .sb0 = two_d ? (int32_t)grad->strides[0] : 0,
                     .sb1 = two_d ? (int32_t)grad->strides[1] : 1,
                     .sc0 = two_d ? (int32_t)out->strides[0] : 0,
                     .sc1 = two_d ? (int32_t)out->strides[1] : 1,
                     .offw = (int32_t)w->offset, .offg = (int32_t)grad->offset,
                     .offo = (int32_t)out->offset };
  VkBuffer buffers[4] = {
    bt_vk_buffer_of(w->storage->gpu), bt_vk_buffer_of(grad->storage->gpu),
    bt_vk_buffer_of(out->storage->gpu), bt_vk_buffer_of(w->storage->gpu),
  };
  bt_status_t status = dispatch(g_vk.sgd_step_pipeline, buffers, &pc, sizeof(pc),
                                (uint32_t)((n + 255) / 256), 1, "sgd_step",
                                3.0 * (double)n * 4.0, 2.0 * (double)n);
  if (status == BT_OK) bt_storage_bump_version(out->storage);
  return status;
}

/* exp_018: dedicated 2D axis reduction (one workgroup, no ones-vector GEMM). */
typedef struct { uint32_t M, N, axis; int32_t sa0, sa1, off; } vk_reduce_pc_t;

bt_status_t bt_vk_reduce_sum(const bt_tensor_t* a, uint32_t axis, bt_tensor_t* out) {
  if (!g_vk.ready) return bt_fail(BT_ERR_BACKEND, "bt_vk_reduce_sum: Vulkan not ready");
  int64_t n = bt_shape_numel(&out->shape);
  if (!n) return BT_OK;
  vk_reduce_pc_t pc = { .M = (uint32_t)a->shape.dims[0],
                        .N = (uint32_t)a->shape.dims[1], .axis = axis,
                        .sa0 = (int32_t)a->strides[0], .sa1 = (int32_t)a->strides[1],
                        .off = (int32_t)a->offset };
  VkBuffer buffers[4] = {
    bt_vk_buffer_of(a->storage->gpu), bt_vk_buffer_of(a->storage->gpu),
    bt_vk_buffer_of(out->storage->gpu), bt_vk_buffer_of(a->storage->gpu),
  };
  bt_status_t status = dispatch(g_vk.reduce_sum_pipeline, buffers, &pc, sizeof(pc),
                                (uint32_t)((n + 255) / 256), 1, "reduce_sum",
                                (double)bt_shape_numel(&a->shape) * 4.0, 0.0);
  if (status == BT_OK) bt_storage_bump_version(out->storage);
  return status;
}

/* exp_019: fused tanh backward, g * (1 - t*t), in one dispatch. */
typedef struct { uint32_t n; } vk_tanh_bwd_pc_t;

bt_status_t bt_vk_tanh_backward(const bt_tensor_t* g, const bt_tensor_t* t, bt_tensor_t* out) {
  if (!g_vk.ready) return bt_fail(BT_ERR_BACKEND, "bt_vk_tanh_backward: Vulkan not ready");
  int64_t n = bt_shape_numel(&out->shape);
  if (!n) return BT_OK;
  vk_tanh_bwd_pc_t pc = { .n = (uint32_t)n };
  VkBuffer buffers[4] = {
    bt_vk_buffer_of(g->storage->gpu), bt_vk_buffer_of(t->storage->gpu),
    bt_vk_buffer_of(out->storage->gpu), bt_vk_buffer_of(g->storage->gpu),
  };
  bt_status_t status = dispatch(g_vk.tanh_backward_pipeline, buffers, &pc, sizeof(pc),
                                (uint32_t)((n + 255) / 256), 1, "tanh_backward",
                                3.0 * (double)n * 4.0, 2.0 * (double)n);
  if (status == BT_OK) bt_storage_bump_version(out->storage);
  return status;
}

/* exp_020: contiguous full-shape gradient from a reduction gradient. */
typedef struct { uint32_t n, outN, gM, gN; int32_t gs0, gs1, offg; } vk_bcast_pc_t;

bt_status_t bt_vk_broadcast_grad(const bt_tensor_t* g, bt_tensor_t* out) {
  if (!g_vk.ready) return bt_fail(BT_ERR_BACKEND, "bt_vk_broadcast_grad: Vulkan not ready");
  int64_t n = bt_shape_numel(&out->shape);
  if (!n) return BT_OK;
  vk_bcast_pc_t pc = { .n = (uint32_t)n, .outN = (uint32_t)out->shape.dims[1],
                       .gM = (uint32_t)g->shape.dims[0], .gN = (uint32_t)g->shape.dims[1],
                       .gs0 = (int32_t)g->strides[0], .gs1 = (int32_t)g->strides[1],
                       .offg = (int32_t)g->offset };
  VkBuffer buffers[4] = {
    bt_vk_buffer_of(g->storage->gpu), bt_vk_buffer_of(g->storage->gpu),
    bt_vk_buffer_of(out->storage->gpu), bt_vk_buffer_of(g->storage->gpu),
  };
  bt_status_t status = dispatch(g_vk.broadcast_grad_pipeline, buffers, &pc, sizeof(pc),
                                (uint32_t)((n + 255) / 256), 1, "broadcast_grad",
                                (double)n * 2.0 * 4.0, 0.0);
  if (status == BT_OK) bt_storage_bump_version(out->storage);
  return status;
}

/* exp_028: fused cross-entropy scalar reduction. */
typedef struct { uint32_t n; float invB; uint32_t offls, offoh, offo; } vk_ce_pc_t;

bt_status_t bt_vk_ce_loss(const bt_tensor_t* ls, const bt_tensor_t* onehot, bt_tensor_t* out) {
  if (!g_vk.ready) return bt_fail(BT_ERR_BACKEND, "bt_vk_ce_loss: Vulkan not ready");
  int64_t n = bt_shape_numel(bt_tensor_shape(ls));
  vk_ce_pc_t pc = { .n = (uint32_t)n,
                    .invB = 1.0f / (float)ls->shape.dims[0],
                    .offls = (uint32_t)ls->offset, .offoh = (uint32_t)onehot->offset,
                    .offo = (uint32_t)out->offset };
  VkBuffer buffers[4] = {
    bt_vk_buffer_of(ls->storage->gpu), bt_vk_buffer_of(onehot->storage->gpu),
    bt_vk_buffer_of(out->storage->gpu), bt_vk_buffer_of(ls->storage->gpu),
  };
  bt_status_t status = dispatch(g_vk.ce_loss_pipeline, buffers, &pc, sizeof(pc),
                                1, 1, "ce_loss", (double)n * 8.0, 0.0);
  if (status == BT_OK) bt_storage_bump_version(out->storage);
  return status;
}

/* exp_027: fused log_softmax, one workgroup per row. */
typedef struct { uint32_t M, N, offx, offo; } vk_logsoftmax_pc_t;

bt_status_t bt_vk_log_softmax(const bt_tensor_t* x, bt_tensor_t* out) {
  if (!g_vk.ready) return bt_fail(BT_ERR_BACKEND, "bt_vk_log_softmax: Vulkan not ready");
  if (x->shape.ndim != 2) return bt_fail(BT_ERR_SHAPE_MISMATCH, "bt_vk_log_softmax: 2D only");
  vk_logsoftmax_pc_t pc = { .M = (uint32_t)x->shape.dims[0],
                            .N = (uint32_t)x->shape.dims[1],
                            .offx = (uint32_t)x->offset, .offo = (uint32_t)out->offset };
  VkBuffer buffers[4] = {
    bt_vk_buffer_of(x->storage->gpu), bt_vk_buffer_of(x->storage->gpu),
    bt_vk_buffer_of(out->storage->gpu), bt_vk_buffer_of(x->storage->gpu),
  };
  bt_status_t status = dispatch(g_vk.log_softmax_pipeline, buffers, &pc, sizeof(pc),
                                pc.M, 1, "log_softmax",
                                2.0 * (double)pc.M * pc.N * 4.0, 0.0);
  if (status == BT_OK) bt_storage_bump_version(out->storage);
  return status;
}

/* exp_024: GEMM with bias added in the epilogue. Requires the vec4-aligned
 * layout of gemm_rb_vec4; callers fall back for other layouts. */
bt_status_t bt_vk_matmul_bias(const bt_tensor_t* a, const bt_tensor_t* b,
                              const bt_tensor_t* bias, bt_tensor_t* out) {
  if (!g_vk.ready) return bt_fail(BT_ERR_BACKEND, "bt_vk_matmul_bias: Vulkan not ready");
  vk_matmul_pc_t pc = {
    .M = (uint32_t)a->shape.dims[0], .K = (uint32_t)a->shape.dims[1],
    .N = (uint32_t)b->shape.dims[1],
    .sa0 = (int32_t)a->strides[0], .sa1 = (int32_t)a->strides[1],
    .sb0 = (int32_t)b->strides[0], .sb1 = (int32_t)b->strides[1],
    .sc0 = (int32_t)out->strides[0], .sc1 = (int32_t)out->strides[1],
    .offa = (int32_t)a->offset, .offb = (int32_t)b->offset, .offc = (int32_t)out->offset,
    .k_start = 0, .k_len = (int32_t)pc.K,
  };
  VkBuffer buffers[4] = {
    bt_vk_buffer_of(a->storage->gpu), bt_vk_buffer_of(b->storage->gpu),
    bt_vk_buffer_of(out->storage->gpu), bt_vk_buffer_of(bias->storage->gpu),
  };
  double m = pc.M, n = pc.N, k = pc.K;
  bt_status_t status;
  if (pc.M <= 512 && pc.N <= 512) {
    /* Small shapes use the tiled kernel (exp_008/009 autotune winner). */
    status = dispatch(g_vk.gemm_tiled_bias_pipeline, buffers, &pc, sizeof(pc),
                      (pc.N + 15) / 16, (pc.M + 15) / 16, "gemm_tiled_bias",
                      (m * k + k * n + m * n) * 4.0, 2.0 * m * n * k);
  } else if (a->strides[1] == 1 && b->strides[1] == 1 &&
             a->strides[0] % 4 == 0 && b->strides[0] % 4 == 0) {
    status = dispatch(g_vk.gemm_bias_pipeline, buffers, &pc, sizeof(pc),
                      (pc.N + 127) / 128, (pc.M + 127) / 128, "gemm_bias",
                      (m * k + k * n + m * n) * 4.0, 2.0 * m * n * k);
  } else {
    return bt_fail(BT_ERR_NOT_IMPLEMENTED, "bt_vk_matmul_bias: unaligned large operands");
  }
  if (status == BT_OK) bt_storage_bump_version(out->storage);
  return status;
}

/* exp_022: two mul-backward gradients in one dispatch. */
typedef struct { uint32_t n, M, N, aM, aN, bM, bN;
                 int32_t sa0, sa1, sb0, sb1;
                 int32_t offg, offa, offb; } vk_dual_mul_pc_t;

static void dual_shape(const bt_tensor_t* t, uint32_t* m, uint32_t* nn,
                       int32_t* s0, int32_t* s1) {
  const bt_shape_t* sh = bt_tensor_shape(t);
  if (sh->ndim == 2) {
    *m = (uint32_t)sh->dims[0]; *nn = (uint32_t)sh->dims[1];
    *s0 = (int32_t)t->strides[0]; *s1 = (int32_t)t->strides[1];
  } else if (sh->ndim == 1) {
    *m = 1; *nn = (uint32_t)sh->dims[0]; *s0 = 0; *s1 = (int32_t)t->strides[0];
  } else {
    *m = 1; *nn = 1; *s0 = 0; *s1 = 0;
  }
}

bt_status_t bt_vk_dual_mul(const bt_tensor_t* g, const bt_tensor_t* a,
                           const bt_tensor_t* b, bt_tensor_t* out) {
  if (!g_vk.ready) return bt_fail(BT_ERR_BACKEND, "bt_vk_dual_mul: Vulkan not ready");
  int64_t n2 = bt_shape_numel(&out->shape);
  int64_t n = n2 / 2;
  if (!n) return BT_OK;
  vk_dual_mul_pc_t pc;
  dual_shape(g, &pc.M, &pc.N, &pc.sa0, &pc.sa1);
  dual_shape(a, &pc.aM, &pc.aN, &pc.sa0, &pc.sa1);
  dual_shape(b, &pc.bM, &pc.bN, &pc.sb0, &pc.sb1);
  pc.n = (uint32_t)n;
  pc.offg = (int32_t)g->offset;
  pc.offa = (int32_t)a->offset;
  pc.offb = (int32_t)b->offset;
  VkBuffer buffers[4] = {
    bt_vk_buffer_of(g->storage->gpu), bt_vk_buffer_of(a->storage->gpu),
    bt_vk_buffer_of(b->storage->gpu), bt_vk_buffer_of(out->storage->gpu),
  };
  bt_status_t status = dispatch(g_vk.dual_mul_pipeline, buffers, &pc, sizeof(pc),
                                (uint32_t)((n + 255) / 256), 1, "dual_mul",
                                4.0 * (double)n * 4.0, 2.0 * (double)n);
  if (status == BT_OK) bt_storage_bump_version(out->storage);
  return status;
}

/* *** convolution *** */

typedef struct {
  uint32_t mode, n;
  int32_t N, C, H, W, O, kH, kW, pad, stride, OH, OW;
  int32_t sa[4], sw[4], sd[4], so[4];
  int32_t has_bias;
} conv_pc_t;

static bt_status_t conv_kernel(const bt_tensor_t* a, const bt_tensor_t* b,
                               const bt_tensor_t* c, bt_conv_params_t params,
                               bt_conv_kind_t kind, bt_tensor_t* out) {
  if (!g_vk.ready) return bt_fail(BT_ERR_BACKEND, "bt_vk_conv: Vulkan not ready");
  conv_pc_t pc = { .mode = (uint32_t)kind, .n = (uint32_t)bt_shape_numel(&out->shape),
                   .N = (int32_t)a->shape.dims[0],
                   .O = (int32_t)(kind == BT_CONV_DB ? out->shape.dims[0] : b->shape.dims[0]),
                   .pad = params.pad, .stride = params.stride,
                   .has_bias = kind == BT_CONV_FORWARD ? 1 : 0 };
  pc.C = (int32_t)(kind == BT_CONV_DX ? b->shape.dims[1] : a->shape.dims[1]);
  pc.H = (int32_t)(kind == BT_CONV_DX ? out->shape.dims[2] : a->shape.dims[2]);
  pc.W = (int32_t)(kind == BT_CONV_DX ? out->shape.dims[3] : a->shape.dims[3]);
  // DX/forward receive weights in `b`; DW receives dy, so take kernel size
  // from the dW output shape instead.
  int has_weights = kind == BT_CONV_FORWARD || kind == BT_CONV_DX;
  pc.kH = (int32_t)(has_weights ? b->shape.dims[2] : out->shape.dims[2]);
  pc.kW = (int32_t)(has_weights ? b->shape.dims[3] : out->shape.dims[3]);
  const bt_shape_t* dy_shape = kind == BT_CONV_DW ? &b->shape : &a->shape;
  const bt_shape_t* os = kind == BT_CONV_FORWARD ? &out->shape : dy_shape;
  pc.OH = (int32_t)os->dims[2];
  pc.OW = (int32_t)os->dims[3];
  for (int i = 0; i < 4; ++i) {
    pc.sa[i] = (int32_t)a->strides[i];
    pc.sw[i] = b ? (int32_t)b->strides[i] : pc.sa[i];
    pc.so[i] = (int32_t)out->strides[i];
    const bt_tensor_t* dy = kind == BT_CONV_DW ? b : a;
    pc.sd[i] = (int32_t)dy->strides[i];
  }
  VkBuffer buffers[4] = {
    bt_vk_buffer_of(a->storage->gpu),
    bt_vk_buffer_of(b ? b->storage->gpu : a->storage->gpu),
    bt_vk_buffer_of(out->storage->gpu),
    bt_vk_buffer_of(c ? c->storage->gpu : a->storage->gpu),
  };
  bt_status_t status = dispatch(g_vk.conv_pipeline, buffers, &pc, sizeof(pc),
                                (uint32_t)((pc.n + 255) / 256), 1,
                                "conv", 3.0 * pc.n * 4.0,
                                2.0 * pc.n * pc.C * pc.kH * pc.kW);
  if (status == BT_OK) bt_storage_bump_version(out->storage);
  return status;
}

#define KERNEL(name, op) \
  static bt_status_t name(const bt_tensor_t* a, const bt_tensor_t* b, bt_tensor_t* out) { \
    return bt_vk_elementwise(a, b, out, op); \
  }

KERNEL(add, BT_OP_ADD)
KERNEL(sub, BT_OP_SUB)
KERNEL(mul, BT_OP_MUL)
KERNEL(op_div, BT_OP_DIV)
KERNEL(relu, BT_OP_RELU)
KERNEL(sigmoid, BT_OP_SIGMOID)
KERNEL(op_tanh, BT_OP_TANH)
KERNEL(op_exp, BT_OP_EXP)
KERNEL(op_log, BT_OP_LOG)

#define REG(op, fn) bt_register_binary(op, BT_BACKEND_VULKAN, BT_FLOAT32, fn)

bt_status_t bt_vk_register(void) {
  bt_status_t s;
  if ((s = bt_vk_init())) return s;
  if ((s = REG(BT_OP_ADD, add))) return s;
  if ((s = REG(BT_OP_SUB, sub))) return s;
  if ((s = REG(BT_OP_MUL, mul))) return s;
  if ((s = REG(BT_OP_DIV, op_div))) return s;
  if ((s = REG(BT_OP_RELU, relu))) return s;
  if ((s = REG(BT_OP_SIGMOID, sigmoid))) return s;
  if ((s = REG(BT_OP_TANH, op_tanh))) return s;
  if ((s = REG(BT_OP_EXP, op_exp))) return s;
  if ((s = REG(BT_OP_LOG, op_log))) return s;
  if ((s = REG(BT_OP_FUSED_ADD_RELU, add_relu_kernel))) return s;
  if ((s = REG(BT_OP_MATMUL, matmul_kernel))) return s;
  if ((s = REG(BT_OP_TRANSPOSE, transpose_kernel))) return s;
  for (int k = 0; k < BT_CONV_KIND_COUNT; ++k) {
    s = bt_register_conv((bt_conv_kind_t)k, BT_BACKEND_VULKAN, BT_FLOAT32, conv_kernel);
    if (s != BT_OK) return s;
  }
  return BT_OK;
}
