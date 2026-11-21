#include "vk_internal.h"

#include <stdint.h>
#include <stdlib.h>

#include "../../core/conv_dispatch_internal.h"
#include "../../core/dispatcher_internal.h"
#include "../../core/error_internal.h"
#include "../../core/storage_internal.h"
#include "../../core/tensor_internal.h"
#include "vk_device_internal.h"

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

/* One descriptor set is enough: every dispatch is serialized by the fence. */
static bt_status_t dispatch(VkPipeline pipeline, VkBuffer buffers[4],
                            const void* pc, size_t pc_size, uint32_t gx, uint32_t gy) {
  VkDescriptorBufferInfo info[4];
  VkWriteDescriptorSet writes[4];
  for (int i = 0; i < 4; ++i) {
    info[i] = (VkDescriptorBufferInfo){ .buffer = buffers[i], .offset = 0, .range = VK_WHOLE_SIZE };
    writes[i] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g_vk.set,
      .dstBinding = (uint32_t)i, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &info[i],
    };
  }
  vkUpdateDescriptorSets(g_vk.device, 4, writes, 0, NULL);

  bt_vk_begin_cmd();
  vkCmdBindPipeline(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          g_vk.pipeline_layout, 0, 1, &g_vk.set, 0, NULL);
  vkCmdPushConstants(g_vk.cmd, g_vk.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                     0, (uint32_t)pc_size, pc);
  vkCmdDispatch(g_vk.cmd, gx, gy, 1);
  bt_vk_barrier();
  return bt_vk_flush();
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
  // 256 lanes per workgroup; a single bounded dispatch keeps TDR out of the picture.
  bt_status_t status = dispatch(g_vk.pipeline, buffers, &pc, sizeof(pc),
                                (uint32_t)((total + 255) / 256), 1);
  if (status == BT_OK) bt_storage_bump_version(out->storage);
  return status;
}

typedef struct {
  uint32_t M, N, K;
  int32_t sa0, sa1, sb0, sb1, sc0, sc1;
  int32_t offa, offb, offc;
} vk_matmul_pc_t;

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
  };
  VkBuffer buffers[4] = {
    bt_vk_buffer_of(a->storage->gpu), bt_vk_buffer_of(b->storage->gpu),
    bt_vk_buffer_of(out->storage->gpu), bt_vk_buffer_of(a->storage->gpu),
  };
  static int tiled = -1;
  if (tiled < 0) {
    const char* env = getenv("BT_GEMM_TILED");
    tiled = env && atoi(env) ? 1 : 0;
  }
  VkPipeline pipeline = tiled ? g_vk.gemm_tiled_pipeline : g_vk.matmul_pipeline;
  uint32_t tile = tiled ? 16 : 8;
  bt_status_t status = dispatch(pipeline, buffers, &pc, sizeof(pc),
                                (pc.N + tile - 1) / tile, (pc.M + tile - 1) / tile);
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
                                (uint32_t)((pc.n + 255) / 256), 1);
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
                                (uint32_t)((pc.n + 255) / 256), 1);
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
  if ((s = REG(BT_OP_MATMUL, matmul_kernel))) return s;
  if ((s = REG(BT_OP_TRANSPOSE, transpose_kernel))) return s;
  for (int k = 0; k < BT_CONV_KIND_COUNT; ++k) {
    s = bt_register_conv((bt_conv_kind_t)k, BT_BACKEND_VULKAN, BT_FLOAT32, conv_kernel);
    if (s != BT_OK) return s;
  }
  return BT_OK;
}
