#include "vk_internal.h"

#include <stdint.h>

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

  VkBuffer buffers[3] = {
    bt_vk_buffer_of(a->storage->gpu),
    bt_vk_buffer_of(b->storage->gpu),
    bt_vk_buffer_of(out->storage->gpu),
  };
  VkDescriptorBufferInfo info[3];
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < 3; ++i) {
    info[i] = (VkDescriptorBufferInfo){ .buffer = buffers[i], .offset = 0, .range = VK_WHOLE_SIZE };
    writes[i] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g_vk.set,
      .dstBinding = (uint32_t)i, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &info[i],
    };
  }
  vkUpdateDescriptorSets(g_vk.device, 3, writes, 0, NULL);

  bt_vk_begin_cmd();
  vkCmdBindPipeline(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pipeline);
  vkCmdBindDescriptorSets(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          g_vk.pipeline_layout, 0, 1, &g_vk.set, 0, NULL);
  vkCmdPushConstants(g_vk.cmd, g_vk.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                     0, sizeof(pc), &pc);
  // 256 lanes per workgroup; a single bounded dispatch keeps TDR out of the picture.
  vkCmdDispatch(g_vk.cmd, (uint32_t)((total + 255) / 256), 1, 1);
  bt_vk_barrier();
  bt_status_t status = bt_vk_flush();
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
  return REG(BT_OP_EXP, op_exp);
}
