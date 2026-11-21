#ifndef BITSY_SRC_BACKENDS_VULKAN_VK_INTERNAL_H
#define BITSY_SRC_BACKENDS_VULKAN_VK_INTERNAL_H

#include <stddef.h>
#include <bitsy/bt_error.h>
#include <bitsy/bt_tensor.h>

/* The core only sees opaque handles; Vulkan types stay inside this backend. */
bt_status_t bt_vk_init(void);
void bt_vk_shutdown(void);
int bt_vk_available(void);
const char* bt_vk_device_name(void);
bt_status_t bt_vk_register(void);
bt_status_t bt_vk_flush(void);
bt_status_t bt_vk_wait(void);
bt_status_t bt_vk_capture_begin(void);
bt_status_t bt_vk_capture_end(void);
bt_status_t bt_vk_replay(int count);
bt_status_t bt_vk_capture_sync_chunk(void);
bt_status_t bt_vk_sgd_step(const bt_tensor_t* w, const bt_tensor_t* grad,
                           float lr, bt_tensor_t* out);
bt_status_t bt_vk_reduce_sum(const bt_tensor_t* a, uint32_t axis, bt_tensor_t* out);
bt_status_t bt_vk_tanh_backward(const bt_tensor_t* g, const bt_tensor_t* t, bt_tensor_t* out);
bt_status_t bt_vk_broadcast_grad(const bt_tensor_t* g, bt_tensor_t* out);
bt_status_t bt_vk_dual_mul(const bt_tensor_t* g, const bt_tensor_t* a,
                           const bt_tensor_t* b, bt_tensor_t* out);
bt_status_t bt_vk_matmul_bias(const bt_tensor_t* a, const bt_tensor_t* b,
                              const bt_tensor_t* bias, bt_tensor_t* out);
bt_status_t bt_vk_log_softmax(const bt_tensor_t* x, bt_tensor_t* out);
bt_status_t bt_vk_ce_loss(const bt_tensor_t* ls, const bt_tensor_t* onehot, bt_tensor_t* out);
bt_status_t bt_vk_matmul_bias_tanh(const bt_tensor_t* a, const bt_tensor_t* b,
                                    const bt_tensor_t* bias, bt_tensor_t* out);
bt_status_t bt_vk_sgd_fused(const bt_tensor_t* w[4], const bt_tensor_t* grad[4],
                            const bt_tensor_t* out[4], float lr);
double bt_vk_device_time_ms(void);

void* bt_vk_alloc(size_t nbytes);
void bt_vk_free(void* handle);
bt_status_t bt_vk_upload(void* handle, const void* host, size_t nbytes);
bt_status_t bt_vk_readback(void* handle, void* host, size_t nbytes);
bt_status_t bt_vk_readback_offset(void* handle, void* host, size_t offset, size_t nbytes);
/* exp_014: deferred-transfer staging pool lifecycle. */
void bt_vk_stage_on_flush(void);
void bt_vk_stage_on_wait(void);
unsigned long bt_vk_stage_alloc_count(void);

/* `op` is the bt_op_id_t value from the dispatcher. */
bt_status_t bt_vk_elementwise(const bt_tensor_t* a, const bt_tensor_t* b,
                              bt_tensor_t* out, int op);

#endif /* BITSY_SRC_BACKENDS_VULKAN_VK_INTERNAL_H */
