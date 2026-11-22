# Bitsy 0.2 — compact core design (tinygrad-inspired)

This is the implementation blueprint derived from `TINYGRAD.md`, `PYTORCH.md`
and `VULKAN.md`. It is the design companion to `experiments/exp_040/`.

## Goal and hard constraints

- C11 + Vulkan core **≤ 4096 lines** (counted by `tools/count_core_lines.py`).
- Four tinygrad layers: lazy graph → fusion scheduler → parameterized kernels
  → capture/replay executor.
- Numeric parity within the existing tolerance table (never relax to hide a bug).
- Windows (MSVC) + Linux (gcc) build; GPU: GTX 1660 Super, Vulkan 1.3 baseline.
- Released as **0.2**, pushed to `main`.

## Layer map (new tree `src/rt/`, public header `include/bitsy_rt/bitsy_rt.h`)

```
frontend      bt_tensor_*       build lazy nodes; no compute, no dispatch
   |          bt_op_*           ADD/SUB/MUL/DIV/unary/MATMUL/REDUCE/TRANSPOSE
   v
graph         bt_graph          interned node {op, shape, src[], scalar}
   |          autograd           backward graph + grad accumulation (C)
   v
scheduler     bt_schedule       topo walk; fuse maximal pointwise chains -> 1 kernel
   |          buffer planning    assign arena slots; views are strides, not copies
   v
kernels       fused.comp         ONE shader, op-program in buffer (fusion)
              gemm.comp          ONE tiled shader, specialization constants
              reduce.comp        ONE shader, parameterized reduction
   v
executor      vk_*               device/arena/bindless/capture/replay
              cpu_*              reference fallback (same scheduler)
```

## Key simplifications that buy the line budget

1. **One op-program shader instead of 24 shaders.** A fused elementwise kernel
   receives an array of op-codes + scalar operands in a storage buffer and
   applies the whole pointwise chain per element in registers. Replaces:
   `elementwise, elementwise_vec4, dual_mul, fused, tanh_backward, gelu,
   sgd_step, sgd_fused, ce_loss, log_softmax, broadcast_grad, relu-bwd, ...`.
2. **One GEMM** with specialization constants (`BM,BN,BK`) instead of ten
   variants (`gemm_tiled, gemm_rb, gemm_rb_vec4*, gemm_bias*, swizzle, k16`).
3. **One reduce** parameterized by op + axis.
4. **Bindless device addresses** (core 1.2) in one SSBO → no per-op descriptor
   sets, no descriptor pool churn.
5. **Lazy graph owns autograd** → delete the Python per-op backward lambdas
   and the host `_unbroadcast` NumPy fallback.
6. **Capture/replay command buffer** → the JIT path is one `vkQueueSubmit2`
   with a timeline semaphore, not N submits.

## Data model

```c
typedef enum { BT_ADD, BT_SUB, BT_MUL, BT_DIV, BT_EXP, BT_LOG, BT_SQRT,
               BT_TANH, BT_SIGM, BT_RELU, BT_GELU, BT_MATMUL, BT_REDUCE_SUM,
               BT_TRANSPOSE, BT_NEG } bt_op;
typedef struct bt_node {
  bt_op op; int64_t shape[4]; int ndim;
  struct bt_node *src[3]; int nsrc;
  float scalar;                 // for constants / reduce / op-program
  uint64_t id;                  // hash-consed
} bt_node;
```
Views (transpose/reshape) carry strides; they only materialise when a kernel
consumes them. `bt_realize(t)` schedules + executes and returns a buffer handle.

## Fusion rule

From a sink node, walk the topological order. A **pointwise** node (unary/binary
elementwise, shape-preserving or broadcasting) is absorbed into the current
fused group if its inputs are already in the group or are leaves/buffers.
MATMUL/REDUCE are cut points (they start a new group and materialise a buffer).
Each group becomes one `fused.comp` dispatch with its op-program. This is the
tinygrad "one CALL = one kernel" rule and the main speed win (kills ~36 µs/op
submit+fence overhead).

## Vulkan executor (from VULKAN.md section 8)

- Vulkan 1.3; one compute queue; one `DEVICE_LOCAL(+HOST_VISIBLE)` arena with a
  bump allocator; `VkPhysicalDeviceBufferDeviceAddressFeatures`.
- One descriptor set: SSBO of `uint64_t` device addresses + lengths, bound once.
  Push constants carry shape/strides/op-program offset.
- Specialization constants: `BM,BN,BK` for GEMM; `WG` for workgroup size.
- Capture: record all dispatches + `vkCmdPipelineBarrier2` into one primary CB.
- Replay: `vkQueueSubmit2` with timeline wait `k`, signal `k+1`; lifetime by
  `vkGetSemaphoreCounterValue` ≥ tag. Fence only at host readback.
- Eager fallback: `bt_flush` = one submit + fence wait (keeps parity harness API).

## Numeric contract

- Keep the EXACT/APPROX split from `tolerances.py`.
- GELU = exact erf (Abramowitz-Stegun 7.1.26, as in exp_037) → ≤1.2e-7.
- sqrt clamps negatives at 0 (documented divergence).
- matmul FP32 tiled; accumulation order differs from cuBLAS → APPROX 1e-5.
- All backward formulas mirror the proven ones in `autograd.py`.

## Staged rollout (honest about what is done each turn)

- R1: graph + scheduler in C; fused/GEMM/reduce kernels; CPU fallback.
- R2: Vulkan bindless executor + eager flush (parity on the op subset).
- R3: autograd in C; parity on gradient subset.
- R4: capture/replay JIT; benchmark vs legacy eager + PyTorch.
- R5: reach full 48-case parity, remove legacy from build, tag 0.2, push main.

Legacy `src/` is retained as a numeric reference until R5; the line counter
reports legacy and compact-core totals separately so the budget is honest.
