# Study: Vulkan 1.4 spec — compute performance for a compact runtime

Source (read 2026-11-22): https://docs.vulkan.org/spec/latest/index.html
(Khronos Vulkan Specification, 1.4.x revision), plus Vulkan feature descriptions
and GLSL extension specs at docs.vulkan.org, and tinygrad's driver rationale.

This is a study record for Bitsy 0.2. Terminology: the handle is `VkFence`
(there is no `VK_FENCE` token — that is OpenGL).

## 1. Minimum viable feature set

| Capability | Core version / extension | Enable via |
|---|---|---|
| Compute pipeline, SPIR-V, push/specialization constants, descriptor sets, command buffers, submit | **1.0** | — |
| Subgroups in GLSL (`GL_KHR_shader_subgroup`) | 1.1 | query `VkPhysicalDeviceSubgroupProperties` |
| Timeline semaphores | **1.2** | `timelineSemaphore` |
| Buffer device address (bindless pointers in shaders) | **1.2** | `bufferDeviceAddress` |
| Descriptor indexing / bindless arrays | 1.2 | `VkPhysicalDeviceDescriptorIndexingFeatures` |
| `synchronization2` (`vkCmdPipelineBarrier2`) | **1.3** | `synchronization2` |
| Push descriptors | 1.4 | `maxPushDescriptors` |
| Subgroup size control | ext `VK_EXT_subgroup_size_control` | feature bit |
| Cooperative matrix (tensor cores for GEMM) | ext `VK_KHR_cooperative_matrix` | `cooperativeMatrix` |
| Cooperative vector (per-lane MLP/mat-vec) | ext `VK_NV_cooperative_vector` | feature bit |
| 16/8-bit storage | 1.2; math via `VK_KHR_shader_float16_int8` / `VK_KHR_shader_bfloat16` | feature bits |

**Decision: target Vulkan 1.3 baseline** (sync2, timelines, device addresses,
descriptor indexing all in core). Cooperative matrix / vector are optional fast
paths. We are currently on a GTX 1660 Super (Turing) — no tensor cores exposed
via cooperative matrix, so GEMM stays FP32 tiled; the design still allows a
cooperative-matrix path on Ampere+.

## 2. Compute pipeline — the minimal path

1. `vkCreateShaderModule` from SPIR-V words.
2. `VkDescriptorSetLayout` (bindings, `VK_SHADER_STAGE_COMPUTE_BIT`).
3. `VkPipelineLayout` (set layouts + `VkPushConstantRange`).
4. `vkCreateComputePipelines` with one `VkPipelineShaderStageCreateInfo`; attach
   `VkSpecializationInfo` for compile-time tile sizes.
5. Descriptor pool + sets; then `vkCmdBindPipeline` → `vkCmdBindDescriptorSets`
   → `vkCmdPushConstants` → `vkCmdDispatch`.

**Argument channels (pick by lifetime):**
- **Push constants**: per-dispatch scalars, shape/strides, one `VkDeviceAddress`
  (8 B). Fastest — in the command stream. Min 128 B total, multiple of 4.
- **Specialization constants**: compile-time tiles, workgroup size, op IDs,
  dtype switches → enable unrolling/register allocation. One pipeline per tuple.
- **Descriptors**: buffers used by many dispatches; bindless tables.
- **Push descriptors (1.4)**: a few fast-changing bindings without pool churn.

`VkSpecializationInfo` wiring: `VkSpecializationMapEntry{constantID,offset,size}`
+ data blob. Use a shared `VkPipelineCache`.

## 3. Descriptors — go bindless, avoid per-op churn

Classic model: create one large pool at init, allocate all sets once, update
once, never free. Descriptor churn is what we avoid.

**Bindless (recommended):** one SSBO holding `uint64_t` `VkDeviceAddress`
values + a length array; bind that single descriptor once; index from
push-constant IDs with GLSL `buffer_reference2` (`GL_EXT_buffer_reference2`).
Requires buffer device address (core 1.2) — allocate memory with
`VkMemoryAllocateFlagsInfo{ VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT }`, buffers
with `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT`, query with
`vkGetBufferDeviceAddress`. This gives arbitrary "many buffers to one kernel"
with **no per-tensor descriptor** — a large simplification for a tiny runtime.

`VK_EXT_descriptor_buffer` (D3D12-style explicit descriptor memory) is an
alternative but not needed in v1. Dynamic descriptors (`*_DYNAMIC`) cannot
combine with update-after-bind/push descriptors and cost a bind-time offset —
a push constant address is cheaper.

## 4. Synchronization — four primitives for compute

| Primitive | Scope | Use |
|---|---|---|
| `VkFence` | host ↔ device | CPU waits for GPU; safe CB/memory reuse. Never inside a CB. |
| `VkSemaphore` (binary) | queue ↔ queue | GPU ordering across queues. |
| `VkSemaphore` (timeline) | anywhere, with `uint64_t` values | progress, lifetime, wait-before-signal. Core 1.2. |
| Pipeline barriers | one queue, at a point | the workhorse between dispatches. |

- **Compute→compute barrier (sync2)**: `VkMemoryBarrier2` +
  `VkDependencyInfo` with `srcStage=COMPUTE_SHADER, srcAccess=SHADER_STORAGE_WRITE,
  dstStage=COMPUTE_SHADER, dstAccess=SHADER_STORAGE_READ|WRITE`,
  `vkCmdPipelineBarrier2`. Prefer `VkBufferMemoryBarrier2` for specific buffers.
  Inside a workgroup, GLSL `barrier()` + `memoryBarrierShared()`; not API barriers.
- **Host/device**: `HOST_COHERENT` needs no flush/invalidate; non-coherent needs
  `vkFlushMappedMemoryRanges` / `vkInvalidateMappedMemoryRanges` aligned to
  `nonCoherentAtomSize`. End with barrier to/from `PIPELINE_STAGE_2_HOST_BIT`.
- Avoid `vkQueueWaitIdle`/`vkDeviceWaitIdle` — they serialize everything.
- **Timeline semaphore = master clock**: tag each resource with timeline value
  of last use; reclaim when `vkGetSemaphoreCounterValue(tl) >= tag`. Enables
  wait-before-signal (submit a waiter before the signaler is submitted) and the
  prebuilt-queue replay model.

## 5. Memory

- `vkGetPhysicalDeviceMemoryProperties2` → types/heaps. Canonical `findType`
  linear scan over `memoryTypeCount` matching `propertyFlags`.
- Prefer `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` (BAR/UMA) for upload-heavy
  weights; pure `DEVICE_LOCAL` for large long-lived tensors; `HOST_VISIBLE|
  COHERENT` for staging.
- **Bump/suballocator**: one `vkAllocateMemory` per heap class, then bump-align
  offsets (`VkMemoryRequirements::alignment`, `nonCoherentAtomSize`). Free by
  resetting the arena at graph teardown. `VK_EXT_memory_budget` avoids overcommit.
- **Dedicated allocation** (1.1) for large fixed tensors via
  `VkMemoryDedicatedRequirements` + `VkMemoryDedicatedAllocateInfo`.
- **Staging**: map HOST_VISIBLE, memcpy, `vkCmdCopyBuffer`, barrier TRANSFER_WRITE
  → SHADER_STORAGE_READ. Skip staging on BAR/UMA. Double/triple-buffer the ring.

## 6. Kernel performance

- Workgroup size = multiple of subgroup/warp (32 NVIDIA, 32/64 AMD/Intel),
  typically 64–256. Query `maxComputeWorkGroupInvocations`, `maxComputeSharedMemorySize`.
- Occupancy limited by registers + workgroup shared memory; too much UPCAST
  spills. tinygrad's numbers: ~1 TB/s HBM + ~160 TFLOP/s → ~100 reuses from HBM,
  ~4 from L1. Tile toward that.
- **Subgroups** (1.1): `gl_SubgroupSize`, `subgroupAdd/Shuffle/Ballot/Reduce` —
  warp-level reductions without shared memory.
- **Cooperative matrix** (ext): `coopmat<float, gl_ScopeSubgroup, BM, BN, ...>`,
  `coopMatLoad/MulAdd/Store`; M/N/K can be specialization constants. Rows/cols
  spread across the scope; all invocations active with uniform params.
- **Cooperative vector** (ext): `coopVecMatMulAddNV` for per-lane small MLPs;
  weights implicitly loaded from memory; pre-transform with
  `vkConvertCooperativeVectorMatrixNV`.
- GLSL: coalesced `vec4` loads from `std430` SSBOs; `readonly`/`writeonly`/
  `restrict`; avoid divergence (use `mix`/`?:`, subgroup ballot); hoist index
  math; FP16/bfloat16 for bandwidth-bound, FP32 accumulate.

## 7. The tinygrad execution model in Vulkan primitives

The "best execution speed ... bypasses the GPU driver and prebuilds the command
queue" claim is ordinary Vulkan architecture, not an exotic extension:

1. **Capture once**: compile model to fixed kernels; record all dispatches,
   binds, barriers into one/few primary command buffers. No `vkCmd*` per step.
2. **Stable resources**: fixed arena slots; descriptors/addresses written once;
   changing scalars via push constants or a mapped coherent scratch UBO.
3. **Replay**: one `vkQueueSubmit2` per step with fresh timeline values
   (`wait=k`, `signal=k+1`).
4. **Bypass**: per-step CPU = write args + one submit + optional poll.
5. **Lifetime via timeline**, not fences: reclaim when counter ≥ tag.

Minimal replay loop (pseudo-C, from the study):
```c
// once: VkCommandBuffer graph = record_once(...); VkSemaphore tl (timeline,0); uint64_t tick=0;
// per step:
write_push_constants_or_scratch(next_args);
VkSemaphoreSubmitInfo wait = { .semaphore=tl, .value=tick,   .stageMask=ALL_COMMANDS };
VkSemaphoreSubmitInfo sig  = { .semaphore=tl, .value=tick+1, .stageMask=ALL_COMMANDS };
VkCommandBufferSubmitInfo cbi = { .commandBuffer=graph };
VkSubmitInfo2 sub = { .waitSemaphoreInfoCount=1, .pWaitSemaphoreInfos=&wait,
  .commandBufferInfoCount=1, .pCommandBufferInfos=&cbi,
  .signalSemaphoreInfoCount=1, .pSignalSemaphoreInfos=&sig };
vkQueueSubmit2(queue, 1, &sub, VK_NULL_HANDLE);
tick++;
```

Caveats: static shapes/control flow required; multiple captured graphs keyed by
shape otherwise; referenced objects must outlive the graph; changing data is
free, changing addresses needs a scratch address table or descriptor buffers.

## 8. Decisions for Bitsy 0.2

1. Vulkan 1.3 baseline; one compute queue; one DEVICE_LOCAL arena + bump allocator.
2. One bindless SSBO of device addresses; push constants for shapes/strides/IDs.
3. One primary command buffer per captured graph; `vkCmdPipelineBarrier2`
   between dependent kernels.
4. Timeline semaphore for replay/lifetime; fences only at true host sync.
5. Specialization constants for tile sizes; cooperative matrix as optional path.
6. Keep the current `bt_flush`/`bt_wait` host fence as the eager fallback and
   add capture/replay for the JIT path — so the parity harness keeps working.
7. GT 1660 Super has no cooperative-matrix path: GEMM stays tiled FP32, and we
   report that honestly rather than claiming tensor-core performance.
