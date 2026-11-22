# Getting Vulkan compute as fast as CUDA — techniques, evidence, benchmarks

A research-backed guide for the v0.2 C+Vulkan core. Target GPU: **NVIDIA GTX
1660 Super (TU116, Turing)**, fp32. Goal (from the dashboard): speed ratio
`torch_ms / bitsy_ms` → 1.00 for every op and model step.

The core thesis, from tinygrad's own docs: *"tinygrad has the best execution
speed of any framework because it usually bypasses the GPU driver and prebuilds
the command queue. It's tons faster than normal CUDA, and often even faster than
CUDA Graph."* Vulkan can match CUDA because the expensive part of small compute
is **CPU/driver submission overhead**, not GPU math — and Vulkan gives explicit
control over that overhead.

## 1. Evidence that Vulkan compute can match CUDA (with numbers)

- **Same algorithm, Vulkan vs CUDA backend — VkFFT on RTX 3070** [2]: Bluestein
  N=2039 batch=8192 → CUDA 1.13 ms (220 GB/s) vs **Vulkan 0.98 ms (255 GB/s)**;
  Stockham N=2048 batch=8192 → CUDA 0.65 ms vs **Vulkan 0.66 ms** (within 2%).
  The API is not the limiter — the algorithm is. On A100/MI300X VkFFT matches
  cuFFT for small sequences and beats it for large ones [3].
- **SGEMM ladder on RTX 5070 Ti** [11]: a hand-written Vulkan GEMM reached
  **105.7% of the equivalent CUDA kernel** and 81.7% of cuBLAS after double
  buffering (26,690 GFLOP/s). Key detail: `glslc` emitted scalar LDS loads where
  `nvcc` emitted `LDS128`; rewriting to explicit `vec4` shared loads recovered
  from 64.3% → 99.9% of CUDA. Compiler-friendly code shape matters.
- **Compute ALU loop, RTX 3070** [12]: wgpu(Vulkan) 7,296 GFLOP/s vs CUDA 7,454
  GFLOP/s — ~2% apart. **LLM inference** [13]: Vulkan within a few percent of
  CUDA on GB10 and sometimes faster (prompt 2355 CUDA vs **2395 Vulkan** tok/s).
- **Dispatch overhead dominates**: cutting dispatches 876→564 improved LLM TTFT
  **53% (71.4→41.6 ms)** because submit was ~40% of CPU time at 12.9 µs/dispatch
  [73]. A pre-recorded command buffer had **4–11× lower submission CPU time**
  than the best CUDA method with negligible jitter [5].
- **Where the gap really is** [4][15]: hand-written Vulkan GEMMs measured 30–157×
  vs cuBLAS by host stopwatch, but GPU timestamps (`VkQueryPool`) showed the
  *kernel* gap was 10–22× — "a normal hand-written-vs-vendor gap, not something
  structurally broken." Other measured gaps: host-dequantized F32 weights
  (Vulkan path got F32 while coopmat needs F16) and an allocator destroying
  dedicated blocks per dispatch caused a **7×** elementwise cliff, fixed by
  pooling. **Methodology rule: always profile with `vkCmdWriteTimestamp2`, not a
  host stopwatch.**
- Caveat for our GPU (TU116): **no tensor cores, no TF32**, 336 GB/s HBM, subgroup
  size 32 [7][8]. "As fast as CUDA" means matching CUDA's mature scalar kernels
  and saturating bandwidth, not tensor-core throughput. On GTX 1660 Ti, Vulkan
  was 2.1× faster on prompt but CUDA 33% faster on generation [70] — so expect a
  remaining deficit on scalar GEMM and document it honestly.

## 2. The four speed axes (tinygrad's model, applied)

1. **Execution speed (driver)** — biggest lever for our small ops. Fix with
   capture/replay (§3).
2. **Model speed (scheduler)** — fusion; one CALL = one kernel; recompute vs
   store decisions. Fix with the fusion scheduler (`C_VULKAN_CORE.md`).
3. **Kernel speed (codegen)** — tiling, arithmetic intensity, tensor cores.
   Fix with GEMM specialization + autotune (§5).
4. **Compile speed** — not a runtime metric; keep kernel cache keyed by shape.

## 3. Driver-bypass: the CUDA-graph-equivalent in Vulkan

The per-step CPU work must shrink to "write args + one submit + optional poll".

| Technique | Vulkan primitive | Effect |
|---|---|---|
| Prebuild the command queue once | one primary `VkCommandBuffer` recorded at JIT capture | zero `vkCmd*` per step |
| Replay with fresh sync value | `vkQueueSubmit2` + **timeline semaphore** (`wait=k`, `signal=k+1`) | no fence per step; lifetime by counter |
| Stable addresses | **buffer device address** (core 1.2): one SSBO of `uint64_t` addresses, index by push-constant ID | no descriptor rewrites per dispatch |
| Push constants for shape/strides/op-id | `VkPushConstantRange` (≤128 B) | args live in the command stream, not memory |
| Push descriptors for fast-changing bindings | `VK_KHR_push_descriptor` / 1.4 `vkCmdPushDescriptorSetKHR` | avoid descriptor pool/set churn (≤32 descriptors) |
| Shader objects (optional) | `VK_EXT_shader_object`, `vkCreateShadersEXT`/`vkCmdBindShadersEXT` | bind one compiled stage without `VkPipeline` combos; good when generating many small variants |
| Pipeline binary capture (optional) | `VK_KHR_pipeline_binary` | serialize compiled pipelines → instant JIT start |
| Fence only at readback | `VkFence` at true host sync only; `vkWaitSemaphores` for progress | no `vkDeviceWaitIdle` bubbles |

Replay loop (from the spec-level design in `../research/VULKAN.md`):
```c
// once: record graph into VkCommandBuffer cb; timeline tl (initial 0); tick=0
// per step:
write_scalars_to_scratch_or_push();                // mapped HOST_COHERENT, or baked
VkSemaphoreSubmitInfo wait={.semaphore=tl,.value=tick,  .stageMask=ALL_COMMANDS};
VkSemaphoreSubmitInfo sig ={.semaphore=tl,.value=tick+1,.stageMask=ALL_COMMANDS};
VkCommandBufferSubmitInfo cbi={.commandBuffer=cb};
VkSubmitInfo2 s={.waitSemaphoreInfoCount=1,.pWaitSemaphoreInfos=&wait,
  .commandBufferInfoCount=1,.pCommandBufferInfos=&cbi,
  .signalSemaphoreInfoCount=1,.pSignalSemaphoreInfos=&sig};
vkQueueSubmit2(queue,1,&s,VK_NULL_HANDLE); tick++;
```
Caveat: prebuilt replay needs static shapes/control flow; key captured graphs by
shape (exactly as tinygrad's JIT does) and keep referenced objects alive.

## 4. Fusion (scheduler) — kill per-op dispatch

A pointwise chain `((a*b)+c).relu()` is **one** dispatch, not four: load a,b,c
into registers, multiply, add, relu, store once. This removes ~36 µs/op of
submit+fence overhead that dominated the legacy core's elementwise ratios
(0.41–0.53). Implementation: `bt_build_fusion` in `C_VULKAN_CORE.md`; proof via
`bt_dispatch_count()==1` for an N-op chain (unit test).

Recompute-vs-store (tinygrad example `a+b+c` and `a+b+d`): recompute `a+b` in
both kernels = 6 reads + 2 writes vs materializing = 6 reads + 3 writes, so
recompute wins. The scheduler should prefer folding over buffering when a node
has one consumer.

## 5. Kernel speed — memory is the bottleneck

From tinygrad's speed doc: *"The main bottleneck in most kernels is accessing
memory … arithmetic intensity"*. On a 4090 (1 TB/s HBM, ~160 TFLOP/s) you need
~100 reuses per HBM load, ~4 per L1 load. TU116 is memory-bound for elementwise
and compute-bound for large GEMM without tensor cores.

| Technique | How | Where |
|---|---|---|
| Tiled GEMM + specialization constants | BM/BN/BK compile-time → register allocation/unrolling; autotune a few tuples | `gemm.comp` |
| Register blocking / UPCAST | split an axis, keep several accumulators per thread → reuse loaded A/B | GEMM |
| Shared-memory (workgroup) tiling | vectorized load tile into `shared[]`, `barrier()`+`memoryBarrierShared()` between phases | GEMM/conv |
| Subgroup ops | `gl_SubgroupSize`, `subgroupAdd/Reduce` for warp-level reductions without shared memory | reduce/softmax |
| vec4 coalesced loads | `std430` SSBO, consecutive lanes → consecutive addresses; `readonly`/`writeonly`/`restrict` | fused |
| Avoid divergence | `mix`/`?:` instead of divergent `if`; full subgroups | all |
| Simplify index math | hoist address expressions; specialization constants for strides | codegen |
| BEAM-lite autotune | compile candidate tile configs, time on device, keep fastest | optional phase 3 |

**Tensor cores / cooperative matrix**: `VK_KHR_cooperative_matrix` spreads a
medium matrix across a subgroup (`coopmat`, `coopMatLoad/MulAdd/Store`) and is
the clean path to tensor-core GEMM — **but TU116 has no tensor cores** exposed,
so v0.2 GEMM stays fp32 tiled and this is an optional fast path for Ampere+.
Document the limit honestly rather than claiming tensor-core speed.

## 6. Memory & bandwidth

- Prefer `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` (BAR/UMA) for upload-heavy
  weights; pure `DEVICE_LOCAL` for large tensors; `HOST_VISIBLE|COHERENT` for
  staging. Host-coherent needs no flush/invalidate.
- One arena per heap class + bump allocator (align to `requirements.alignment`
  and `nonCoherentAtomSize`); free by reset at graph teardown. Avoid per-tensor
  `vkAllocateMemory`.
- Buffer device address (core 1.2) → bindless; descriptor indexing for large
  tables; `VK_EXT_memory_budget` to avoid overcommit.
- Double/triple-buffer the staging ring so uploads overlap compute.
- `vkCmdPipelineBarrier2` (sync2) compute→compute; avoid `queueWaitIdle`.

## 7. Benchmarks to build (methodology in `../BENCHMARK_METHODOLOGY.md`)

| Benchmark | Measures | Success criterion |
|---|---|---|
| `ops/elementwise.py` | per-op wall + device time (timestamp queries), 60 iters, warmup | ratio trend up from 0.41–0.53 |
| `ops/gemm.py` | 256²/512²/1024²; tiles autotuned; clock logged | trend toward 0.71→1.0 |
| `models/step.py` | full + per-step (fwd/bwd/update) for MLP/RNN/LSTM/GPT-2 | per-step breakdown localizes slow stage |
| `perf/replay_vs_eager.py` | captured replay vs eager vs PyTorch CUDA graph | replay ≫ eager; approach CUDA graph |
| `perf/dispatch_count.py` | dispatches per chain (fusion proof) | 1 for pointwise chain |

Every benchmark: warmup, correct sync (`bt_wait` / `cuda.synchronize`), median +
spread, GPU clock, dataclass config, JSON result. The dashboard
(`docs/progress.json`) is updated from these results only.

## 8. Clever tricks checklist (adopt where applicable)

- [x] Prebuilt command queue + timeline replay (driver bypass) — phase 3
- [x] Bindless device addresses in one SSBO — phase 1/3
- [x] Push constants for shape/op-id — phase 1
- [x] One fused op-program kernel instead of per-op shaders — phase 2
- [x] Specialization constants + BEAM-lite tile autotune — phase 3
- [ ] Push descriptors / shader objects for many-small-variant dispatch — optional
- [ ] Subgroup warp reductions — phase 3
- [ ] Cooperative matrix GEMM — optional (not on TU116)
- [ ] Multi-queue async compute — optional
- [ ] On-chip SRAM staging / L2-aware tiling — optional
- [ ] Pipeline binary capture for instant JIT — optional

## Sources

Cross-API evidence: [1] https://github.com/vincefn/pyvkfft · [2]
https://vulkan.org/user/pages/09.events/vulkanised-2023/vulkanised_2023_vkfft_a_story_of_vulkan_compute_gpu_hpc_library_development.pdf
· [3] https://sc22.supercomputing.org/proceedings/tech_poster/poster_files/rpost143s3-file2.pdf
· [10] https://api-depositonce.tu-berlin.de/server/api/core/bitstreams/e4e143e2-9da2-4112-aa09-cd0576355e03/content
· [11] https://github.com/hangyugit/GEMM_Vulkan · [12]
https://github.com/appsjuragan/rust-wgpu-bench · [13]
https://forums.developer.nvidia.com/t/vulkan-as-alternative-backend-for-llama-cpp/363516
· [70] https://www.cryptoprofi.info/?p=19608.

Gap analysis / profiling: [4] https://github.com/HartsyAI/HartsyInference/blob/main/benchmarks/scoreboards/VULKAN.md
· [15] same scoreboard (allocator/coopmat/fallback findings) · [18]
https://docs.vulkan.org/guide/latest/profiling.html · [73]
https://arxiv.org/html/2608.08730v3 · [16]
https://github.com/ggml-org/llama.cpp/issues/17273.

Driver/dispatch: [5] https://drops.dagstuhl.de/storage/00lipics/lipics-vol133-ecrts2019/LIPIcs.ECRTS.2019.22/LIPIcs.ECRTS.2019.22.pdf
· [6] https://zolotukhin.ai/blog/2026-04-25-why-one-vkqueuesubmit-per-prompt-is-the-next-quiet-rdna4-prefill-unlock/
· [19] https://developer.nvidia.com/blog/vulkan-dos-donts · [20]
https://docs.vulkan.org/samples/latest/samples/performance/command_buffer_usage/README.html
· [22] https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdPushDescriptorSetKHR.html
· [23][24][25] descriptor buffer: https://www.khronos.org/blog/vk-ext-descriptor-buffer
· [27] BDA: https://docs.vulkan.org/tutorial/latest/Advanced_Vulkan_Compute/06_Advanced_Data_Structures/04_device_addressable_buffers.html
· [29] bindless: https://docs.vulkan.org/samples/latest/samples/extensions/descriptor_indexing/README.html
· [31] shader objects: https://www.khronos.org/blog/you-can-use-vulkan-without-pipelines-today
· [34] pipeline cache: https://www.khronos.org/blog/bringing-explicit-pipeline-caching-control-to-vulkan
· [37][38] timeline semaphores: https://docs.vulkan.org/tutorial/latest/Advanced_Vulkan_Compute/08_Asynchronous_Compute/03_timeline_semaphores.html
· [41][42] tinygrad HCQ: https://docs.tinygrad.org/developer/hcq
· [44] device-generated commands: https://docs.vulkan.org/spec/latest/chapters/device_generated_commands/generatedcommands.html.

Kernel optimization: [47] https://docs.vulkan.org/tutorial/latest/ML_Inference/Vulkan_Compute_for_ML/04_matrix_operations.html
· [49] https://www.khronos.org/blog/vulkan-subgroup-tutorial · [54] VkFFT shared-memory design
https://community.khronos.org/t/new-vulkan-fft-library-vkfft-open-source-crossplatform-faster-than-cufft/106192
· [57][58] coopmat2: https://vulkan.org/user/pages/09.events/vulkanised-2025/T47-Jeff-Bolz-NVIDIA.pdf
· [59] coopmat flash attention: https://github.com/ggml-org/llama.cpp/pull/13506.

Memory / TU116: [62] https://docs.vulkan.org/spec/latest/chapters/memory.html · [67]
https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_memory_budget.html
· [7] https://www.nvidia.com/en-me/geforce/graphics-cards/compare · [8]
https://gpuspecs.com/card/nvidia-geforce-gtx-1660-super-6gb.

Foundational: tinygrad https://docs.tinygrad.org/developer/speed/ · PyTorch 2.15
https://docs.pytorch.org/docs/2.15/community/design.html · Vulkan spec
https://docs.vulkan.org/spec/latest/index.html · VkFFT https://github.com/DTolm/VkFFT
(white paper https://ieeexplore.ieee.org/document/10036080).
