# GAP_LEDGER — ranked performance gaps

**Phase:** TOGISHI Phase 2 (profiling only; no framework optimization yet)
**Evidence:** `benchmarks/results/three_way.json`, `benchmarks/results/torch_kernels.json`,
per-op Chrome traces `benchmarks/results/trace_*.json`, `experiments/exp_007/`.

Method: 50 iterations, warm-up 5, median reported. PyTorch CUDA is given
`cudnn.benchmark = True`. Bitsy wall time includes `bt.flush() + bt.wait()`
per iteration; PyTorch wall time includes `torch.cuda.synchronize()`. Bitsy
GPU time is the sum of per-dispatch timestamp queries — a **kernel proxy**, not
a 1000-repeat isolated measurement (that distinction is preserved in the data).

Calibrated ceilings from Phase 1: DRAM **304.4 GB/s**, FP32 **5.49 TFLOP/s**
@1950 MHz, submit+fence **~36 µs**, allocate+free **~363 µs**, empty dispatch
**~3.1 µs**.

## Ranked ledger

| Rank | Op | Wall gap (µs) | Bitsy GPU/dispatch (µs) | PyTorch kernel (µs/call) | Classification | Evidence |
|---:|---|---:|---:|---:|---|---|
| 1 | matmul 256² | **155.5** | 165.8 | 18.0 (`volta_sgemm_128x32_nn`) + 5.1 splitK | compute/method + wave quantization | `trace_matmul_prof.json`, 3.7% of FP32 peak (exp_008 improved 1024² 3.54× but the three-way 256² case still uses tiled) |
| 2 | mul 1M | **52.8** | 52.2 | 46.1 (`vectorized_elementwise_kernel<4>`) | bandwidth + launch/sync | 79.1% of DRAM ceiling; host overhead 52.8 µs |
| 3 | relu 1M | **37.0** | 48.9 | 33.3 (`vectorized_elementwise_kernel<4>`) | launch/sync + bandwidth | 84.6% of DRAM ceiling |
| 4 | add 1M | **12.9** | 48.4 | 42.3 (`vectorized_elementwise_kernel<4>`) | launch/sync | 85.5% of DRAM ceiling |

## Per-op profiler counters (50 iterations)

| Op | dispatches | barriers | submits | allocations | cache hits | profiler overhead (µs) |
|---|---:|---:|---:|---:|---:|---:|
| add | 50 | 50 | 50 | 0 | 50 | +2.8 |
| mul | 50 | 50 | 50 | 0 | 50 | −21.0 (noise) |
| relu | 50 | 50 | 50 | 0 | 50 | +4.2 |
| matmul | 50 | 50 | 50 | 0 | 50 | +15.2 |

The caching allocator is working: **0** `vkAllocateMemory` calls in 50 steps,
all 50 allocations served from the size-class cache. One submit + one barrier
per iteration remains because the benchmark explicitly waits each iteration.

## PyTorch kernel parameters to copy

| Op | PyTorch kernel | Notes |
|---|---|---|
| add / mul | `at::native::vectorized_elementwise_kernel<4, ...>` | vec4 loads/stores; Bitsy uses scalar float per lane |
| relu | `vectorized_elementwise_kernel<4>` via `clamp_min` | vec4 |
| matmul | `volta_sgemm_128x32_nn` + `splitKreduce_kernel<32,16>` | 128×32 tile, split-K reduction; Bitsy uses naive/16×16 tiled SGEMM at 3.7% of peak |

## Interpretation / next fixes (ranked)

1. **GEMM (155.5 µs):** implement the GEMM ladder from the campaign —
   register-blocked 128×128×8 / 128×64×8 tiles, vec4 + conflict-free transposed
   A in shared memory, double buffering, split-K, L2-friendly block order,
   autotune cached per device. This is the single largest gap.
2. **Elementwise vectorization (~50 µs combined):** use `vec4` loads/stores and
   remove the unconditional global barrier between independent dispatches;
   target ≥90% of measured DRAM BW (currently 79–86%).
3. **Submission overhead (~14–53 µs):** the benchmark forces a wait per
   iteration; the profiler shows one submit and one barrier per dispatch.
   Overlap host work across iterations (timeline semaphore already present) and
   use synchronization2 with hazard-tracked minimal barriers.
4. **Allocator:** already solved for the steady state (0 real allocations in
   50 steps); keep the cache, add stream-ordered reuse and memory-budget
   awareness in Phase 4.

## Phase 5 update (exp_011)

- GPU clock is now logged per benchmark run; at a fixed 1530 MHz the GEMM ratio
  is 0.71 and elementwise ratios are 0.34–0.54. The earlier 0.69–0.78 GEMM
  spread was partly 1530–1950 MHz clock variance.
- Hazard-tracked per-buffer barriers replaced the unconditional global barrier,
  but a 20-add dependency chain still needs 19 barriers for 20 dispatches
  (only the first is avoided). The win is narrower barriers, not fewer.
- Submitting 8 chains before one wait improves wall time 1.214 → 1.107 ms per
  chain (1.10×), below the 1.2× prediction. Host overhead is no longer dominant.
- On-disk pipeline cache persistence is **blocked**: driver 621.01 crashes on
  `vkGetPipelineCacheData(pData=NULL)` and returns `VK_INCOMPLETE` for all sizes
  up to 64 MiB otherwise. Loading a supplied cache file still works.

Conclusion: the remaining elementwise gap is bandwidth/kernel-bound
(86% of measured DRAM ceiling in the chain), not launch/sync-bound.

## Phase 6 update (exp_012)

- vec4 aligned elementwise kernel: 20-add chain device time 0.858 ms for
  240 MiB = **293 GB/s (96% of measured DRAM ceiling)**, up from 86% in exp_011.
  Single-op device-time bandwidth is 269–284 GB/s; wall-time bandwidth remains
  89–111 GB/s because submit+fence is ~36 µs per op.
- Wall ratios versus PyTorch remain 0.43–0.49 at 1M. The elementwise gap is now
  **launch/sync-bound**, not bandwidth-bound. Next priority is submission
  overhead (timeline semaphore without per-op host wait in the real API,
  record-once command buffers), not more elementwise tuning.

## Phase 5 update (exp_013)

- Record-once Vulkan command buffers + timeline replay: 20-add chain 1.080 →
  0.899 ms/chain (1.20×). Bitsy replay versus PyTorch CUDA graph replay is
  **1.00×** — first measured performance parity, but only on the captured
  execution path. Default eager API ratios are unchanged (elementwise
  0.40–0.64, GEMM 0.71 at 1530 MHz).
- The ≥1.5× replay speed prediction was not met because GPU work (~0.86 ms) is
  now the floor. Next: make deferred/captured submission the default for
  chains and measure end-to-end training steps.

## Phase 4/5 update (exp_014)

- Default transfers are now deferred: uploads/readbacks use a persistent
  host-visible staging pool instead of allocating + flush + wait per call.
  A synthetic MLP training step drops 9.575 → 3.220 ms/step at matched
  1530 MHz (**2.97×**) with **0** staging allocations during 200 timed steps.
- The ≥0.85 end-to-end throughput target is **still missed** (PyTorch/Bitsy
  0.34–0.37, i.e. Bitsy ≈2.7× slower). The remaining cost is per-op
  Python/ctypes dispatch and autograd construction over many small dispatches;
  it is launch/CPU-bound, not bandwidth- or transfer-bound.
- Numeric parity: 42/42; MNIST report identical, CIFAR Bitsy losses
  bit-identical between the two transfer modes. The ~1e-8 torch loss variation
  is cuDNN benchmark nondeterminism.
- Next: capture the whole training step (CUDA-graph equivalent) with managed
  input/parameter buffers and replay it, then measure against PyTorch graphs.

## Phase 5 update (exp_015)

- Full-step capture/replay works and is bit-exact (captured loss == eager loss,
  abs diff 0.0). It was made possible by splitting the capture into
  command-buffer chunks at the autograd's CPU-gradient readbacks; a single
  command buffer deadlocked because `bt_vk_wait()` collected timestamp queries
  from the unsubmitted buffer.
- Captured step wall time 3.220 (eager) → **0.294–0.355 ms/step** (~9–11×), but
  PyTorch CUDA graph is 0.067 ms/step: **graph ratio 0.19–0.23**, missing both
  the ≥0.85 end-to-end target and the ≥0.80 gate.
- Ranked remaining causes: (1) one extra submit/replay from the CPU readback in
  `_unbroadcast` (~36 µs); (2) many small dispatches from the interim Python
  autograd. Next: device-side reductions to remove the readback and get a
  single-chunk capture, then re-measure with parity verification.
- A first chunking implementation that recycled resources at chunk boundaries
  produced bit-wrong replays; fixed by pinning resources until the final wait.
  Replaying a capture after that wait is now documented as invalid.

## Phase 6 update (exp_016)

- Device-side unbroadcast reductions (2D) removed the CPU gradient readback
  from every backward. Capture-phase submit count dropped 2 → 1 (single-chunk
  capture for the MLP step); the eager step improved 3.220 → 2.626–2.911 ms
  (1.1–1.2×), ratio vs PyTorch 0.34–0.37 → 0.37–0.41.
- **Captured graph throughput did not improve** (0.20–0.23 vs PyTorch CUDA
  graph, same as exp_015): the ≥0.28 prediction was missed. One fewer shared-
  wait submit saves only a few µs, and the extra reduction dispatches offset it.
- Numeric drift from the device reduction is 1.07e-7 relative (1e-5 tolerance);
  MNIST report identical, CIFAR unchanged, parity 42/42. >2D gradients still
  use the CPU fallback.
- Ranked remaining cause is now clearer: **~66 small Vulkan dispatches plus
  barriers per step** dominate the captured graph. Next: fuse backward/
  optimizer op chains (or a fused step kernel) to cut dispatch count, then
  re-measure with full parity verification.

## Phase 5/6 update (exp_017)

- Profiler-guided fusion: 66 dispatches/step broke down as 22 scalar
  elementwise, 20 tiled GEMM, **16 transpose**, 6 vec4 elementwise, 2 rb-vec4
  GEMM. Added a zero-dispatch transpose **view** for matmul backward (GEMM
  already consumes strides) and a fused `sgd_step` shader (lr as push
  constant). **66 → 46 dispatches (30.3%)**, barriers 66 → 52.
- Captured step 0.296–0.334 → **0.257 ms** (1.15–1.30×; 1.30× only against the
  slower baseline run), graph ratio 0.20–0.23 → **0.259**. The ≥1.3× and ≥0.28
  stretch predictions are **not reliably met** and are reported as misses.
- Eager step 2.626 → **2.46 ms**, ratio vs PyTorch 0.41–0.47 — best eager
  figure so far, but still far below ≥0.85.
- Parity: 42/42; CTest 1/1 Vulkan and CPU-only (new SGD op has a CPU path);
  MNIST W1 diff 1.79e-07, CIFAR conv1 W diff 6.63e-07 — small drift, no
  systematic change (fused SGD keeps the multiply-then-subtract order).
- Ranked remaining cost: **~20 tiled GEMM dispatches**, many of them tiny
  ones-vector reductions from the interim autograd. Next: a dedicated
  reduction kernel and fused loss/softmax backward, with parity re-verified.

## Phase 6 update (exp_018)

- Dedicated 2D axis-reduction shader (`reduce_sum`) replaces ones-vector GEMMs
  in `sum`, `log_softmax`, `cross_entropy`, `mse_loss` and `_unbroadcast`.
  Tiled GEMMs per step 20 → 10; total dispatches 46 → 43.
- Eager step 2.46 → **1.898–1.955 ms** (1.26–1.30×), ratio vs PyTorch
  **0.55–0.56** — the best eager figure yet. Captured step 0.257 →
  **0.234–0.246 ms**, graph ratio 0.259 → **0.277–0.287** (the ≥0.29 stretch
  target was just missed).
- Parity: 42/42; CTest 1/1 both builds; MNIST report identical, CIFAR conv1
  W diff 6.41e-07; captured step bit-identical to eager (diff 0.0). The new
  reduction has an autograd-aware backward with broadcast gradient.
- Ranked remaining cost: ~19 scalar elementwise + 10 GEMM dispatches per step.
  Next: a general fused elementwise-chain kernel (multiple ops per dispatch),
  with full parity re-verification.

## Phase 5/6 update (exp_019)

- Attempted zero-dispatch broadcast views for reduction gradients. They gave
  captured 0.213 ms / graph ratio 0.312 / eager 1.67 ms, **but broke training
  parity**: MNIST loss 1.87 vs 1.33, accuracy 0.434 vs 0.896, W1 diff 0.149;
  all four parameter gradients changed while forward loss stayed identical.
  Reverted and the API removed. Making `reduce_sum`/`sgd_step` stride-aware did
  not fix it; the cause is elsewhere in the backward graph.
- Kept only fused `tanh_backward`: dispatches 43 → 41, no measurable captured
  gain (0.240 ms, ratio 0.278). Eager 1.67 ms (ratio 0.57) within variance.
- **Methodology gap found:** 42/42 isolated-op parity did not catch a gradient
  flow error; end-to-end training parity must be a required gate for autograd
  /view changes. Next: a safe full-shape contiguous gradient kernel or a fully
  stride-audited tensor stack, then re-measure.

## Phase 2/6 update (exp_020)

- **Required gate added:** `tests/parity/grad_flow.py` compares loss, all
  parameter gradients and post-step parameters for a full MLP step in Bitsy vs
  PyTorch; wired into `run.py` (43/43) and both CI jobs. Proven to FAIL on a
  corrupted gradient with unchanged loss — exp_019's blind spot is closed.
- Safe broadcast elimination: a dedicated `broadcast_grad` kernel writes a
  contiguous full-shape gradient, replacing `ones * grad` without stride-0
  views. Captured 0.240 → **0.219 ms** (~1.10×), graph ratio 0.278 → **0.304**
  (≥0.31 just missed), eager 1.669 → **1.57 ms**, ratio 0.62. Dispatch count
  stays 41 (1:1 replacement). Parity 43/43 + training curves.
- Ranked remaining cost: ~15 scalar elementwise + 10 tiled GEMM dispatches.
  Next: multi-stage elementwise fusion or the GEMM class; all changes must pass
  the new gradient gate.

## Phase 6 update (exp_021)

- GEMM ladder L2 block order measured and **closed as negative**: 4×4 group
  swizzle (variant 6) was 1.32× slower at 256² (0.227 vs 0.172 ms) and 1.21×
  slower at 1024² (1.193 vs 0.985 ms) than the current vec4 winner. Cache keeps
  variant 4; no training-step change. 1024² ratio still 0.68–0.76 vs ≥0.80.
- Untried GEMM rungs: fused epilogues, packed f16, other tile aspect ratios.
  Alternatively return to the ~15 scalar elementwise dispatches per step.
  All changes must pass the 43-case gate including the gradient flow check.

## Phase 5/6 update (exp_022)

- Fused dual-output mul backward (packed 2n output + offset views) was tried;
  isolated outputs matched unfused muls but the 43-case gradient gate failed
  with identical loss and wrong parameter gradients. Reverted; kernel kept as
  a negative-result variant. The gate proved load-bearing again.
- **Correctness fix:** view offsets were ignored by `bt_tensor_copy_to_host`
  and by the fused `sgd_step`/`reduce_sum`/`broadcast_grad`/`dual_mul` shaders.
  Added `bt_vk_readback_offset()` and offset push constants. Likely root cause
  of exp_019's broadcast-view gradient corruption. No perf change (offset 0 in
  normal paths): captured 0.220 ms / ratio 0.303, eager 1.56 ms / ratio 0.627.
- Next: retry zero-dispatch views now that offsets are honoured, or pursue
  another fusion/GEMM rung. All changes must pass the 43-case gate.

## Phase 5/6 update (exp_023)

- Zero-dispatch stride-0 broadcast views retried after the exp_022 view-offset
  fix. Isolated 42 op cases passed but the mandatory gradient gate FAILED
  (42/43, identical loss, wrong parameter gradients) — the second rejection.
  Reverted immediately; API removed with a header warning. exp_022's offset fix
  was not the root cause; some other stride-blind consumer remains.
- Final tree unchanged: captured 0.224 ms / ratio 0.300, eager 1.75 ms / ratio
  0.604, elementwise 0.42–0.69, GEMM 1024² 0.72, parity 43/43.
- Next: find the remaining non-contiguous-view consumer, or pursue the
  remaining GEMM rungs / a stride-audited autograd rewrite. The 43-case gate
  remains mandatory.

## Phase 6 update (exp_024) — end-to-end target MET

- Fused GEMM+bias epilogue for `nn.Linear.forward` (tiled/vec4 shape selection,
  CPU fallback) with a custom backward reproducing the same gradients.
  Dispatches/step 41 → **35**.
- **Eager end-to-end training throughput 0.60 → 0.95–0.97 of PyTorch CUDA —
  the ≥0.85 target is MET** (first time with the ordinary API). Eager step
  1.75 → 1.065–1.145 ms. Removing the redundant `ones @ b` GEMM and add per
  Linear was worth ~1.6×.
- Captured graph: 0.224 → 0.186–0.199 ms, ratio 0.30 → 0.34–0.37 (improved,
  still far below 0.85). Standalone GEMM 0.72 and elementwise ≤0.54 remain
  missed; those are the next ranked gaps.
- Two bugs caught during implementation and fixed before merge: a `self._op`
  reference to the `Linear` module instead of a Tensor, and a stale
  `*out = result` in the fallback that returned zeros for unaligned shapes.
  The 43-case gate stayed green throughout.

## Phase 6 update (exp_025)

- 128×64×8 vec4 GEMM tile (variant 7) measured and **rejected**: 1.15× slower
  at 256² (0.148 vs 0.129 ms) and 1.09× slower at 1024² (1.091 vs 1.005 ms).
  Never auto-selected; kept as a negative-result variant. 1024² ratio still
  0.68–0.76 vs ≥0.80.
- Tuner methodology issue found: single-sample runs can flip winners by <1%
  (this run preferred variant 6 at 1024², which exp_021 measured 21% slower).
  Cache restored to the established variant 4; a future tuner should use the
  median of several samples.
- End-to-end target remains MET (eager 1.006 ms, ratio 1.01); captured graph
  0.196 ms / ratio 0.339. Parity 43/43 incl. gradient gate, CTest both builds.
- Next: packed f16, other tile dimensions, or backward-GEMM fusion; all changes
  must pass the 43-case gate.

## Phase 6 update (exp_026)

- `gemm_tune.py` now uses the median of 3 samples; two consecutive runs picked
  the same winners, closing the exp_025 noise-flip window (a 0.3% sample had
  earlier preferred a variant measured 21% slower).
- Added 64×128×8 vec4 GEMM (variant 8), the third aspect-ratio/L2 rung. It won
  1024² in both median runs: 0.982 / 1.001 ms vs 1.069 / 1.016 ms for variant 4
  (1.02–1.09×). Autotune cache now selects variant 8 at 256² and 1024².
- GEMM ratio still 0.72–0.76 vs ≥0.80 (clock-dependent); ≥1.05× prediction
  only met in one of two runs, reported as a range.
- End-to-end target remains MET (eager 0.979 ms, ratio 1.17); captured graph
  0.194 ms / ratio 0.343. Parity 43/43 incl. gradient gate, CTest both builds.
- Next: packed f16 or backward-GEMM fusion; median tuner is now the baseline
  for any variant promotion.

## Phase 5/6 update (exp_027)

- Fused log_softmax forward (exp + reduction + log + subtract → one dispatch)
  with an analytic backward `g - exp(y)*sum_cols(g)` and a CPU fallback.
  Dispatches/step 35 → **30**, barriers 35 → 30.
- Captured step 0.194 → **0.179–0.182 ms** (1.07×), graph ratio 0.343 →
  **0.366–0.368** (still far below 0.85). Eager step 0.896–0.980 ms, ratio
  1.09–1.11 (end-to-end target still MET). Standalone GEMM 0.71 and
  elementwise ≤0.54 remain missed.
- Parity: 43/43 incl. gradient gate, CTest both builds; MNIST W1 diff 3.3e-07
  (small fp32 drift from the fused reduction order, within 1e-5), CIFAR W diff
  7.2e-07.
- Ranked remaining cost: 10 elementwise + 5 reduce_sum + 4 sgd_step + 6 GEMM
  dispatches per step. Next: fuse backward reduction chains or packed f16;
  all changes must pass the 43-case gate.

## Phase 5/6 update (exp_028)

- Fused cross-entropy loss reduction: one workgroup reduces
  `onehot*log_softmax` over both axes and applies `-1/B`; backward is
  `-g*onehot/B`. Combined with exp_027's log_softmax, the old reduction chain
  and its `broadcast_grad` backwards disappear.
- Dispatches/step 30 → **22** (8 fewer), barriers 30 → 22. Captured step
  0.180 → **0.156 ms** (1.15×), graph ratio 0.366 → **0.425–0.542** (PyTorch
  graph timing varied; still below 0.85). Eager ratio 1.11–1.24 (target MET).
  GEMM 0.68 and elementwise ≤0.75 remain missed.
- Parity 43/43 incl. gradient gate, CTest both builds; MNIST W1 diff 3.3e-07,
  CIFAR W diff 7.3e-07 (within 1e-5).
- Remaining step: 5 elementwise, 4 sgd_step, 3 reduce_sum, 6 GEMM dispatches.
  Next: optimizer fusion, remaining reduction chains, or packed f16; all
  changes must pass the 43-case gate.

## Phase 5/6 update (exp_029)

- Fused the four SGD parameter updates into one dispatch (12-binding descriptor
  layout, dedicated pipeline, CPU fallback). Dispatches/step 22 → **19** but
  captured time unchanged (0.153–0.159 ms vs 0.156) — the ≥1.02× prediction was
  **not met**; kept as a neutral result because it is correct and reduces
  dispatch count. Three implementation bugs (descriptor pool size, C pointer
  level, ctypes out-array) were caught by CTest/the gate and fixed.
- Eager ratio 1.11–1.17 (end-to-end target still MET); graph ratio 0.43–0.45;
  GEMM and elementwise targets still missed. Parity 43/43 incl. gradient gate,
  CTest both builds, MNIST/CIFAR unchanged.
- Next: packed f16 GEMM or a different step-level fusion; dispatch count alone
  is not a reliable proxy for speed. All changes must pass the 43-case gate.

## Phase 6 update (exp_030)

- Packed f16 GEMM (last ladder rung) tried and **removed**: ~3× slower at
  1024² (3.508 ms / 612 GFLOP/s / ratio 0.26) and numerically wrong on a K=16
  check even after fixing swapped tile indices. TU116 CUDA cores offer no
  packed-f16 advantage; the Phase 1 "not measured" f16 rate is now answered.
  Predictions (≥1.05×, ≥0.80) not met; no code retained, tree equals exp_029.
- GEMM ladder is now comprehensively measured with no variant reaching ≥0.80
  on this GPU. Best FP32 winner remains variant 8 (64×128) from exp_026.
- Remaining gaps unchanged: standalone GEMM 0.68–0.76, elementwise ≤0.76,
  captured graph ~0.43. End-to-end target still MET (eager 1.11–1.17).

## Honesty

- Timestamp GPU time is a proxy, not the 1000-repeat isolated kernel time the
  campaign asks for; the standalone probe in `perf/` provides isolated numbers
  only for copy/FMA/memory measurements, not for these framework ops. This is a
  known gap in the Phase 2 method and is recorded rather than hidden.
- Profiling overhead is small (≤15 µs over a 50-iteration median) and one
  measurement (mul) is within noise.
- No framework kernel or algorithm was changed in this phase.
