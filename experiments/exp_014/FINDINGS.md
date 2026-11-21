# exp_014 — Findings: deferred upload and stream-like default submission

**Date:** 2026-11-21
**Scope:** Phase 4/5 allocator + submission defaults. No kernel or tolerance
change. Parity: CTest 1/1 Vulkan and CPU-only, 42/42 parity cases, MNIST and
CIFAR loss curves re-verified with the new default.

## Implementation

- `bt_vk_upload()` now copies host data into a **persistent host-visible staging
  pool** and enqueues the `vkCmdCopyBuffer` into the current command buffer;
  it no longer flushes+waits per call. `bt_vk_readback()` uses the same pool.
- Staging slots move `used -> submitted (on bt_vk_flush) -> pooled (on
  bt_vk_wait)`, so a slot is never reused while the GPU may still access it.
  The pool is size-classed and swap-removed; a safety flush prevents in-use
  reuse beyond 64 outstanding slots.
- `BITSY_DEFERRED_UPLOAD=0` restores the old synchronous path as an A/B escape
  hatch; **the default is now deferred** because the speed prediction was met.
- New `bt.stage_alloc_count()` exposes the staging allocation counter. New
  `benchmarks/models/train_step_bench.py` adds the end-to-end training-step
  wall-time measurement the campaign previously lacked (dataclass config,
  synthetic data, median+spread, GPU clock, separate processes per mode).

## Measurements — training step (MLP 784→64→10, batch 64, 200 timed steps)

| mode | clock | Bitsy ms/step | torch ms/step | torch/Bitsy | stage allocs / 200 steps |
|---|---:|---:|---:|---:|---:|
| baseline (`=0`) | 1530 MHz | 9.575 | 1.215 | 0.127 | 0 (old path) |
| deferred (`=1`) | 1530 MHz | 3.220 | 1.200 | **0.373** | **0** |
| baseline (`=0`) | 1620 MHz | 8.774 | 1.143 | 0.130 | 0 (old path) |
| deferred (new default) | 1530 MHz | 3.322 | 1.121 | 0.338 | **0** |

**Speedup at matched 1530 MHz: 2.97×** (9.575 → 3.220 ms/step); the second
pair gives 2.83× at a *lower* deferred clock, so the result is conservative.
The staging pool makes **zero host-visible staging allocations during the 200
timed steps** (total counter plateaus at 2–3 distinct sizes after warmup).

## Predictions

- "≥1.3× training-step speedup": **met** (2.83–2.97×).
- "throughput ≥0.85× PyTorch": **not met** (0.34–0.37). This is a real miss;
  the remaining ~3.1 ms/step is not transfer synchronization.
- "zero staging allocations per step after warmup": **met** (0 during timed
  steps).
- "numeric parity unchanged": **met** — 42/42; MNIST report numerically
  identical to the committed one; CIFAR Bitsy losses are **bit-identical**
  between baseline and deferred (per-epoch diff 0.0). The ~1e-8 variation in
  `torch_loss` and the ~1e-7 variation in final weight drift are PyTorch cuDNN
  benchmark algorithm nondeterminism, not a Bitsy change (both modes were run
  back-to-back with the same build).

Ordinary op benchmarks are unchanged within noise (elementwise ratios
0.43–0.67, GEMM 1024² 0.71 at 1530 MHz) because they never call `from_numpy`
inside the timed region.

## Interpretation / next

Deferred upload removed the two avoidable per-step syncs and the per-call
staging allocation: a large, real improvement to the *default* API. The
remaining step time is dominated by per-op Python/ctypes dispatch and autograd
graph construction over many small Vulkan dispatches. The next experiment
should capture the **whole training step** (CUDA-graph equivalent) with managed
input/parameter buffers and replay it, then measure end-to-end step throughput
against PyTorch CUDA graphs — its own `exp_NNN`.

Raw data: `experiments/exp_014/{train_step_*,before_*,after_*,cifar_*}`.
