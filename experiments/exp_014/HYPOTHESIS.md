# exp_014 — deferred upload and stream-like default submission

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 4/5 (allocator + submission defaults)

## Prerequisite

exp_013 reached 1.00× of PyTorch CUDA **only** with explicit capture/replay;
the ordinary eager API still misses (elementwise 0.40–0.64, GEMM 0.71 at
1530 MHz). The C core already batches dispatches into one command buffer and
submits only on `bt.flush()` — but `bt_vk_upload()` (`vk_allocator.c:151-171`)
allocates a fresh staging buffer and calls `bt_vk_flush()` + `bt_vk_wait()` on
every `Tensor.from_numpy(..., "vulkan")`. `train_bitsy` uploads `xb` and `yb`
each step (two full syncs), then syncs again on `loss.numpy()`; PyTorch uploads
asynchronously and syncs once per step. The training harness records only
loss/accuracy — no wall time exists (BENCHMARKS.md target "end-to-end training
throughput ≥ 0.85 of PyTorch" is currently unmeasured).

## Hypothesis

Most of the *default-API* training-step gap is the two forced upload syncs plus
their per-call staging allocation, not GPU work. A persistent host-visible
staging pool with deferred copy submission — copy into pooled staging memory,
enqueue `vkCmdCopyBuffer` into the current command buffer, and release the
staging slot only after the timeline wait — makes ordinary chains behave like
a stream: the CPU runs ahead and blocks only on the loss readback, matching
PyTorch's semantics.

## Predictions

- Bitsy training-step wall time improves by **≥1.3×** with deferred upload
  (toggle `BITSY_DEFERRED_UPLOAD=0` = old path, `1` = deferred; default becomes
  `1` if the prediction is met).
- End-to-end training-step throughput reaches **≥0.85×** PyTorch CUDA at
  matched clock (the documented BENCHMARKS.md target).
- Zero `vkAllocateMemory` calls per step from uploads after warmup (staging
  pool reuse), verified by the allocator profiler counter.
- Numeric parity unchanged: 42/42 cases, MNIST/CIFAR loss curves bit-comparable
  to the committed reports, elementwise/GEMM ratios unchanged (no kernel edit).

## Measurement

1. Snapshot current elementwise/GEMM JSON and training JSON into `experiments/exp_014/`.
2. Implement pooled staging with deferred upload behind `BITSY_DEFERRED_UPLOAD`
   (default off during measurement); release staging only after the submitted
   timeline value is waited on.
3. Add `benchmarks/models/train_step_bench.py` (dataclass config, no argparse):
   MLP step (784→64→10, batch 64) on synthetic data, median+spread, GPU clock,
   Bitsy toggle off vs on versus PyTorch CUDA with `cudnn.benchmark=True`.
4. Run before (off) / after (on) in separate processes; re-run elementwise,
   GEMM, `tests/parity/run.py`, and MNIST/CIFAR loss-curve parity.
5. If the prediction is met, flip the default to deferred (`1`) and keep the
   toggle as an A/B escape hatch. Write FINDINGS including negatives.

## Honesty

No tolerance or kernel change. Wall time includes the final completion wait on
each side; upload mode is recorded in every report. If deferred upload does not
close the gap, keep the toggle default `0`, record the negative result, and
classify the remaining cause. All reports must state missed targets plainly.
