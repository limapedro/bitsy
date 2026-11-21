# exp_003 — Batched Vulkan submissions and GPU timestamp queries

**Date:** 2026-11-21
**Owner:** Bitsy M10

## Hypothesis

Every Bitsy Vulkan op currently records one command buffer, calls
`vkQueueSubmit`, and waits on a fence. That fixed per-op submission cost is the
dominant term for small ops (see exp_001) and part of the GEMM gap (exp_002).
Deferring submission — recording many dispatches into one command buffer and
flushing only when the host reads data or explicitly asks — should cut wall
time materially for multi-op workloads and at least reduce per-op overhead.

GPU timestamp queries should let us separate kernel time from submission time,
so we can say which bottleneck remains instead of guessing.

## Prediction

- A 20-op elementwise chain drops from ~20 ms to under 5 ms wall time when
  batched into one submission.
- Single-op wall time improves modestly (one submit is still one submit), but
  timestamp queries report kernel time well below wall time, quantifying the
  submission overhead.
- 40/40 parity cases remain green; readback still forces a flush so correctness
  is unchanged.

## Measurement

1. Snapshot current elementwise and GEMM results (`before.json`).
2. Implement deferred recording with rotating descriptor sets, explicit
   `bt_vulkan_flush()`, and timestamp queries around each dispatch.
3. Re-run `benchmarks/ops/elementwise.py` and `benchmarks/ops/gemm.py` with an
   explicit flush per iteration (fair wall-time comparison) and record device
   time separately. Save `after.json`.
4. Re-run `tests/parity/run.py`, `gradcheck.py`, `conv_check.py`, CTest.
5. If the chain benchmark does not improve, keep the code only if timestamps
   still provide needed evidence, and record the negative result honestly.

## Honesty

PyTorch wall time is measured with `torch.cuda.synchronize()` per iteration.
Bitsy wall time will include `bt_vulkan_flush()` (submit + fence wait) per
iteration. Device time is reported only for Bitsy because PyTorch timestamp
query integration is not implemented; comparing Bitsy device time to PyTorch
wall time would be unfair and will not be done.
