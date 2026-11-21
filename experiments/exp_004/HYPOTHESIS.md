# exp_004 — Async submission (timeline semaphore) and pointwise fusion

**Date:** 2026-11-21
**Owner:** Bitsy M10

## Hypothesis

`bt_vulkan_flush()` still waits on a fence every batch. Replacing that with a
timeline semaphore lets the CPU submit the next batch while the GPU executes
the previous one. A second source of overhead is one dispatch per pointwise op;
fusing `add + relu` into a single shader removes one dispatch and one barrier.

## Prediction

- A 20-op chain measured with an explicit completion wait at the end gets at
  least 1.2x faster with async submission (CPU can run ahead).
- A 20-pair `add + relu` chain gets at least 1.3x faster with the fused kernel
  versus calling `add` and `relu` separately.
- Parity remains 40/40 (plus new fused cases), gradcheck/conv_check unchanged.

## Measurement

1. Snapshot current chain, elementwise and GEMM JSON into `before_*.json`.
2. Add timeline semaphore: flush submits + signals, no wait; new
   `bt_vulkan_wait()` waits for the last submitted value and collects timestamp
   results. Readback/upload wait where the host needs the data.
3. Add `bt_add_relu` fused kernel and Python `add_relu`; add parity cases.
4. Re-run elementwise/gemm benchmarks with `flush() + wait() +
   torch.cuda.synchronize()` per iteration for a fair wall-time comparison.
5. Re-run `batch_chain.py` and a new `fusion.py` benchmark; save after JSON.
6. Keep both changes only if parity stays green; report ratios honestly even
   if targets remain missed.

## Honesty

Bitsy wall time includes an explicit completion wait per iteration, matching
PyTorch's `torch.cuda.synchronize()`. Device time comes from Vulkan timestamp
queries and is never compared to PyTorch wall time.
