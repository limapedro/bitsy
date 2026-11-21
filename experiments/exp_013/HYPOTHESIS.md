# exp_013 — record-once command buffers and timeline replay

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 5 (submission overhead)

## Prerequisite

exp_012: elementwise kernels now reach 96% of measured DRAM ceiling on device
time, but wall ratios stay 0.43–0.49 because ~36 µs per op is submit+fence.
exp_011 async batching of 20-op chains gave only 1.10× because the CPU still
re-records every dispatch.

## Hypothesis

The remaining wall-time gap is CPU dispatch recording plus one submit/wait per
op. A record-once/replay command buffer executed many times per submission,
combined with timeline semaphore signalling only once per batch, will cut
enqueue overhead enough to raise wall ratios for repeated chains.

## Predictions

- Replaying a captured 20-add chain is ≥1.5× faster per chain than re-recording
  the same chain each iteration.
- Bitsy replay throughput versus PyTorch CUDA-graph replay is ≥0.80 (matching
  the framework goal for captured execution) or the gap is classified
  explicitly.
- Numeric parity stays 42/42; captured kernels produce identical bits.

## Measurement

1. Snapshot current elementwise/GEMM JSON into `before_*.json`.
2. Add `bt_vulkan_capture_begin/end/replay` to the Vulkan backend (one reusable
   command buffer, timeline signal per submission, one host wait).
3. Add `benchmarks/ops/replay_chain.py` comparing recorded-each-iteration with
   capture+replay on Bitsy, and PyTorch eager with `torch.cuda.CUDAGraph` replay.
4. Re-run elementwise/GEMM plus full parity; write FINDINGS including negatives
   and any API limitations (e.g. buffer lifetime during replay).

## Honesty

Wall comparisons include the final completion wait on both sides. Device time
is Bitsy-only from timestamp queries. If replay does not close the gap, record
the negative result and the classified remaining cause. Do not change any
tolerance or kernel algorithm in this experiment.
