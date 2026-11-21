# exp_031 — elementwise small-op launch overhead, measured fairly

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 3 (three-way timing) + Phase 5 (submission)

## Prerequisite

The campaign requires three-way timing per op: **isolated kernel (1000+
repeats in one command buffer)**, via the framework, and PyTorch. The current
`elementwise.py` times one op per `flush()+wait()` on the Bitsy side and one
op per `cuda.synchronize()` on the PyTorch side, so its 0.42–0.76 ratios include
per-op submit+fence (~36 µs) rather than kernel time. exp_013 already showed a
captured pointwise chain matches PyTorch CUDA graphs at 1.00×, but no
per-elementwise-op captured comparison exists.

## Hypothesis

Under the campaign's own batched/captured methodology the elementwise kernels
are at parity with PyTorch CUDA graphs; the reported wall gap is launch/sync
overhead in the eager path, not a kernel or bandwidth deficit. Measuring this
properly tells us whether more kernel work is justified.

## Predictions

- Bitsy captured/replayed single elementwise ops reach **≥0.85** of PyTorch
  CUDA graph throughput for large tensors (1M) and for small tensors where the
  kernel dominates after launch removal.
- Eager ratios remain ~0.4–0.76, confirming the gap is host-side.
- No kernel change is made if the captured comparison meets parity; the result
  is a measured classification, recorded honestly in FINDINGS.
- 43/43 parity incl. gradient gate remains green (no autograd change).

## Measurement

1. Snapshot current elementwise/graph/eager JSON.
2. Add `benchmarks/ops/elementwise_graph.py` (dataclass config, no argparse):
   for each op/size measure Bitsy eager, Bitsy capture+replay, PyTorch eager,
   PyTorch CUDA graph; median of trials, GPU clock logged.
3. Compare ratios and classify the remaining gap (host launch vs GPU kernel).
4. No tolerance change; FINDINGS records all numbers, including misses.

## Honesty

If captured elementwise does **not** reach parity, the result is a genuine
kernel gap and is reported as such. No eager number is hidden or replaced.
