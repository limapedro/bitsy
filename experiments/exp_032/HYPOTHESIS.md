# exp_032 — lazy timestamp collection to cut eager wait cost

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 5 (submission overhead)

## Prerequisite

exp_031 classified the elementwise eager gap as host submit/fence overhead
(eager 0.17–0.47× vs captured 0.88–1.03× of PyTorch CUDA graphs). The
calibrated submit+fence cost is ~36 µs. `bt_vk_wait()` unconditionally collects
device-time timestamp queries with `vkGetQueryPoolResults(..., WAIT_BIT)` and
resets the query pool on every wait, even when no caller reads
`bt_vulkan_device_time_ms()`. That is an extra GPU sync in the hot path.

## Hypothesis

Recording and collecting device-time timestamps only after a caller asks for
device time removes an unconditional sync/reset from every eager wait,
reducing per-op wall time and raising eager ratios. The profiler uses a
separate query pool and is unaffected.

## Predictions

- Eager elementwise ratios improve **≥1.1×** (add 1M 0.44 → ≥0.48) at matched
  clock; captured/step throughput is unchanged (already collected lazily? no —
  captured replays also call wait, so may improve slightly).
- `bt_vulkan_device_time_ms()` still reports correct device time once requested
  (first post-enable sample may be 0; median of many samples unaffected).
- 43/43 parity incl. gradient gate; CTest both builds green.
- If no improvement, record the negative and keep the simpler always-on path.

## Measurement

1. Snapshot current elementwise/graph/eager/train JSON.
2. Gate `bt_vk_timestamp_start/end` and the device-time collection block in
   `bt_vk_wait()` on a `timestamps_wanted` flag set by
   `bt_vk_device_time_ms()`; keep profiler queries untouched.
3. Re-run `elementwise.py`, `elementwise_graph.py`, `gemm.py`, graph/eager step
   benchmarks with medians, plus 43/43 parity, MNIST/CIFAR, CTest both builds.
4. FINDINGS with all numbers including negatives. No tolerance change.

## Honesty

No result is claimed from a run where device time was never requested if that
invalidates the comparison; each report states its timing mode. The hard gate
still blocks any regression.
