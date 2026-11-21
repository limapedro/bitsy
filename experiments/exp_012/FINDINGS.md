# exp_012 — Findings: vec4/aligned elementwise kernels

**Date:** 2026-11-21
**Scope:** Phase 6 kernel work. No tolerance changes. Parity: 42/42, CTest 1/1,
gradcheck/conv unchanged.

## Implementation

- Added `elementwise_vec4.comp`: vec4 loads/stores, linear index, no runtime
  div/mod, same op semantics as the scalar shader (including the fp64 div path
  and exp/log/tanh). Push-constant layout matches `vk_pc_t`.
- Host selects it only when all three tensors are contiguous, equal-shaped,
  have a final dimension divisible by 4, and 16-byte-aligned offsets. All other
  cases (broadcast, strided, scalar, odd sizes) keep the proven scalar path.
- Profiler trace confirms the fast path is used: 20-add chain events are
  `elementwise_vec4`.

## Measurements (all at 1530 MHz)

| op | n | before ms | after ms | device ms | speedup | device-time GB/s |
|---|---:|---:|---:|---:|---:|---:|
| add | 1M | 0.1162 | 0.1059 | 0.0423 | 1.09× | ~284 |
| mul | 1M | 0.1156 | 0.1091 | 0.0431 | 1.06× | ~279 |
| relu | 1M | 0.1140 | 0.0883 | 0.0297 | **1.29×** | ~269 |

20-add chain (1M elements, one submit): device time 0.858 ms for 240 MiB moved
= **293 GB/s, 96% of the measured 304 GB/s DRAM ceiling** (up from ~86% in
exp_011). This is the strongest evidence that the kernel itself is now near the
hardware limit.

Wall ratios versus PyTorch CUDA remain 0.43–0.49 at 1M because Bitsy wall time
includes a ~36 µs submit+fence per op while PyTorch's kernel time is ~39–52 µs.
For a 0.106 ms op, ~0.036 ms is host/sync overhead.

## Predictions

- "≥1.3× faster at 1M": **not met** overall (1.06–1.29×); relu nearly met it.
- "≥180 GB/s at 1M": **met** on device time (269–284 GB/s); the benchmark's
  wall-time GiB/s (89–111) remains dominated by submit overhead.
- "parity stays green": **met**.
- "ratio versus PyTorch improves for large tensors": **partially met** —
  absolute ms improved, but ratio stayed ~0.44–0.49 because PyTorch timings
  also varied.

## Interpretation / next

The scalar → vec4 kernel change moved the GPU kernel to ~90–96% of measured DRAM
ceiling for large tensors. The remaining wall-time gap is now overwhelmingly
submission/fence overhead (~36 µs per op) and small-tensor launch latency, not
memory bandwidth. The next work should target submission overhead (timeline
semaphore without per-op host wait in the real API, record-once command
buffers) with a new `exp_NNN`, rather than further elementwise kernel tweaks.

Raw data: `experiments/exp_012/before.json`, `after.json`.
