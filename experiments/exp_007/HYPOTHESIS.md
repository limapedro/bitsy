# exp_007 — TOGISHI Phase 2: profiler and gap ledger

**Date:** 2026-11-21
**Owner:** Bitsy performance campaign TOGISHI

## Prerequisite

`experiments/exp_006` gates pass: vec4 copy 93.6% of measured DRAM BW, FP32
FMA 95.3% of clock-normalized peak. Phase 1 ceilings are recorded in
`perf/HARDWARE_LEDGER.md`: DRAM ~304 GB/s, FP32 ~5.49 TFLOP/s @1950 MHz,
submit+fence ~36 µs, allocate+free ~363 µs, empty dispatch ~3.1 µs.

## Hypothesis

The 0.23–0.53 elementwise and 0.22 GEMM ratios are dominated by a small number
of measurable overheads (submits, allocation, barriers, launch gaps), not by
kernel arithmetic alone. An instrumented profiler plus three-way timing will
rank those overheads in microseconds and identify the top fix with evidence.

## Prediction

- Per-op tracing shows at least one op where submit+wait time is a larger share
  of wall time than GPU dispatch time.
- Three-way timing isolates each op's gap into bandwidth/compute/launch/sync
  categories with microsecond amounts.
- PyTorch kernel profiling shows cuBLAS/cuDNN tile and register choices that
  explain the GEMM gap; these become concrete targets for later experiments.

## Scope of this experiment

- Add opt-in `hg.profiler` instrumentation in the Vulkan backend: per-dispatch
  timestamp queries, CPU enqueue time, bytes/FLOPs with % of measured ceilings,
  Chrome-trace JSON export, and per-step counts of dispatches/barriers/submits/
  allocations plus an honest GPU-idle estimate.
- Build `GAP_LEDGER.md` ranked by microseconds lost from three-way timing:
  isolated kernel (standalone probe or timestamp proxy), via Bitsy, via PyTorch.
- Profile PyTorch kernels with `torch.profiler` and save names/shapes.
- **No framework kernel or algorithm optimization in this experiment.**

## Honesty rules

- Profiling overhead is measured by running the same workload with profiling
  on and off; if it changes results materially, it is reported.
- Timestamp-only kernel time is labelled as a proxy, not a 1000-repeat isolated
  measurement, unless the standalone probe actually produced that number.
- Negative/inconclusive findings are recorded, not deleted.
