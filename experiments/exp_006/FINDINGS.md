# exp_006 — Findings: TOGISHI Phase 1 hardware calibration

**Date:** 2026-11-21
**Scope:** standalone C + Vulkan only (`perf/`). No Hagane source was compiled,
linked or modified for this experiment.

## Environment during measurement

- GPU clock: 1530 MHz idle before, **1950 MHz** at the end of the run.
- Power/temperature: 42 W / 56 °C before, 62 W / 58 °C after.
- Driver 621.01, Vulkan device API 1.4.357, timestampPeriod = 1 ns.
- Device limits: shared memory 49152 B, push constants 256 B, max workgroup 1024.

## Measurements (final run, `perf/calibration.json`)

| Measurement | Result | Interpretation |
|---|---:|---|
| vec4 copy, 1 GiB | 7.28 ms, **295.1 GB/s** | 87.8% of 336 GB/s theoretical |
| vec4x4 copy, 1 GiB | 1.98 ms, 1083 GB/s | **invalid**: >3× theor. bandwidth; compression/write-combining artifact, excluded |
| FP32 FMA (naive) | 928 GFLOP/s | 17% of 5.49 TFLOP/s peak at 1950 MHz |
| FP32 FMA (8 accumulators, 524k lanes) | 4100 GFLOP/s | 75% of peak |
| FP32 FMA (16 accumulators) | 3537 GFLOP/s | 64% of peak |
| shared stride 1 / 2 / 32 | 4.72 / 4.77 / 11.71 µs | 32-way bank conflict costs **2.5×** |
| "L2" read / DRAM read | 169 / 303 GB/s | L2 result unreliable (too little work), DRAM ~303 GB/s agrees with copy |
| empty dispatch + barrier (GPU) | 3.80 µs | per-dispatch floor |
| CPU barrier / dispatch recording | 0.27 / 0.32 µs | cheap |
| CPU submit + fence wait | 35.2 µs | dominant per-submit overhead |
| CPU timeline signal + wait | 37.4 µs | similar |
| `vkAllocateMemory` + free | 390 µs | huge; confirms allocator work is essential |
| staging H2D+D2H 64 MiB | 12.6 GB/s combined | PCIe ceiling |

FMA varied 4.10–4.35 TFLOP/s across three runs, so the honest figure is
**~4.1–4.35 TFLOP/s**, not a single exact value.

## Gate assessment

| Gate | Target | Measured | Verdict |
|---|---:|---:|---|
| Bare copy vs theoretical BW | ≥ 90% | 87.8% | **missed by 2.2 points** (best trustworthy kernel) |
| Bare FP32 FMA vs peak | ≥ 95% | ~75–79% | **missed** |
| Measurement method trustworthy | yes | partially | copy4 and L2 results are invalid and excluded |

Per the campaign rule, the framework must not be optimized until these gates
pass. The method issues found (compression artifact, too-small L2 workload) are
fixed by excluding those measurements; the remaining FMA gap needs a deeper
look at SASS/occupancy with `VK_KHR_pipeline_executable_properties` and Nsight
Compute in Phase 2.

## What this says about the framework's 0.22–0.53 ratios

- A single `vkQueueSubmit` + fence wait costs ~35 µs, which alone explains much
  of the ~60–70 µs small-op floor seen in `BENCHMARKS.md`.
- One GPU dispatch + barrier costs ~3.8 µs; the 20-op chain measurement (0.94 ms
  device time) is real GPU work, not just launch overhead.
- Allocating device memory costs ~390 µs — the caching allocator's 5× win was
  expected, and further suballocation work is justified.
- Real DRAM is ~295–303 GB/s. Bitsy's best elementwise result was ~77 GB/s
  (~26% of measured ceiling), so the remaining gap is in the framework's
  kernels, not in the hardware.

## Next steps (Phase 2, after gates pass)

Build `hg.profiler` with per-dispatch timestamps, CPU enqueue time, bytes/FLOPs
as % of the ceilings above, Chrome-trace export, and per-step
dispatch/barrier/submit/allocation counts. Then create `GAP_LEDGER.md` ranked by
microseconds lost. Raw numbers are in `perf/calibration.json`.
