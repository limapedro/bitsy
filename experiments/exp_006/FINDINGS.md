# exp_006 — Findings: TOGISHI Phase 1 hardware calibration (final)

**Date:** 2026-11-21
**Scope:** standalone C + Vulkan only (`perf/`). No Hagane source was compiled,
linked or modified.

## Method fixes made after the first run

The first calibration failed both gates. Per the campaign rule we fixed the
measurement method before touching the framework:

1. `copy4` reported >1 TB/s, physically impossible on 336 GB/s memory. Root
   cause: the destination buffer was never read, so writes could be
   compressed/discarded. The result is **excluded**; the trustworthy copy uses
   a 1 GiB buffer, one pass, and full-entropy `uintBitsToFloat(hash(i))` fill.
2. The initial "L2" test did too little work (169 GB/s < DRAM). Replaced with a
   512 MiB streaming read; DRAM read now agrees with copy (~298–305 GB/s).
3. FP32 FMA was launch/latency-bound (17% of peak with 8 dependent FMAs).
   Pipeline executable properties showed only 17–45 registers/thread and zero
   shared memory, so occupancy was not the limit. A 32-way unrolled shader
   (`fma8u.comp`) removed the loop overhead and reached the gate.
4. GPU clocks move between 1530 and 1950 MHz during a run. The harness now
   samples `nvidia-smi` immediately before/after the FMA kernel and normalizes
   peak FLOP/s to the measured clock.

Nsight Compute was installed but profiles CUDA kernels only in this release
(`==WARNING== No kernels were profiled` for our Vulkan app); Nsight Graphics
CLI was unavailable. Shader-level evidence therefore comes from
`VK_KHR_pipeline_executable_properties`, which we enabled in the probe.

## Final measurements (`perf/calibration.json`)

| Measurement | Result | Ceiling | Ratio |
|---|---:|---:|---:|
| vec4 copy, 1 GiB, random fill | 284.9 GB/s | 304.4 GB/s measured DRAM read | **93.6%** |
| DRAM streaming read, 512 MiB | 304.5 GB/s | — | — |
| FP32 FMA, 8 acc, 32-way unrolled | 5232.8 GFLOP/s | 5491 GFLOP/s @1950 MHz | **95.3%** |
| FP32 FMA, same kernel across 3 runs | 5091–5233 GFLOP/s | clock-normalized | 93.1–95.3% |
| FP32 FMA naive / 8-acc loop / 16-acc | 928 / 3993 / 4438 GFLOP/s | — | 17 / 73 / 81% |
| pipeline executable properties | fma8u: 16 registers, subgroupSize 32, 0 shared | — | no spills reported |
| shared stride 1 / 2 / 32 | 4.0 / 3.9 / 10.8 µs | — | 2.7× bank-conflict cost |
| empty dispatch + barrier (GPU) | 3.1 µs | — | — |
| CPU submit + fence / timeline | 35.9 / 36.2 µs | — | — |
| `vkAllocateMemory` + free | 363 µs | — | — |
| staging H2D+D2H 64 MiB | 12.6 GB/s | — | — |

## Gate decision

| Gate | Target | Measured | Verdict |
|---|---:|---:|---|
| copy vs measured DRAM BW | ≥ 90% | 93.6% | **pass** |
| FP32 FMA vs clock-normalized peak | ~95% | 95.3% best, 93.1–95.3% spread | **pass** (with spread recorded) |
| method trustworthy | yes | invalid copy4/L2 results excluded | **pass** |

Both gates are now met, so Phase 1 is complete. The framework has still not
been touched; Phase 2 (profiler + GAP_LEDGER) is the next step and requires its
own HYPOTHESIS.md.

## Relevance to the framework gap

- Real DRAM ceiling is ~300 GB/s; Bitsy's best elementwise kernel was ~77 GB/s
  (~26% of the ceiling), so the remaining gap is in framework kernels.
- A single submit+fence costs ~36 µs, explaining much of the 60–70 µs small-op
  floor. `vkAllocateMemory` at 363 µs confirms allocator work is essential.
- A 32-way shared-memory bank conflict costs 2.7×; GEMM must use conflict-free
  shared layouts.
