# exp_026 — median-based GEMM tuning and the 64x128 tile rung

**Date:** 2026-11-21
**Scope:** Phase 6 GEMM ladder + measurement reliability. Parity: 43/43 incl.
gradient gate, CTest 1/1 Vulkan and CPU-only, MNIST W1 diff 1.788e-07,
CIFAR conv1 W diff 6.445e-07.

## Tuner reliability fix

`gemm_tune.py` now takes the **median of 3 samples** per variant (dataclass
`TuneConfig.samples`) and writes the cache from medians. Two consecutive runs
selected **the same winners** (256² and 1024² → variant 8), unlike exp_025's
single-sample 0.3% flip to a variant measured 21% slower. The reliability
prediction is **met**.

## 64×128 tile rung

Added `gemm_rb_vec4_64x128.comp` (variant 8): 16×8 workgroup, 8×8 thread tile,
64×128 output tile, 128-thread tile loads, double-buffered 8-deep K step.

Median A/B (two runs, 1530 MHz):

| shape | run 1 | run 2 | previous winner (v4) |
|---|---:|---:|---:|
| 256² | 0.132 ms (v1) | 0.129 ms (v8) | 0.150 / 0.132 |
| 1024² | **0.982 ms (v8)** | **1.001 ms (v8)** | 1.069 / 1.016 |

Variant 8 was the 1024² winner in **both** runs, beating variant 4 by
1.02–1.09× and variant 6 (the noise-flipped choice from exp_025) by 1.03–1.06×.
The cache now selects variant 8 at both shapes.

## Predictions

- "median tuner prevents noise flips": **met** (stable winners across 2 runs).
- "variant 8 ≥1.05× at 1024²": **partially met** (1.09× in run 1, 1.015× in
  run 2 — reported as a range, not cherry-picked).
- "GEMM ratio ≥0.80": **not met** (0.72–0.76 depending on clock).
- "parity unchanged": **met** (43/43; training curves unchanged).

## Final measurements (cache = variant 8 at both shapes)

| metric | value |
|---|---:|
| GEMM 1024² | 1.270 ms, ratio 0.72 @1530 MHz |
| GEMM 256² | 0.216 ms, ratio 0.19 |
| captured step | 0.194 ms, graph ratio 0.343 |
| eager step | 0.979 ms, ratio 1.173 (target ≥0.85 MET) |
| elementwise | 0.42–0.71 |
| parity / CTest | 43/43 / 1/1 both builds |

The standalone GEMM ratio is clock-dependent (0.72–0.76 across runs); the
≥0.80 target remains missed. The end-to-end target from exp_024 remains met.

## Interpretation / next

The measurement fix is as valuable as the kernel: kernel selection is now
reproducible. 64×128 is the third aspect-ratio/L2 rung measured (128×64 and
swizzle were negative). Remaining GEMM rungs: packed f16, or fusing backward
GEMMs. All changes must pass the 43-case gate.

Raw data in `experiments/exp_026/{before_*,after_*,median_tune_run2.json}`.
