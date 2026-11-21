# exp_025 — 128x64x8 tile aspect ratio (negative result)

**Date:** 2026-11-21
**Scope:** Phase 6 GEMM ladder. No tolerance changed. Parity: 43/43 incl.
gradient gate, CTest 1/1 Vulkan and CPU-only, MNIST W1 diff 1.788e-07,
CIFAR conv1 W diff 6.407e-07.

## Implementation

Added `gemm_rb_vec4_128x64.comp` (variant 7): 16×8 workgroup, 8×8 thread tile,
128×64 output tile, 128-thread tile loads (256 A vec4s + 128 B vec4s),
double-buffered 8-deep K step. Same vec4 alignment guard, a dedicated pipeline,
a dispatch branch with grid `(N+63)/64 × (M+127)/128`, and a tuner entry.

## Controlled A/B (gemm_tune.py, one process/clock)

| shape | current winner | variant 7 (128×64) | result |
|---|---:|---:|---|
| 256² | variant 1: 0.129 ms | 0.148 ms | 1.15× slower |
| 1024² | variant 4: **1.005 ms** | 1.091 ms | **1.09× slower** |

The 128×64 footprint did not improve occupancy enough to offset the extra
output tiles; the ≥1.05× and ≥0.80 predictions are **not met**. Variant 7 is
never selected by the cache and is kept only as a negative-result variant.

**Tuner noise note:** this run's single samples flipped the 256² winner from 4
to 1 and the 1024² winner from 4 to 6 (1.002 vs 1.005 ms — a 0.3% difference).
exp_021's controlled A/B measured variant 6 as 1.21× *slower*, so accepting
that flip would have been a noise-driven regression. The cache was restored to
the established `(256:4, 1024:4)` winners; no cache change is claimed. This
motivates a future tuner improvement (median of several samples).

## Final measurements (cache restored, variant 7 unused)

| metric | value |
|---|---:|
| GEMM 1024² | 1.318 ms, ratio 0.68 @1530 MHz |
| GEMM 256² | 0.240 ms, ratio 0.17 |
| captured step | 0.196 ms, graph ratio 0.339 |
| eager step | 1.006 ms, ratio 1.011 (target ≥0.85 MET) |
| elementwise | 0.40–0.99 |
| parity | 43/43 incl. gradient gate |
| CTest | 1/1 Vulkan and CPU-only |

The end-to-end target from exp_024 remains met; the standalone GEMM target
(≥0.80) is still missed. The 0.68 figure is clock-dependent (runs range
0.68–0.76).

## Interpretation / next

Two GEMM ladder rungs are now measured negatives (L2 swizzle, 128×64 tile).
Remaining untried: packed f16, other tile dimensions (e.g. 64×128), and
fusing backward GEMMs. The tuner's single-sample noise should be addressed
before promoting any variant with <5% measured advantage. All changes must
still pass the 43-case gate.

Raw data in `experiments/exp_025/{before_*,after_*,tile_ab.json}`.
