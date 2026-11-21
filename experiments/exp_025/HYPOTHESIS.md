# exp_025 — 128x64x8 tile aspect ratio for the vec4 GEMM

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 6 (GEMM ladder)

## Prerequisite

exp_024 met the end-to-end target but the standalone 1024² GEMM is still
0.72 of PyTorch CUDA (target ≥0.80). The ladder explicitly lists both
128×128×8 and **128×64×8** register-blocked tiles. Only the 128×128 variant
has been measured (exp_008–010); the L2 swizzle rung (exp_021) was negative.
With 22 SMs on TU116, a 128×64 tile produces twice as many output tiles, which
may improve SM occupancy when the 128×128 kernel is register/smem limited.

## Hypothesis

A 128×64×8 register-blocked vec4 GEMM (16×8 workgroup, 8×8 thread tile) is
faster than 128×128×8 at 1024² because the smaller per-CTA footprint raises
occupancy, without changing arithmetic per output element.

## Predictions

- At 1024², the 128×64 variant is **≥1.05× faster** than variant 4 and the
  GEMM ratio reaches **≥0.80** at matched clock.
- At 256² (tiled regime) it is not selected; the cache keeps the current
  winner there.
- Parity stays green: 43/43 including the gradient gate; GEMM output within
  1e-5 tolerance (per-element accumulation order unchanged).
- If it fails, record the negative and keep the autotune cache unchanged.

## Measurement

1. Snapshot current GEMM/graph/eager JSON and the autotune cache.
2. Add `gemm_rb_vec4_128x64.comp` as variant 7 (same vec4 alignment guard),
   a pipeline, a dispatch branch with grid `(N+63)/64 × (M+127)/128`, and a
   tuner entry.
3. Run `gemm_tune.py` for a matched-clock A/B at 256² and 1024²; promote
   variant 7 only if it wins.
4. Re-run GEMM/graph/eager benchmarks, 43/43 parity, MNIST/CIFAR, CTest both.
5. FINDINGS with all numbers including negatives. No tolerance change.

## Honesty

The autotune cache decides; a slower variant is never forced. All public
reports state missed targets plainly and do not cherry-pick clock samples.
