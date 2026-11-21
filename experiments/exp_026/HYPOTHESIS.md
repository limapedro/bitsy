# exp_026 — median-based GEMM tuning and the 64x128 tile rung

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 6 (GEMM ladder + measurement reliability)

## Prerequisite

exp_025 showed a single-sample tuner run flipping the 1024² winner to a
variant exp_021 had measured 21% slower (1.002 vs 1.005 ms — 0.3% noise).
Selecting kernels from one sample is a performance regression risk. The
128×64×8 rung was negative; the other aspect ratio listed by the campaign,
64×128×8, has not been measured.

## Hypothesis

Using the median of several timed samples per variant prevents noise-driven
selection of a slower kernel. Under that reliable measurement, a 64×128×8
vec4 GEMM (8×16 workgroup, 8×8 thread tile) may beat the current 128×128×8
winner at 1024² by increasing output-tile count without reducing K-reuse.

## Predictions

- The median-based tuner selects the same variant across repeated runs
  (no 0.3% noise flips) and never selects a variant measured >5% slower in a
  controlled A/B.
- Variant 8 (64×128×8) is **≥1.05× faster** at 1024² and the standalone GEMM
  ratio reaches **≥0.80** at matched clock.
- If it is slower, the cache keeps the current winner and the rung is recorded
  as a negative.
- Parity stays green: 43/43 including the gradient gate; GEMM output within
  1e-5 tolerance.
- Improving the tuner must not change any kernel's arithmetic or tolerances.

## Measurement

1. Snapshot current GEMM/graph/eager JSON and the autotune cache.
2. Change `gemm_tune.py` to take the median of 3 samples per variant (dataclass
   config, no argparse) and to write the cache from medians.
3. Add `gemm_rb_vec4_64x128.comp` as variant 8 with a dedicated pipeline and
   dispatch branch; grid `(N+127)/128 × (M+63)/64`.
4. Run the tuner twice; verify winner stability. Promote variant 8 only if it
   wins by >5%.
5. Re-run GEMM/graph/eager benchmarks, 43/43 parity, MNIST/CIFAR, CTest both.
6. FINDINGS with all numbers including negatives. No tolerance change.

## Honesty

The cache is the authority; a slower variant is never forced. All public
reports state missed targets plainly and do not cherry-pick samples.
