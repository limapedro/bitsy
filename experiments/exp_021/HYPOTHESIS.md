# exp_021 — L2-friendly block order for the vec4 GEMM

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 6 (GEMM ladder)

## Prerequisite

exp_020 leaves the 1024² GEMM at 0.69–0.75 of PyTorch CUDA (target ≥0.80) and
a captured training step at 0.304 of PyTorch CUDA graphs. The GEMM ladder
experiments so far covered register blocking (exp_008), vec4 + double
buffering (exp_009), autotune + a 16-deep K step (exp_010, negative), but not
the **L2-friendly block order** rung. Current `gemm_rb_vec4` maps
`gl_WorkGroupID.x/y` directly to output tiles in row-major order; on TU116
(1.5 MB L2) a group swizzle may improve A/B tile reuse.

## Hypothesis

A group-swizzled tile order (4×4 tile groups, consecutive workgroups covering
a small group of output tiles) improves L2 locality of the shared A/B tiles
without changing the arithmetic, raising large-shape GEMM throughput.

## Predictions

- At 1024², the swizzled variant is **≥1.05× faster** than the current
  `gemm_rb_vec4` winner and reaches **≥0.80** of PyTorch CUDA at matched clock.
- It is never slower on the small (256²) tiled regime; if it is, the autotune
  cache keeps the old variant there.
- Numeric parity stays green: 43/43 including the gradient gate; GEMM output
  within the existing 1e-5 tolerance (the swizzle only changes tile order, not
  accumulation order per output element).
- If the prediction fails, record the negative and keep the existing cache.

## Measurement

1. Snapshot current GEMM/graph/eager JSON and training reports.
2. Add `gemm_rb_vec4_swz.comp` (4×4 group swizzle) as variant 6, selectable by
   `BT_GEMM_VARIANT=6`, with the same vec4 alignment guard.
3. Run `benchmarks/ops/gemm_tune.py` at 256² and 1024² to compare variants 4
   (current winner) and 6 at matched clock; promote 6 into the cache only if
   it wins.
4. Re-run gemm/graph/eager benchmarks, 43/43 parity, MNIST/CIFAR, CTest both.
5. FINDINGS with all numbers including negatives. No tolerance change.

## Honesty

Arithmetic per output element is unchanged, so parity should be exact; any
drift is measured. If the swizzle does not help, it stays out of the cache and
the negative result is recorded. All public reports state missed targets.
