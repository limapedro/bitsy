# exp_030 — packed f16 storage with FP32 accumulate GEMM

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 6 (GEMM ladder, final rung)

## Prerequisite

The standalone FP32 GEMM is 0.68–0.76 of PyTorch CUDA (target ≥0.80) after
testing register blocking, vec4, double buffering, split-K, L2 swizzle, fused
epilogue and two tile aspect ratios. `VK_KHR_shader_float16_int8` is supported
but the Phase 1 f16vec2 FMA rate was never measured (HARDWARE_LEDGER line 71).
The campaign lists a separate target class: GEMM with FP16 storage / FP32
accumulate ≥0.80.

## Hypothesis

Converting A/B tiles to f16 in shared memory and using packed `f16vec2`
products with FP32 accumulation halves tile storage and may double arithmetic
throughput on TU116, giving a faster GEMM variant for the f16-storage class.

## Predictions

- The f16 variant is **≥1.05× faster** than the FP32 tiled winner at 1024² and
  reaches **≥0.80** of PyTorch CUDA at matched clock for the f16-storage class.
- Output differs from the FP32 GEMM by ~1e-3 relative (f16 input rounding),
  which is acceptable for the f16-storage class but must be measured and
  reported; it is NOT added to the 43-case fp32 harness or autograd.
- If it is slower or no faster, it is kept only as a negative-result variant,
  never auto-selected for fp32 workloads.
- The mandatory 43-case gate stays green because training/autograd are
  unchanged.

## Measurement

1. Snapshot current GEMM JSON and autotune cache.
2. Add `gemm_f16.comp` (f16vec2 shared tiles, FP32 accumulate) as variant 9,
   a pipeline, dispatch branch, and tuner entry.
3. Verify numeric drift vs FP32 on a representative GEMM with explicit
   rtol/atol for the f16 class (reported, not hidden).
4. Run the median tuner twice; promote only if it wins by >5%.
5. Re-run GEMM benchmark, 43/43 parity, graph/eager, MNIST/CIFAR, CTest both.
6. FINDINGS with all numbers including negatives. No tolerance relaxed.

## Honesty

f16 input rounding is intrinsic; the result is measured and reported. The
43-case gate remains mandatory for any merged change.
