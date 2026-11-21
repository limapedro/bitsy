# exp_030 — packed f16 GEMM rejected (negative result)

**Date:** 2026-11-21
**Scope:** Phase 6 GEMM ladder, final rung. **No code retained**: the attempted
variant was removed; the tree is functionally identical to exp_029. Parity:
43/43 incl. gradient gate, CTest 1/1 Vulkan.

## Attempt

Added a tiled GEMM (`gemm_f16.comp`) that stores A/B tiles as `f16vec2` in
shared memory (two consecutive k values per entry), converts to fp32 for the
multiply/accumulate, and writes an fp32 result. Wired as variant 9 with a
pipeline and a tuner entry.

## Result: negative on both speed and correctness

- 1024²: **3.508 ms** (612 GFLOP/s, ratio 0.26) versus ~1.0–1.3 ms for the
  FP32 variants — **~3× slower**. The strided B loads and fp32 conversions
  dominate; TU116's CUDA cores give no packed-f16 throughput advantage
  (the Phase 1 note "packed f16vec2 FMA: compiled, not measured" is now
  answered: not useful here without tensor cores).
- Correctness: the first implementation had swapped B tile indices; after
  fixing that, a 2×16×16 K=16 case still produced `[[0.33, 0.0], [2.57, 0.0]]`
  versus `[[24.8, 26.0], [63.2, 66.96]]` expected. Replacing packed f16
  multiply with explicit fp32 conversion of f16 values did not fix it, so the
  `f16vec2` shared-memory path is unreliable on this driver/hardware.
- Because a wrong kernel must not remain selectable, **variant 9 was removed
  entirely** (shader, pipeline, dispatch branch, tuner entry). No tolerance
  was relaxed and no training path was changed.

The ≥1.05× / ≥0.80 predictions were **not met**. The final tree was
re-verified: 43/43 parity, CTest 1/1, GEMM 1024² 1.282 ms / ratio 0.71 at
1530 MHz, captured/eager step unchanged from exp_029.

## Interpretation / next

The GEMM ladder is now comprehensively measured: register blocking, vec4,
double buffering, split-K, L2 swizzle, fused epilogue, 128×64 and 64×128 tile
aspect ratios, and packed f16. The best FP32 variant wins; none reaches the
≥0.80 standalone target on TU116. Further gains likely require a fundamentally
different approach (algorithmic fusion at the step level, or accepting that
PyTorch's cuBLAS is near peak here). All future changes must still pass the
43-case gate.

Raw console measurements are recorded above; `before_gemm.json` and
`before_gemm_autotune.txt` snapshot the starting state.
