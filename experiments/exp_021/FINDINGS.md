# exp_021 — L2-friendly block order for the vec4 GEMM (negative result)

**Date:** 2026-11-21
**Scope:** Phase 6 GEMM ladder. No tolerance changed. Parity: 43/43 including
the gradient gate, CTest 1/1 Vulkan and CPU-only.

## Implementation

Added `gemm_rb_vec4_swz.comp` (variant 6): a 4×4 group-swizzled tile order
over a linear workgroup grid, otherwise bit-identical to the `gemm_rb_vec4`
winner. Exposed as `BT_GEMM_VARIANT=6`, with the same vec4 alignment guard, and
added to the tuner's variant list. The autotune cache decides whether it is used.

## Controlled A/B (gemm_tune.py, same process and clock)

| shape | variant 4 (current) | variant 6 (swizzle) | ratio 6/4 |
|---|---:|---:|---:|
| 256² | 0.172 ms | 0.227 ms | **1.32× slower** |
| 1024² | 0.985 ms | 1.193 ms | **1.21× slower** |

The group swizzle is clearly **worse** on both shapes; the row-major tile order
already reuses A/B tiles well enough in L2 for these sizes, and the extra
index arithmetic plus less favourable access pattern costs more than any
locality gain. The tuner therefore kept variant 4 at both shapes (the 256²
cache entry changed from 1 to 4 by 0.002 ms; 1024² stayed at 4). Variant 6 is
retained as a measured negative result and is never auto-selected.

## Predictions

- "≥1.05× faster at 1024² and ≥0.80 ratio": **not met** (1.21× slower).
- "never slower at 256²": **not met** (1.32× slower).
- "parity unchanged": **met** (43/43, gradient gate included).
- "cache keeps old variant on failure": **met**.

## Post-change measurements

Gemm.py at 1928 MHz: 1024² 1.055 ms vs torch 0.721 (ratio 0.68) — clock variance
makes the headline ratio range 0.68–0.76 across runs; the ≥0.80 target is still
missed. Captured step 0.220–0.222 ms (ratio 0.301–0.303) and eager step
1.84–1.92 ms (ratio 0.58–0.62) are within run variance of exp_020 — the swizzle
is not selected, so it has no effect on training.

## Interpretation / next

The L2 block-order rung of the GEMM ladder is now measured and closed as a
negative. Remaining untried rungs are fused epilogues, packed f16 and possibly
a different tile aspect ratio; the next experiment should pick one of those,
or return to the ~15 scalar elementwise dispatches in the training step. Every
change must still pass the 43-case gate including gradient flow.

Raw data in `experiments/exp_021/{swizzle_ab.json,before_*,after_*,before/after_gemm_autotune.txt}`.
