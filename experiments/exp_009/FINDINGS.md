# exp_009 — Findings: vec4 loads + double-buffered GEMM

**Date:** 2026-11-21
**Scope:** continuation of the GEMM ladder (top GAP_LEDGER item). Parity
re-verified: CTest 1/1, 42/42 cases, gradcheck 1.19e-07 / 5.96e-08.

## Implementation

- Added `gemm_rb_vec4.comp`: 128×128×8 register-blocked GEMM, 16×16 workgroup,
  8×8 thread tile, `vec4` global loads, **two shared-memory tile buffers**
  (`As[2][8][128]`, `Bs[2][8][128]`, 16 KB) so the next tile loads while the
  current one is consumed.
- Host checks `strides[1] == 1` and `strides[0] % 4 == 0` for both inputs before
  selecting this variant; otherwise it safely falls back to `gemm_rb` with
  identical numerics. Selectable with `BT_GEMM_VARIANT=4`.
- Fixed an earlier wrong vec4 mapping (it packed four K values where a row was
  needed, producing up to 216 absolute error). The corrected mapping packs
  four K values per row for A and four columns per K for B; max error versus
  PyTorch CUDA is now 2.1e-4 at 1024², inside the matmul rtol/atol 1e-5 policy.

## Measurements

| n | variant | Bitsy ms | GFLOPS | ratio vs CUDA |
|---:|---|---:|---:|---:|
| 1024 | 2 (rb scalar) | 1.213 | 1770 | 0.77 |
| 1024 | 4 (rb vec4 + double buffer) | **1.167** | **1840** | 0.77 |
| 256 | 1 (tiled, autotune) | 0.134 | 250 | 0.31 |

Autotune before → after (exp_008 → exp_009): 1024² **1.224 → 1.178 ms**
(1.04×), **1754 → 1823 GFLOPS**, ratio **0.77 → 0.78**.

Raw snapshots: `experiments/exp_009/before.json`, `after.json`,
`variant_2.json`, `variant_4.json`.

## Predictions

- "≥1.08× faster than gemm_rb at 1024²": **not met** (1.04× absolute ms).
- "ratio ≥0.80": **not met** (0.78, short by 0.02).
- "parity stays green": **met**.
- "specialization falls back safely": **met** (stride check in host code).

This is a small but real improvement; it is kept as the default for aligned
large shapes because the absolute time is lower, but the campaign target is
still missed and the prediction is recorded as failed.

## Interpretation / remaining causes

The kernel now reaches 1823 GFLOPS (33% of the 5.49 TFLOP/s ceiling). Remaining
losses: one submit + fence wait per iteration (~36 µs of the ~1.18 ms wall),
no runtime autotune cache, no L2-friendly block order, no fused epilogue, and
shared-memory loads still limited by the 8-deep K step. The next rungs need
their own `exp_NNN` with before/after measurements.

## Next

Continue the GEMM ladder (larger K step / double-buffer depth, L2-friendly tile
order, on-disk autotune cache) before moving to the next GAP_LEDGER item.
