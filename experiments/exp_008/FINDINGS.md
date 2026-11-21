# exp_008 — Findings: GEMM ladder

**Date:** 2026-11-21
**Scope:** top-ranked GAP_LEDGER item. Numeric parity re-verified: CTest 1/1,
42/42 parity cases, all matmul cases within the existing rtol/atol 1e-5.

## Variants implemented

- `BT_GEMM_VARIANT=0`: naive one-lane-per-output (baseline).
- `1`: 16×16 shared-memory tiled (previous default).
- `2`: **register-blocked 128×128×8**, 16×16 workgroup, 8×8 thread tile,
  conflict-free transposed A/B shared layouts, 64 accumulators per lane.
- `3`: variant 2 with **split-K=4** into a deferred-free scratch buffer plus a
  reduction dispatch. Scratch buffers are released in `bt_vk_wait()` after the
  timeline wait, never before the GPU is done.
- `-1` (default): autotune by shape from the measurements below.
- Added `k_start`/`k_len` to the matmul push constants so split-K processes
  real K ranges. An earlier implementation wrote the full K four times and
  produced 4× the correct result; the unit test caught it and it was fixed.

## Measured variants (median wall, `benchmarks/ops/gemm.py`)

| n | variant | Bitsy ms | GFLOPS | ratio vs CUDA |
|---:|---|---:|---:|---:|
| 256 | 0 naive | 0.284 | 118 | 0.25 |
| 256 | 1 tiled | 0.143 | 235 | 0.50 |
| 256 | 2 rb | 0.163 | 206 | 0.25 |
| 256 | 3 rb+splitK | 0.232 | 145 | 0.18 |
| 1024 | 0 naive | 12.722 | 169 | 0.07 |
| 1024 | 1 tiled | 4.265 | 504 | 0.21 |
| 1024 | 2 rb | **1.211** | **1773** | **0.73** |
| 1024 | 3 rb+splitK | 1.558 | 1378 | 0.57 |

Split-K is a **negative result**: the extra reduction dispatch costs more than
the occupancy gained on 22 SMs for these sizes. It stays in the tree as a
selectable variant and the reason is recorded here.

## Before → after (autotune default)

| n | before ms | after ms | speedup | before ratio | after ratio | before GFLOPS | after GFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.177 | 0.164 | 1.08× | 0.27 | 0.38 | 189 | 205 |
| 1024 | 4.329 | 1.224 | **3.54×** | 0.22 | **0.77** | 496 | 1754 |

Raw snapshots: `experiments/exp_008/before.json`, `after.json`, `variant_0..3.json`.

## Predictions

- "≥1.5× faster than tiled at 1024": **met** (3.54×).
- "≥10% of FP32 peak": **met** (1754/5490 = 32%).
- "Split-K improves 256² ≥1.3×": **not met** (it is 1.4× slower). Negative
  result preserved.
- "Autotune picks fastest per shape": **met for the measured shapes**
  (tiled at 256², rb at 1024²). The selection is a simple size threshold, not
  a cached runtime autotuner; that is a known limitation.

## Targets

The GEMM target was ≥0.80 initially. Final ratio **0.77**, so the target is
still **missed by 0.03**, despite a 3.5× improvement. Remaining causes:
one submit+fence per iteration (~36 µs host overhead), no double buffering or
vectorized (`vec4`) global loads, no L2-friendly block ordering, no fused
epilogue, and no runtime autotune cache. Each needs its own experiment.

## Next

Do not move to the next GAP_LEDGER item yet: continue the GEMM ladder with
vec4 loads / double buffering and re-measure, with a new `exp_NNN`.
