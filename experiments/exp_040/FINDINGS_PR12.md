# exp_040 — FINDINGS (PR #12: dead-script cleanup, GEMM tiling, more APIs)

Status: PR #12 of the v0.2 rewrite. Follows PR #11 (legacy cutover complete).

## 1. Dead legacy benchmark scripts removed
Deleted 15 scripts under `benchmarks/models/` and `benchmarks/ops/` that still
`import bitsy` (the package deleted in PR #11). All result JSONs were kept as
historical evidence; `benchmarks/ops/torch_kernel_profile.py` (torch-only) and
all active `benchmarks/compact/` scripts remain functional. No active script
imports the deleted package now.

## 2. GEMM tile autotune — the biggest remaining perf lever
Baseline (before): the GEMM shader had **no data reuse** — every output element
looped over K reading A/B from global memory. New `benchmarks/compact/gemm_bench.py`:
```
BEFORE (naive, warmup/median/spread/clock 1530->1935 MHz):
  n=256    5.76 ms  vs torch 0.130 ms   44.2x
  n=512   32.84 ms  vs torch 0.303 ms  108.5x
  n=1024 304.38 ms  vs torch 1.471 ms  206.9x
```
Rewrote `src/rt/shaders/gemm.comp` with shared-memory tiling: cooperative
BM×BK / BK×BN tile loads into `shared`, barrier, per-thread accumulation with
a compile-time-unrolled `acc[BM*BN/256]` array, transA/transB/bias preserved.

Tile/BK sweep (median ms, lower is better):
```
          16x16x16   32x32x8   64x64x8
n=256       0.565     0.586     0.557   <- 64x64
n=512       1.910     1.414     1.305   <- 64x64
n=1024      8.107     5.791     5.292   <- 64x64
BK for 64x64: 4 -> 5.38 ms, 8 -> 5.29 ms, 16 -> 6.22 ms (n=1024) -> BK=8
```
Selected 64×64×8 via `gemm_variant()` (env override `BT_GEMM_VARIANT` kept for
re-tuning). Dispatch group count now uses the selected BM/BN (was hardcoded 16).

AFTER (final, clock 1530->1935 MHz):
```
  n=256    0.83 ms  vs torch 0.123 ms    6.7x
  n=512    2.35 ms  vs torch 0.343 ms    6.8x
  n=1024   6.80 ms  vs torch 1.694 ms    4.0x
```
Gap reduced **~207x -> ~4x** at n=1024 (and 44x -> 6.7x at n=256). Numeric diff
stays within 1e-4 (8.4e-5 at n=1024). GEMM is still 4-7x off cuBLAS — this is an
honest improvement, not parity.

## 3. More API surface (pure Python, zero new C lines)
Added `Tensor.min`, `Tensor.all`, `Tensor.any`, `Tensor.numel`, and
`functional.isclose` / `functional.allclose`. `benchmarks/compact/api_parity.py`
extended from 44 to **48 cases**, all passing vs PyTorch CUDA.
`docs/FEATURE_INVENTORY.md` updated.

## Gates (all green)
v2 parity 48/48, op 27/27, gradcheck 25/25, model 10/10, API 48/48,
CTest 2/2. Core **2,073 lines** (budget 4,096, 2,023 spare).
Fused-pointwise replay perf unchanged (0.9x per-step vs PyTorch, clock logged).

## Honest scope / remaining work
- GEMM still 4-7x off PyTorch; further gains need register blocking, warp-level
  MMA / subgroup reductions, and larger-tile occupancy tuning.
- Full 285-function map still incomplete (RNG, conv/pool, linalg decompositions,
  serialization, data loading, AMP, distributed, FFT/sparse).
- Speed parity remains benchmark-scoped; single-step forced download still
  slower (0.89 vs 0.85 ms in this run); clock not pinned.
- The `gemm_bench.py` tol is 1e-4; n=1024 diff 8.4e-5 is close to it. Larger
  matrices may need a documented tolerance review.
