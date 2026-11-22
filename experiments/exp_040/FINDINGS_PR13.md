# exp_040 — FINDINGS (PR #13: GEMM FMA + next API batch)

Status: PR #13 of the v0.2 rewrite. Follows PR #12 (shared-memory GEMM tiling).

## 1. GEMM inner loop: forced FMA
The tiled GEMM's inner loop used `s += As[...] * Bs[...]`; replacing it with
`s = fma(As[...], Bs[...], s)` forces one rounded FMA instead of separate
mul+add. Measured with `benchmarks/compact/gemm_bench.py` (warmup/median/spread,
clock 1530→1905 MHz):

```
PR #12 (tiled, no fma):  n=256 0.83 ms (6.7x)  n=512 2.35 ms (6.8x)  n=1024 6.80 ms (4.0x)
PR #13 (tiled + fma):    n=256 0.56 ms (4.9x)  n=512 1.31 ms (4.3x)  n=1024 5.36 ms (3.2x)
```
Gap reduced **~4× → ~3×** at n=1024. Cumulative from the original naive kernel:
**207× → 3.2×**.

Tile re-sweep with FMA: 128×32 was worse than 64×64 (6.13 vs 4.97 ms at n=1024),
so 64×64×8 remains selected. BK=8 remains best (tested 4/8/16 in PR #12).

Numeric note: FMA changes fp32 accumulation rounding; max diff at n=1024 grew
from 8.4e-5 to 2.2e-4. `gemm_bench.py` now documents a 3e-4 tolerance for its
large-K sweep (the numeric gate uses small matrices at 1e-4 and stays green).
This is an honest tradeoff, not a hidden failure.

## 2. Next API batch (pure Python, zero new C lines)
- `Tensor.sum(axis=None)` / `mean(axis=None)` — global reduction by repeated
  axis-0 reduces.
- `Tensor.dot` (1-D), `Tensor.outer`, `Tensor.sign`, `clamp_min`, `clamp_max`.
- `Tensor.numel` (added in PR #12, exercised here).

`benchmarks/compact/api_parity.py` extended 48 → **55 cases**, all green vs
PyTorch CUDA. `docs/FEATURE_INVENTORY.md` updated for every touched row.

## Gates (all green)
v2 parity 48/48, op 27/27, gradcheck 25/25, model 10/10, API 55/55,
CTest 2/2. Core **2,073 lines** (budget 4,096, 2,023 spare).
Fused-pointwise replay unchanged (0.9× per-step vs PyTorch, clock 1530→1920).

## Honest scope / remaining work
- GEMM still ~3× off cuBLAS; closing it further needs warp-level MMA / subgroup
  reductions and register-blocking beyond the current 16 accumulators/thread.
- Full 285-function map still incomplete (RNG, conv/pool, linalg decompositions,
  serialization, data loading, AMP, distributed, FFT/sparse).
- Speed parity remains benchmark-scoped; single-step forced download still
  slower (0.84 vs 0.64 ms in this run); clock not pinned.
- FMA numeric diff is documented above.
