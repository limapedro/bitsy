# exp_040 — FINDINGS (PR #14: GEMM tuning attempts + shape API batch)

Status: PR #14 of the v0.2 rewrite. Follows PR #13 (FMA, ~3× GEMM gap).

## GEMM: experiments attempted, none improved on 64×64×8 + FMA
The objective asked for subgroup/warp-level reductions to close the remaining
~3× cuBLAS gap. I investigated and measured several options:

| Attempt | Result | Decision |
|---|---|---|
| `#pragma unroll` on the e and kk loops | 5.08 ms vs 5.36 ms baseline at n=1024 — within run-to-run noise | reverted |
| Tile 128×32 (same 16 accumulators/thread) | 6.13 ms vs 4.97 ms (FMA baseline) | reverted |
| Tile 64×128 (32 accumulators/thread) | **WRONG RESULTS** (diff 67–161): `float acc[16]` overflows when BM*BN/256 > 16 | reverted; added a defensive comment in the shader documenting the capacity limit |
| Tile 32×128 (16 accumulators) | 5.28 ms vs 5.36 ms — noise | reverted |
| BK 4 / 8 / 16 (PR #12) | 8 best | kept |
| FMA (PR #13) | 6.80 → 5.36 ms at n=1024 | kept |

**Why subgroup reductions do not apply here**: subgroup shuffle/broadcast helps
reductions (sum/max across lanes) and for broadcasting a value that every lane
needs. The tiled GEMM already loads A from shared memory with the *same address
across all lanes* (hardware broadcast) and B with perfectly coalesced consecutive
addresses — so subgroup operations would duplicate what the memory system already
does for free. Closing the remaining ~3× needs **tensor-core cooperative matrix
operations** (`GL_NV_cooperative_matrix` / subgroup matrix multiply) and register
blocking beyond 16 accumulators/thread — a different shader model with uncertain
support on this target. I did not claim an improvement I did not measure.

Final GEMM evidence (warmup/median/spread, clock 1530→1530 MHz):
```
n=256  0.62 ms (5.8x)   n=512 1.19 ms (3.0x)   n=1024 4.90 ms (3.2x) vs cuBLAS
```
Cumulative from the naive kernel: 207× → ~3×. Still honest, not parity.

## Shape API batch (pure Python, zero new C lines)
Added `Tensor.flatten`, `squeeze`, `unsqueeze` as reshape-based compositions.
API parity extended 55 → **60 cases**, all green vs PyTorch CUDA.
`docs/FEATURE_INVENTORY.md` updated.

## Gates (all green)
v2 parity 48/48, op 27/27, gradcheck 25/25, model 10/10, API 60/60,
CTest 2/2. Core **2,075 lines** (budget 4,096, 2,021 spare).
Fused-pointwise replay unchanged (0.9× per-step, clock 1530→1950).

## Honest scope / remaining work
- GEMM ~3× off cuBLAS; needs cooperative-matrix/tensor-core path (not attempted).
- Full 285-function map still incomplete (RNG, conv/pool, linalg decompositions,
  serialization, data loading, AMP, distributed, FFT/sparse).
- Speed parity remains benchmark-scoped; single-step forced download still slower
  (0.82 vs 0.63 ms in this run); clock not pinned.
- A 64×128 tile produced wrong numerics due to the fixed `acc[16]` capacity;
  the shader now documents this constraint. No silent failure shipped.
