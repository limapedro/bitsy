# exp_002 — Findings: tiled SGEMM

**Date:** 2026-11-21
**Decision:** keep the tiled kernel (opt-in via `BT_GEMM_TILED=1`).

## What was tried

Added a 16x16x16 shared-memory tiled SGEMM shader (`gemm_tiled.comp`). Each
workgroup loads A/B tiles once into `shared float[16][16]` arrays, synchronizes,
accumulates 16 products per tile, then synchronizes again. The naive
one-lane-per-output shader remains selectable with `BT_GEMM_TILED=0`.

## Measurements (same command, same GPU; raw JSON in this folder)

| n | naive ms | tiled ms | speedup | naive GFLOPS | tiled GFLOPS | tiled ratio vs CUDA |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.237 | 0.146 | 1.62x | 141 | 229 | 0.31 |
| 1024 | 10.224 | 4.337 | 2.36x | 210 | 495 | 0.22 |

PyTorch CUDA times were 0.043 ms (256) and 0.908–0.961 ms (1024). Both modes
pass the full parity suite: 40/40 cases each, including the 3 matmul cases
against PyTorch CUDA with rtol/atol 1e-5.

## Interpretation

The hypothesis was supported: shared-memory reuse more than doubled throughput
on the 1024 matrix and crossed the 1.5x prediction. The larger-tile prediction
that small matrices benefit less held (1.62x vs 2.36x).

The M5 performance target is still **missed**: 495 GFLOPS is ~0.22 of PyTorch
CUDA on this GPU, versus a >= 0.80 initial target. Remaining causes, in order:
one submit + fence wait per GEMM, no register blocking or vectorized loads, no
autotuning of tile sizes, no fused epilogues, and no device timestamp queries
to separate launch overhead from kernel time. Each needs its own experiment.

The tiled kernel is kept as an explicit opt-in rather than silently replacing
the naive one, so before/after remains verifiable with one environment variable.
