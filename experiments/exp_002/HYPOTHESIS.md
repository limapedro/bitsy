# exp_002 — Shared-memory tiled SGEMM for the Vulkan backend

**Date:** 2026-11-21
**Owner:** Bitsy M5

## Hypothesis

The current GEMM shader computes one output per lane and re-reads both input
matrices from device memory for every output element. On a GTX 1660 Super with
no tensor cores, reusing A/B tiles from shared memory should substantially
improve arithmetic intensity and therefore throughput on medium/large square
matrices.

## Prediction

- On (1024, 1024) x (1024, 1024) FP32 GEMM, tiled kernel wall time is at least
  1.5x lower than the naive kernel.
- Parity stays green: matmul cases still match PyTorch CUDA within the existing
  rtol/atol 1e-5 tolerance.
- Small matrices (256) may improve less because of shared-memory setup.

## Measurement

1. Snapshot the naive result with `python benchmarks/ops/gemm.py` (env
   `BT_GEMM_TILED=0`) into `before.json`.
2. Add a 16x16x16 shared-memory tiled shader and select it with
   `BT_GEMM_TILED=1`; re-run the identical benchmark into `after.json`.
3. Re-run `python tests/parity/run.py` in both modes.
4. Keep the tiled kernel only if the prediction is met. If it is slower, keep
   both kernels and record the negative result; do not delete the experiment.

## Honesty

Wall time includes one submit + fence wait per GEMM on the Bitsy side, while
PyTorch CUDA is synchronized per iteration too. Device timestamp queries are
still absent, so no device-time claim is made. Ratios versus PyTorch CUDA will
be reported as measured, including misses.
