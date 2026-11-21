# exp_008 — GEMM ladder: register blocking, conflict-free smem, split-K, autotune

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 6 (top GAP_LEDGER item)

## Prerequisite evidence

`GAP_LEDGER.md` ranks GEMM #1 at **155.5 µs** wall gap / ratio 0.29.
The current Vulkan GEMM achieves **3.7% of FP32 peak**; the profiler shows
16×16 shared-memory tiles. PyTorch profile shows
`volta_sgemm_128x32_nn` + `splitKreduce_kernel<32,16>` (18.0 + 5.1 µs/call).

## Hypothesis

The GEMM gap is dominated by method, not hardware: too little work per thread,
one barrier per small tile, non-vectorized loads, and only 4 output tiles for
256² (wave quantization on 22 SMs). A register-blocked 128×128×8 kernel with
conflict-free shared memory, followed by split-K for small matrices, will close
a material part of the 155 µs gap.

## Predictions

- Register-blocked GEMM is ≥1.5× faster than the current 16×16 tiled kernel at
  1024², and reaches ≥10% of FP32 peak (from 3.7%).
- Split-K improves 256² by ≥1.3× by filling more of the 22 SMs.
- Autotune picks the fastest variant per shape; both modes remain bit-parity
  safe within the existing matmul tolerance (rtol/atol 1e-5).
- No regression in the 42-case parity suite.

## Measurement

1. Snapshot `benchmarks/results/gemm.json` (current tiled) into `before.json`.
2. Add variants selectable by `BT_GEMM_VARIANT`: 0 naive, 1 tiled 16²,
   2 register-blocked 128×128×8, 3 register-blocked + split-K=4.
3. Benchmark all variants at 256² and 1024² with `benchmarks/ops/gemm.py`
   (flush+wait per iteration, median+spread, clock logged).
4. Wire the measured winner per shape into the runtime auto-selection and
   re-run the final benchmark; record `after.json`.
5. Re-run parity/gradcheck/conv_check. Write FINDINGS.md including negatives.

## Honesty

Wall time includes submit+wait on both sides. Device time is Bitsy-only from
timestamp queries. If a variant is slower, it stays in the tree as a documented
negative result. No copyright/code from cuBLAS is used — only the public tile
parameters observed in the PyTorch profile.
