# exp_009 — GEMM rung: vec4 loads + double-buffered shared memory

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 6 (continuation of exp_008)

## Prerequisite

exp_008 delivered a 3.54× GEMM speedup (1024²: 4.329 → 1.224 ms, 496 →
1754 GFLOPS, ratio 0.22 → 0.77) but missed the ≥0.80 target by 0.03.
`experiments/exp_008/FINDINGS.md` lists missing vec4 loads and double buffering
as the next rungs. Calibrated ceilings: DRAM 304.4 GB/s, FP32 5.49 TFLOP/s.

## Hypothesis

The register-blocked kernel still loads shared-memory tiles with scalar loads
and serializes tile load → barrier → compute → barrier. Packing global loads
into `vec4` and double-buffering the shared tiles will overlap the next tile
transfer with the current compute, closing the remaining 0.03 ratio gap.

## Predictions

- vec4 + double buffering is ≥1.08× faster than exp_008's `gemm_rb` at 1024².
- Ratio versus PyTorch CUDA reaches ≥0.80 at 1024².
- Parity (42 cases, rtol/atol 1e-5 for matmul) remains green.
- The specialization only applies when strides/alignment permit it; otherwise
  the runtime falls back to `gemm_rb` with identical numerics.

## Measurement

1. Snapshot current `benchmarks/results/gemm.json` into `before.json`.
2. Add `BT_GEMM_VARIANT=4` shader with vec4 loads and double-buffered smem.
3. Benchmark variants 2 and 4 at 256² and 1024²; save `variant_2.json` and
   `variant_4.json`.
4. Wire measured winner into autotune; re-run final benchmark into `after.json`.
5. Re-run CTest + parity; write FINDINGS including negatives and any fallback.

## Honesty

If vec4 specialization does not meet the prediction, keep it as a selectable
variant and record the negative result. No tolerance changes. Wall time keeps
`bt.flush()+bt.wait()` against `torch.cuda.synchronize()`.
