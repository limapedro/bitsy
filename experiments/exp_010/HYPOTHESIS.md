# exp_010 — GEMM rung: 16-deep K step and measured autotune cache

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 6 (continuation of exp_009)

## Prerequisite

exp_009: 1024² GEMM 1.224 → 1.178 ms (1754 → 1823 GFLOPS, ratio 0.77 → 0.78),
but its ≥1.08× prediction and the ≥0.80 target were both missed. Remaining
causes listed: small 8-deep K step, no runtime/on-disk autotune cache, no L2
tile-order experiment.

## Hypothesis

Doubling the K step from 8 to 16 halves the number of barriers per tile and
gives each thread 128 FMAs between synchronizations, improving throughput.
An on-disk autotune cache populated by `benchmarks/ops/gemm.py` lets the runtime
select the measured winner per `(device, M, N, K)` instead of a hard-coded size
threshold, so small and large shapes each get their best kernel.

## Predictions

- 16-deep K step is ≥1.05× faster than the 8-deep vec4 kernel at 1024².
- Ratio versus PyTorch CUDA reaches ≥0.80 at 1024².
- Cached autotune selects the measured winner and parity stays 42/42.
- FP32 parity within matmul rtol/atol 1e-5 is preserved.

## Measurement

1. Snapshot current `benchmarks/results/gemm.json` into `before.json`.
2. Add `BT_GEMM_VARIANT=5`: 16-deep K step, double-buffered shared tiles
   (32 KB), vec4 loads, same 128×128 tile / 8×8 thread tile.
3. Benchmark variants 4 and 5 at 256² and 1024²; save `variant_4.json`,
   `variant_5.json`.
4. Have `gemm.py` write `benchmarks/results/gemm_autotune.txt` with measured
   winners; make the C runtime read this cache at first matmul.
5. Re-run final benchmark into `after.json`, plus CTest and 42-case parity.
6. Write FINDINGS including negatives and the cache hit/miss evidence.

## Honesty

No PyTorch source is copied; only tile parameters observed in the public
profile guide the design. Wall time keeps `bt.flush()+bt.wait()` against
`torch.cuda.synchronize()`. Failed predictions are recorded, not removed.
