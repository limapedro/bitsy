# exp_012 — vec4/aligned elementwise kernels

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 6 (kernels)

## Prerequisite

exp_011 at a fixed 1530 MHz: elementwise ratios 0.34–0.54 versus the ≥0.95
target. A 20-add chain reached ~86% of measured DRAM ceiling (304 GB/s), while
individual large ops reached only 68–101 GB/s. PyTorch's profiler showed
`at::native::vectorized_elementwise_kernel<4, ...>` — vec4 loads/stores. The
current Hagane elementwise shader uses one scalar float per lane and a runtime
odometer with div/mod for broadcasting.

## Hypothesis

Most of the remaining elementwise gap is memory-transaction inefficiency and
integer index math. A vec4 fast path for contiguous, equal-shaped, 4-aligned
tensors will use one 16-byte load/store per lane and a linear index with no
runtime div/mod, raising large-tensor bandwidth toward ≥90% of the measured
304 GB/s ceiling.

## Predictions

- add/mul/relu at 1M elements improve by ≥1.3× versus the scalar kernel.
- Achieved bandwidth on 1M tensors reaches ≥180 GB/s (≈60% of measured
  ceiling) as an intermediate step toward the ≥90% long-term target.
- Non-contiguous/broadcast/special-value cases keep the scalar path and all
  42 parity cases stay green (exact modes unchanged).
- Ratio versus PyTorch CUDA improves for large tensors.

## Measurement

1. Snapshot `benchmarks/results/elementwise.json` into `before.json`.
2. Add `elementwise_vec4.comp` and host alignment check; select vec4 only for
   contiguous equal-shaped inputs with 16-byte-aligned offsets/sizes.
3. Re-run `benchmarks/ops/elementwise.py` and `gemm.py` (GEMM is unaffected but
   confirms no regression), plus full parity/gradcheck/conv_check.
4. Save `after.json`; write FINDINGS including negatives and fallback coverage.

## Honesty

No tolerance changes. If the fast path does not meet its prediction, keep it as
a selectable/auto path only if it is measurably faster, and record the negative
result. All comparisons keep `bt.flush()+bt.wait()` versus
`torch.cuda.synchronize()` and log GPU clocks.
