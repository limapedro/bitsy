# exp_018 — Findings: dedicated reduction kernel

**Date:** 2026-11-21
**Scope:** Phase 6 reductions. No tolerance changed. Parity: 42/42,
CTest 1/1 Vulkan and CPU-only.

## Implementation

- Added `reduce_sum.comp` + `bt_reduce_sum()` C/Python API: a 2D axis reduction
  with one thread per output looping over the reduced axis (single workgroup),
  replacing ones-vector GEMMs. CPU-only path uses the same loop order via host
  copy/create.
- Added `reduce_sum_op()` in autograd with a proper backward (broadcast the
  upstream gradient back to the input shape), and switched `Tensor.sum()`,
  `log_softmax()`, `cross_entropy()`, `mse_loss()` and `_unbroadcast_device()`
  from ones-vector matmuls to the dedicated kernel. >2D still uses the CPU
  fallback as stated in the hypothesis.

## Dispatch profile

| | before (exp_017) | after |
|---|---:|---:|
| dispatches/step | 46 | **43** |
| barriers/step | 52 | 48 |
| tiled GEMM | 20 | **10** |
| reduce_sum | 0 | 4 |
| elementwise (scalar+vec4) | 20 | 23 |

Ten tiled GEMM reductions became four reduction dispatches; the autograd-aware
backward of each reduction adds a few elementwise ops, so the net dispatch
saving is 3 rather than 10.

## Throughput

| path | before | after | change |
|---|---:|---:|---:|
| captured step (1732 MHz) | 0.257 ms | **0.234–0.246 ms** | **1.04–1.10×** |
| graph ratio vs PyTorch CUDA graph | 0.259 | **0.277–0.287** | improved |
| eager step (1530 MHz) | 2.46 ms | **1.898–1.955 ms** | **1.26–1.30×** |
| eager ratio vs PyTorch | 0.41–0.47 | **0.548–0.561** | best yet |

Elementwise 0.43–0.71 and GEMM 1024² 0.68 are unchanged within run variance
(the op benchmarks do not use reductions).

## Predictions

- "captured ≥1.1×, ratio ≥0.29": **marginally missed** — 1.10× only in the
  best run, ratio 0.277–0.287 (just below 0.29). Reported as a narrow miss.
- "eager ≥1.05×": **met** (1.26–1.30×).
- "parity within 1e-5": **met** — 42/42; MNIST report numerically identical to
  the committed one; CIFAR final conv1 W diff 6.407e-07 (4D gradients keep the
  CPU fallback); captured step bit-identical to eager (abs diff 0.0).

## Interpretation / next

The dedicated reduction is the best-value change since deferred uploads: it
removed ten tiled GEMMs and cut the eager step from 2.46 to ~1.93 ms (ratio
0.55). The captured graph is now 0.234–0.246 ms, still ~4× PyTorch's 0.067 ms;
the remaining ~19 scalar elementwise and 10 GEMM dispatches are the next
target. A general fused elementwise-chain kernel (multiple ops in one dispatch)
would attack the dominant remaining class; any such change needs its own
`exp_NNN` and full parity re-verification.

Raw data in `experiments/exp_018/{before_*,after_*,after_step_profile.json}`.
