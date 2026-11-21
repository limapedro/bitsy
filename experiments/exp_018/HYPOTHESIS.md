# exp_018 — dedicated reduction kernel for ones-vector GEMMs

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 6 (reductions)

## Prerequisite

exp_017 cut a training step from 66 to 46 dispatches; the largest remaining
single class is **~20 tiled GEMM dispatches**, many of them tiny ones-vector
reductions used to implement `Tensor.sum()`, `log_softmax`, `cross_entropy`,
`mse_loss` and `_unbroadcast`. Those reductions run as 16×16 tiled GEMMs with
several workgroups when a single workgroup loop would suffice.

## Hypothesis

A dedicated 2D axis-reduction shader (one thread per output, sequential sum
along the reduced axis) can do the same math with one workgroup and less GPU
work than a tiled GEMM. Replacing every ones-vector matmul with this kernel
will reduce captured/eager step time without changing dispatch count.

## Predictions

- Captured step time improves **≥1.1×** (0.257 → ≤0.234 ms/step) and graph
  ratio improves from 0.259 to **≥0.29**.
- Eager step improves **≥1.05×** (2.46 → ≤2.34 ms/step).
- All 42 parity cases stay green and training loss curves remain within the
  1e-5 tolerance. The reduction order differs from GEMM accumulation, so fp32
  drift must be measured, not assumed absent.
- Only 2D reductions are targeted; higher-dimensional cases keep their
  current path (recorded explicitly).

## Measurement

1. Snapshot current graph/eager/elementwise/GEMM JSON and training reports into
   `experiments/exp_018/`.
2. Add `reduce_sum.comp` + `bt_reduce_sum()` C/Python API, and replace the
   ones-vector matmuls in `autograd.py`/`nn.py` with it.
3. Measure reduction-vs-GEMM numeric drift, re-profile dispatch count, re-run
   graph/eager/elementwise/GEMM benchmarks, parity, MNIST/CIFAR, CTest both
   builds.
4. FINDINGS with all numbers including negatives. No tolerance change.

## Honesty

If the reduction kernel does not improve throughput (e.g. dispatch overhead
dominates and GEMM time was already negligible), keep it only if it is neutral
and correct, record the negative and classify the remaining cost. All public
reports must state missed targets plainly.
