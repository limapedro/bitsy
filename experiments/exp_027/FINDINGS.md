# exp_027 — fused log_softmax kernel

**Date:** 2026-11-21
**Scope:** Phase 5 fusion + Phase 6 kernels. Parity: 43/43 incl. gradient gate,
CTest 1/1 Vulkan and CPU-only, MNIST W1 diff 3.278e-07, CIFAR conv1 W diff
7.153e-07 (small drift from the fused reduction order, both within tolerance).

## Implementation

- `log_softmax.comp`: one workgroup per row, shared-memory tree reduction over
  the row of `exp(x)`, then `x - log(sum)`; replace exp + GEMM reduction + log
  + subtract (four dispatches) with one. CPU path uses the same fp32 order.
- `bt_log_softmax()` C/Python API; `Tensor.log_softmax()` autograd op with the
  analytic backward `dL/dx = g - exp(y) * sum_cols(g)` (three existing ops).
- `nn.log_softmax` now calls the fused op; non-2D inputs can use the old path.

## Measurements (median of several runs)

| metric | before (exp_026) | after | target | status |
|---|---:|---:|---:|---|
| dispatches / step | 35 | **30** | ≤32 | **met** |
| barriers / step | 35 | 30 | — | improved |
| captured step @1740 MHz | 0.194 ms | **0.179–0.182 ms** | ≤0.19 | **met** (1.07×) |
| graph ratio vs PyTorch CUDA graph | 0.343 | **0.366–0.368** | ≥0.85 | still missed |
| eager step | 0.979 ms | 0.896–0.980 ms | — | no regression |
| eager ratio vs PyTorch | 1.17 | 1.09–1.11 | ≥0.85 | **met** |
| GEMM 1024² | 0.71 | 0.71 | ≥0.80 | still missed |
| elementwise | 0.42–0.71 | 0.40–0.54 | ≥0.95 | still missed |

The ≥1.02× captured-step prediction was met (1.07×). Parity held: 43/43 incl.
the mandatory gradient gate; training loss curves match at reported precision
with slightly larger fp32 drift (MNIST 3.3e-07 vs 1.8e-07) caused by the
different reduction order, well within the 1e-5 tolerance.

## Predictions

- "≥3 fewer dispatches": **met** (5 fewer).
- "captured ≥1.02×, eager no regression": **met**.
- "43-case gate green": **met**.
- "log_softmax within 1e-5": **met** (training drift measured and reported).

## Interpretation / next

Fusion of the forward chain delivered another small, safe gain. Remaining
step: 30 dispatches (10 elementwise, 5 reduce_sum, 4 sgd_step, 6 GEMM-ish, 2
broadcast_grad, plus others). The captured graph ratio is still 0.37 vs 0.85;
the standalone GEMM and elementwise targets remain missed. Next: fuse the
backward reduction chains or apply packed f16. All changes must pass the
43-case gate.

Raw data in `experiments/exp_027/{before_*,after_*,after_step_profile.json}`.
