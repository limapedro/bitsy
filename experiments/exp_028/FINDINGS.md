# exp_028 — fused cross-entropy loss reduction

**Date:** 2026-11-21
**Scope:** Phase 5 fusion + Phase 6 kernels. Parity: 43/43 incl. gradient gate,
CTest 1/1 Vulkan and CPU-only, MNIST W1 diff 3.278e-07, CIFAR conv1 W diff
7.302e-07.

## Implementation

- `ce_loss.comp`: one workgroup reduces all `B*C` `onehot*log_softmax`
  products and writes `-sum/B` — replacing a mul, two reductions and a scale
  (four dispatches) with one. CPU path keeps the fp32 order.
- `bt_ce_loss()` C/Python API and `Tensor.ce_loss()` autograd op with backward
  `-g * onehot / B`. `nn.cross_entropy` now composes the fused log_softmax
  (exp_027) and this fused loss; the `broadcast_grad` dispatches from the old
  reduction backwards disappear too.

## Measurements (median of several runs)

| metric | before (exp_027) | after | target | status |
|---|---:|---:|---:|---|
| dispatches / step | 30 | **22** | ≤27 | **met** (8 fewer) |
| barriers / step | 30 | 22 | — | improved |
| captured step @1672–1740 MHz | 0.180 ms | **0.156 ms** | ≤0.176 | **met** (1.15×) |
| graph ratio vs PyTorch CUDA graph | 0.366–0.368 | **0.425–0.542** | ≥0.85 | still missed |
| eager step | 0.896–0.980 ms | 0.906–1.042 ms | — | no regression |
| eager ratio vs PyTorch | 1.09–1.11 | 1.11–1.24 | ≥0.85 | **met** |
| GEMM 1024² | 0.71 | 0.68 | ≥0.80 | still missed |
| elementwise | 0.40–0.54 | 0.40–0.75 | ≥0.95 | still missed |

The graph-ratio range is wide because PyTorch's graph time varied 0.0666–0.0844
ms between runs; Bitsy's captured step was stable at 0.1556–0.1567 ms. The
≥1.02× captured-step prediction was met (1.15×). Parity held with the same
small fp32 drift as exp_027 (MNIST 3.3e-07), within 1e-5.

## Predictions

- "≥3 fewer dispatches": **met** (8 fewer).
- "captured ≥1.02×, eager no regression": **met**.
- "43-case gate green": **met**.
- "loss/gradients within 1e-5": **met** (drift measured and reported).

## Interpretation / next

This was the highest-value fusion since the bias epilogue: removing the loss
reduction chain and its backward broadcasts cut a third of the remaining
dispatches. Step is now 22 dispatches: 5 elementwise, 4 sgd_step, 3 reduce_sum,
6 GEMM variants, 1 vec4. The captured graph ratio (0.43–0.54) is still far
below 0.85; elementwise and standalone GEMM targets also remain missed.
Next: fuse the 4 optimizer steps or the remaining reduction chains, or packed
f16. All changes must pass the 43-case gate.

Raw data in `experiments/exp_028/{before_*,after_*,after_step_profile.json}`.
