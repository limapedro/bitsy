# exp_028 — fused cross-entropy loss reduction

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 5 (fusion) + Phase 6 (kernels)

## Prerequisite

After exp_027 the step has 30 dispatches, including 5 `reduce_sum` and 2
`broadcast_grad`. `nn.cross_entropy` is still
`reduce_sum(-onehot * log_softmax(logits), row) -> reduce_sum(total) -> * (1/B)`
— four dispatches for the loss reduction alone, plus its backward broadcasts.

## Hypothesis

A fused cross-entropy loss kernel that reduces `onehot * log_softmax` over both
axes and applies `-1/B` in one dispatch replaces those four dispatches with
one. Its backward is simply `-g * onehot / B`, so the autograd graph stays
correct and the gradient path also shortens.

## Predictions

- Dispatches per step drop by **≥3** (30 → ≤27).
- Captured step improves **≥1.02×** (0.180 → ≤0.176 ms); eager does not regress.
- 43/43 parity incl. gradient gate; loss and gradients within 1e-5 (fp32
  reduction-order drift measured and reported).
- Non-2D or mismatched shapes fall back to the old path.

## Measurement

1. Snapshot current graph/eager/elementwise/GEMM JSON.
2. Add `ce_loss.comp` + `bt_ce_loss()` (CPU fallback), Python binding, and a
   `ce_loss` autograd op with backward `-g * onehot / B`.
3. Make `nn.cross_entropy` use it; keep the old path as fallback.
4. Run `tests/parity/run.py` (43 cases) FIRST; only if green continue.
5. Re-profile dispatch count, re-run graph/eager/elementwise/GEMM benchmarks,
   MNIST/CIFAR, CTest both builds.
6. FINDINGS with all numbers including negatives. No tolerance change.

## Honesty

The 43-case gate is hard; any regression is reverted. Reports state missed
targets plainly and report fp32 drift rather than hiding it.
