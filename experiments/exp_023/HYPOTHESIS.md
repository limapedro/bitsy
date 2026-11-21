# exp_023 — zero-dispatch broadcast views retried after offset fix

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 5 (submission) + Phase 6

## Prerequisite

exp_019 tried zero-dispatch stride-0 broadcast views for reduction gradients.
They gave captured 0.213 ms / graph ratio 0.312 / eager 1.67 ms but broke
end-to-end training (MNIST loss 1.87 vs 1.33, W1 diff 0.149) while isolated
parity stayed green; the API was removed. exp_022 found the likely root cause:
`bt_tensor_copy_to_host` and four fused shaders ignored `tensor->offset`, so
non-zero-offset views were read from element 0. That is now fixed and verified
by the 43-case gate.

Currently reduction gradients are written by a dedicated `broadcast_grad`
kernel (one dispatch each, ~3–4 per step).

## Hypothesis

With offsets honored end-to-end, a zero-dispatch stride-0 broadcast view is
now safe. Replacing the `broadcast_grad` dispatches with views removes ~3–4
dispatches per training step and recovers the best measured throughput without
breaking the gradient gate.

## Predictions

- The mandatory 43-case gate (including gradient flow) PASSES.
- Dispatches/step drop by **≥3** (41 → ≤38).
- Captured step ≤0.210 ms (from 0.220) and graph ratio **≥0.31** (from 0.303).
- Eager step ≤1.50 ms (from 1.56), ratio ≥0.65.
- All other parity/benchmark results unchanged within variance.
- If the gate fails, revert immediately and record the offset fix as
  insufficient (negative).

## Measurement

1. Snapshot current graph/eager/elementwise/GEMM JSON.
2. Re-add `bt_tensor_broadcast_view()` (C + Python) and use it in
   `reduce_sum_op` backward instead of `C.broadcast_grad`.
3. Run `tests/parity/run.py` (43 cases) FIRST; only if green continue.
4. Re-profile dispatch count, re-run graph/eager/elementwise/GEMM benchmarks,
   MNIST/CIFAR parity, CTest both builds.
5. FINDINGS with all numbers including negatives. No tolerance change.

## Honesty

The 43-case gate is a hard gate: any gradient-flow regression blocks the PR and
the change is reverted. The earlier broken numbers are not claimed as progress
until this gate passes. All public reports state missed targets plainly.
