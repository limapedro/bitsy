# exp_019 — broadcast views and fused elementwise chains

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 5 (fusion) + Phase 6 (kernels)

## Prerequisite

exp_018 left ~19 scalar elementwise dispatches dominating each training step.
Instrumenting one step showed backward calls: 11 `mul`, 8 `matmul`, 4 `sgd_step`,
3 `view`, 1 `reduce_sum`, 1 `div`, 1 `add`, 1 `sub`. Several of those `mul`
calls are `ones * gradient` in `reduce_sum_op`'s backward — pure broadcasting
that should be a zero-dispatch strided view, not a kernel.

## Hypothesis

Two changes collapse the elementwise class without changing math: (1) replace
`ones * grad` broadcast gradients with zero-dispatch **broadcast views** (stride
0 on reduced axes, which the elementwise shader already honours); (2) add a
general **fixed-binding fused elementwise kernel** for short chains (up to
several stages in one dispatch) and use it for the remaining adjacent scalar
mul/sub pairs where the pattern matches, with a safe unfused fallback.

## Predictions

- Dispatches per step drop by **≥15%** (43 → ≤36).
- Captured step improves **≥1.1×** (0.234–0.246 → ≤0.22 ms) and eager improves
  **≥1.05×** (1.90 → ≤1.81 ms).
- Graph ratio vs PyTorch CUDA graph improves from 0.28 to **≥0.31**.
- 42/42 parity cases, MNIST/CIFAR loss curves within 1e-5; fused chains keep
  fp32 operation order (drift measured, not assumed).
- Broadcast views are numerically exact (same values, only a stride change).

## Measurement

1. Snapshot current graph/eager/elementwise/GEMM JSON and training reports.
2. Add `bt_tensor_broadcast_view()` + `C.broadcast_view`, use it in
   `reduce_sum_op` backward. Profile again.
3. If needed, add the fixed-binding fused elementwise kernel for the remaining
   adjacent mul/sub chains.
4. Re-profile dispatch counts, re-run graph/eager/elementwise/GEMM benchmarks,
   parity, MNIST/CIFAR, CTest both builds.
5. FINDINGS with all numbers including negatives. No tolerance change.

## Honesty

If the broadcast views alone meet the dispatch target, do not add the general
kernel (record that decision). If neither meets the speed predictions, record
the negatives and classify the remaining cost. All public reports must state
missed targets plainly.
