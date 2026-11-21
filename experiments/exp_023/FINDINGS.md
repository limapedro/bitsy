# exp_023 — zero-dispatch broadcast views retried (negative result)

**Date:** 2026-11-21
**Scope:** Phase 5/6. No tolerance changed. Final state is a revert of this
experiment's only functional change; 43/43 parity, CTest 1/1 both builds.

## What was tried

Re-added `bt_tensor_broadcast_view()` (C + Python) and used it in
`reduce_sum_op`'s backward instead of the dedicated `broadcast_grad` kernel,
now that exp_022 had fixed view-offset handling in `bt_tensor_copy_to_host`
and the fused shaders.

## Result: the mandatory gate still FAILS

```
[FAIL] end_to_end_gradient_flow
42/43 parity cases passed. 0 skipped.
```

The isolated 42 op cases passed, but the gradient-flow gate failed with the
same signature as exp_019: identical loss, wrong parameter gradients. The
exp_022 offset fix was therefore **not** the root cause (or not sufficient).
Per the hard-gate rule the change was immediately reverted: the API was removed
again, `reduce_sum_op` uses `C.broadcast_grad` again, and the tree is back to
the exp_022 state. The header carries a warning that this has now failed twice.

This is the second measured rejection of zero-dispatch stride-0 views. It is
kept as a negative result, not silently dropped.

## Final measurements (reverted tree)

| metric | value |
|---|---:|
| captured step (1732 MHz) | 0.2235 ms, graph ratio 0.300 |
| eager step (1530 MHz) | 1.751 ms, ratio 0.604 |
| elementwise | 0.42–0.69 |
| GEMM 1024² | 1.309 ms, ratio 0.72 |
| parity | 43/43 incl. gradient gate |
| CTest | 1/1 Vulkan and CPU-only |

These are within run variance of exp_022; the earlier 0.213 ms / 0.312 figures
from the exp_019 broken path are **not** claimed as a result.

## Interpretation / next

Stride-0 views remain unsafe despite correct offsets; some other consumer of a
non-contiguous view still misreads it. The safe `broadcast_grad` kernel stays.
Further work on this path requires finding that consumer (or abandoning views
entirely). Faster progress is more likely from the remaining GEMM ladder rungs
(fused epilogues, packed f16, tile aspect ratio) or from a fully stride-audited
autograd rewrite. Every change must pass the 43-case gate.

Raw data in `experiments/exp_023/{before_*,after_*,grad_flow_PASS.json}`.
