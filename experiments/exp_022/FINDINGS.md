# exp_022 — fused dual-output multiply rejected; view-offset bug fixed

**Date:** 2026-11-21
**Scope:** Phase 5 fusion + a correctness fix. No tolerance changed. Final
state: 43/43 parity (incl. gradient gate), CTest 1/1 Vulkan and CPU-only,
MNIST W1 diff 1.788e-07, CIFAR conv1 W diff 6.519e-07.

## Attempt: fused dual-output multiply

Added `dual_mul.comp` + `bt_dual_mul()` (bindings g, a, b, packed output of
2n floats) and `bt_tensor_offset_view()` to split the packed output into two
gradient tensors, wired into `Tensor.__mul__` backward. Isolated checks showed
the two outputs bit-identical to the unfused `C.mul(g,a)` / `C.mul(g,b)`.

**But the mandatory gradient gate FAILED** (loss identical, all four parameter
gradients wrong) — exactly the class of bug the gate exists to catch. A
substitution test replacing `C.dual_mul` with two unfused `C.mul` calls passed
all checks, confirming the packed-view path was at fault. Since parity is a
hard gate, the fused dual path was **reverted**; the unused kernel/API is kept
as a negative-result variant.

## Real bug found and fixed while debugging

The investigation exposed a genuine correctness defect in the tensor view
layer:

- `bt_tensor_copy_to_host()` passed only `storage->gpu` to readback and
  **ignored `tensor->offset`**, so any non-zero-offset view read from element 0.
- The fused kernels `sgd_step`, `reduce_sum`, `broadcast_grad` (and `dual_mul`)
  likewise ignored tensor offsets in their shaders.

Fixed by adding `bt_vk_readback_offset()` (byte offset in the buffer copy),
using `tensor->offset` in `bt_tensor_copy_to_host`, and adding offset push
constants to all four shaders. This is the most likely root cause of exp_019's
broadcast-view gradient corruption: the stride-0 views themselves may have
been fine, but reading them back / consuming them ignored their offsets. The
broadcast-view API remains removed; reintroducing it would be a new experiment.

## Predictions

- "≥2 fewer dispatches and measurable speedup": **not met** (path reverted).
- "43-case gate catches the regression": **met** (it failed before the revert,
  passed after).
- "parity preserved": **met** in the final tree.

## Final measurements (after revert + offset fix)

Captured step 0.2198 ms (ratio 0.303), eager step 1.563 ms (ratio 0.627),
elementwise 0.34–0.76, GEMM 1024² 0.70 — all within run variance of exp_020 /
exp_021. No performance regression from the correctness fix (normal tensors
have offset 0). The ≥0.85 targets remain missed.

## Interpretation / next

Fusion attempt rejected, but the campaign gained a correctness fix and stronger
evidence that the gradient gate is load-bearing. A future experiment can retry
zero-dispatch views now that offsets are honoured, or move on to other fusion
patterns; all changes must pass the 43-case gate.

Raw data in `experiments/exp_022/{before_*,after_*,grad_flow_PASS.json}`.
