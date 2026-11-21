# exp_020 — gradient-flow parity gate and safe broadcast elimination

**Date:** 2026-11-21
**Scope:** Phase 2 gates + Phase 5/6. No tolerance relaxed. Final state: 43/43
parity cases (42 isolated + 1 gradient flow), CTest 1/1 Vulkan and CPU-only,
MNIST W1 diff 1.788e-07, CIFAR conv1 W diff 6.258e-07.

## 1. Required end-to-end gradient-flow gate

Added `tests/parity/grad_flow.py` (dataclass config, no argparse): same MLP,
same synthetic batch and init in Bitsy and PyTorch; compares loss, all four
parameter gradients and all four post-step parameters (rtol 1e-4, atol 1e-5).
It is called from `tests/parity/run.py` (now **43/43** cases) and from both CI
jobs, with its JSON uploaded as an artifact. Broken paths produce a clean FAIL
report rather than a crash.

**Proven sensitivity:** injecting a wrong-value gradient (multiplying the
broadcast gradient by 0) made the gate FAIL on all eight gradient/parameter
checks while the loss still matched — `grad_flow_FAULT_detected.json`. Restoring
the correct path made it PASS — `grad_flow_PASS.json`. This is exactly the
class of error that exp_019's 42 isolated cases missed.

## 2. Safe broadcast elimination

The zero-dispatch stride-0 view remains unsafe (exp_019), so instead of a view
a dedicated `broadcast_grad.comp` / `bt_broadcast_grad()` kernel writes a
**contiguous full-shape** gradient from the small reduction gradient. It
replaces `ones * grad` with one dispatch, no ones-tensor upload, no stride-0
views. It has a CPU path. The gradient-flow gate passes with it enabled.

## Measurements

| metric | before (exp_019 final) | after | target | status |
|---|---:|---:|---:|---|
| dispatches / step | 41 | 41 | ≤40 | **missed** (1:1 replacement) |
| captured step (1732 MHz) | 0.240 ms | **0.219–0.220 ms** | ≤0.225 | **met** (~1.10×) |
| graph ratio vs PyTorch graph | 0.278 | **0.304–0.305** | ≥0.31 | narrowly missed |
| eager step (1530 MHz) | 1.669 ms | **1.565–1.582 ms** | ≤1.60 | **met** (~1.06×) |
| eager ratio vs PyTorch | 0.569 | **0.62–0.63** | — | improved |

Elementwise 0.39–0.53 and GEMM 1024² 0.73 are unchanged within run variance.
The dispatch target was missed because the dedicated kernel replaces one mul
dispatch with one broadcast dispatch; the gain comes from removing the ones
tensor (smaller reads, no upload), not from fewer dispatches.

## Predictions

- "gate fails on known-bad broadcast, passes on current tree": **met**.
- "captured ≤0.225 ms, eager ≤1.60 ms": **met**.
- "≥3 dispatches removed": **not met** (0 removed) — recorded.
- "graph ratio ≥0.31": **narrowly missed** (0.304–0.305).
- "parity within tolerance": **met** (43/43 + training curves).

## Interpretation / next

The campaign now has a hard gate that would have prevented exp_019's bad
change. Safe broadcast elimination is achieved with a small, real speedup
(captured 0.219 ms, eager 1.57 ms, ratio 0.62). Remaining cost is still ~41
dispatches per step, dominated by 15 scalar elementwise and 10 tiled GEMM ops.
Next: fuse the remaining adjacent elementwise ops with a multi-stage kernel
or attack the tiled-GEMM class; every change must now pass the gradient gate.

Raw data in `experiments\exp_020\{before_*,after_*,grad_flow_*,after_step_profile.json}`.
