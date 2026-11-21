# exp_019 — Findings: broadcast views (reverted) and fused tanh backward

**Date:** 2026-11-21
**Scope:** Phase 5 fusion + Phase 6 kernels. No tolerance changed. Final state:
42/42 parity, CTest 1/1 Vulkan and CPU-only, MNIST W1 diff 1.788e-07.

## What was attempted and what was kept

Instrumenting one step's op calls showed backward is dominated by 11 `mul`
calls, several of them `ones * gradient` purely to broadcast a reduction
gradient. Two changes were tried:

1. **Zero-dispatch broadcast view** (`bt_tensor_broadcast_view`, stride 0 on
   reduced axes) in `reduce_sum_op`'s backward. Also made `reduce_sum` and
   `sgd_step` shaders stride-aware so they could consume such views safely.
2. **Fused `tanh_backward`** kernel (`g * (1 - t*t)` in one dispatch) with a
   CPU path, wired into `Tensor.tanh().backward()`.

## CRITICAL negative result — broadcast views broke training parity

With broadcast views enabled, the isolated 42-case parity harness **still
passed 42/42**, but end-to-end training was badly wrong:

```
MNIST epoch 10: torch 1.33477 / bitsy 1.87195, accuracy 0.896 / 0.434,
final W1 diff 1.486e-01
CIFAR epoch 3:  torch 1.68931 / bitsy 2.20080, accuracy 0.322 / 0.213
```

Forward loss was identical (2.437650203704834 in both paths); all four
parameter gradients differed (w1 max 0.0207, w2 max 0.0667, b2 max 0.0783 —
see `grad_with_broadcast.json` vs `grad_materialized.json`). Making `reduce_sum`
and `sgd_step` stride-aware **did not** fix it, so the cause is elsewhere in the
backward graph. Since numeric parity is non-negotiable, the broadcast view was
**permanently reverted and the API removed**; its removal is documented in the
header so it is not reintroduced without auditing every stride-blind consumer.
The stride-awareness added to `reduce_sum`/`sgd_step` is harmless and retained.

**Methodology lesson:** the 42-case harness checks isolated ops and did not
catch a gradient-flow error that training loss curves caught immediately.
End-to-end training parity must be treated as a required gate for any autograd
or tensor-view change, not an optional extra.

## Measured results of the kept change (fused tanh backward)

| metric | before (exp_018) | after | target | status |
|---|---:|---:|---:|---|
| dispatches / step | 43 | **41** | ≤36 | **missed** (only −2) |
| captured step | 0.234–0.246 ms | 0.240 ms | 1.10× | missed (neutral) |
| graph ratio vs PyTorch graph | 0.277–0.287 | 0.278 | ≥0.31 | missed |
| eager step | 1.898–1.955 ms | 1.669 ms | 1.05× | met (within variance) |
| eager ratio vs PyTorch | 0.55–0.56 | 0.569 | — | slightly better |

Elementwise 0.40–0.73 and GEMM 1024² 0.69 are unchanged within run variance.
The fused tanh kernel is bit-order-equivalent to the unfused square/subtract/
multiply chain and is kept as a small, correct simplification, but it does not
move the captured-graph needle on its own.

## Interpretation / next

The available wins from broadcast elimination were real (during the broken
variant: captured 0.213 ms, graph ratio 0.312, eager 1.67 ms) but **unusable
without parity**. The next step is not more micro-fusion: it is to make
end-to-end gradient parity a first-class harness check and then find a safe way
to eliminate the `ones * grad` broadcasts — most likely a dedicated kernel
that writes a full-shape contiguous gradient in one dispatch (same count as
today, but no extra input tensor) or a fully stride-audited tensor stack.
Recorded as a negative experiment with the parity break preserved as evidence.

Raw data in `experiments/exp_019/{before_*,after_*,grad_*,after_step_profile.json}`.
