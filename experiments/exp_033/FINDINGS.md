# exp_033 — fused softmax-cross-entropy backward (reverted, negative)

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 5/6. **No code retained**: the tree is functionally
identical to exp_032 after a full revert and re-verification.

## Attempt

A fused output-layer backward kernel computing `dlogits = g*(softmax-onehot)/B`
in one dispatch, plus a `fused_cross_entropy` autograd op using the existing
fused forward (log_softmax + ce_loss). `nn.cross_entropy` was switched to it.

## Measured before the parity failure

- Dispatches/step **19 → 14** (5 fewer), barriers 19 → 15.
- Captured step 0.145 → **0.137–0.143 ms** (ratio 0.486–0.488 at 1740 MHz with
  normal PyTorch graph timing; one outlier run showed 0.723 because PyTorch's
  graph time was 0.0995 ms instead of ~0.068 ms).
- Eager step 0.806–1.040 ms (ratio 1.02–1.28) — noisy but not worse.
- **The mandatory 43-case gate PASSED**, including its gradient-flow check.

## CRITICAL: end-to-end training parity FAILED

Running the committed MNIST/CIFAR training parity scripts showed:

```
MNIST final W1 max diff: 7.015e-01   (was 3.278e-07)
CIFAR final conv1 W diff: 4.916e-01 (was 7.302e-07)
```

The gradient-flow gate uses a small config (batch 8, 32→16→8) and did not
catch this. This is the **second** incident (after exp_019's broadcast views)
where isolated + small-step parity passed but full training diverged. Per the
non-negotiable rule, the change was reverted completely (shader, pipeline,
dispatch, C/Python APIs, autograd op, nn change).

After revert and rebuild: 43/43, CTest 1/1 Vulkan and CPU-only, MNIST W1 diff
3.278e-07, CIFAR 7.302e-07 — the exp_032 baseline is restored.

## Predictions

- "≥3 fewer dispatches": **met** (5 fewer).
- "captured ≥1.05× / ratio ≥0.50": **not reliably met** (0.137–0.143 ms;
  ratio 0.486–0.488 with normal PyTorch timings).
- "43-case gate green": **met**, but insufficient — training parity failed.
- "no regression": **not met** — reverted.

## Interpretation / next

Two lessons: (1) the small-config gradient gate is not sufficient to protect
training parity; a CI gate using the actual training config (batch 64,
784→64→10) is needed before any further autograd changes; (2) the remaining
19-dispatch gap cannot be closed by ad-hoc op fusion without end-to-end
verification at training scale. The dispatch-reduction path has now produced
one neutral result (exp_029) and one parity failure (this exp). Further work
should either add the stronger gate first, or conclude the practical ceiling.

Raw numbers above; `before_graph_step.json` and `before_train_step.json`
snapshot the starting state.
