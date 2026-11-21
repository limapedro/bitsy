# exp_022 — fused dual-output multiply for backward chains

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 5 (fusion) + Phase 6 (kernels)

## Prerequisite

exp_021 closed the GEMM L2 rung as a negative. A profiled training step still
has 41 dispatches, of which ~15 are scalar elementwise. Instrumenting backward
showed 6 `mul` calls: 4 come from `__mul__`'s backward, which issues two
independent gradient muls (`g*b` and `g*a`) in two dispatches. Both read the
same upstream gradient and differ only in the second operand.

## Hypothesis

A fused dual-output multiply kernel that writes `g*a` and `g*b` in one dispatch
(second operand may be broadcast/scalar) removes one dispatch per mul node
without changing arithmetic order. Together with a zero-dispatch row view for
the packed output, this should cut two dispatches per training step and improve
captured/eager throughput slightly.

## Predictions

- Dispatches per step drop by **≥2** (41 → ≤39).
- Captured step improves slightly (≥1.02×, 0.220 → ≤0.216 ms) and eager
  improves ≥1.02× (1.84 → ≤1.80 ms); both within a noisy measurement regime,
  so median of several runs will be reported, not a lucky sample.
- The 43-case gate including gradient flow stays green; fused outputs are
  bit-identical to the two unfused muls (same fp32 multiply order).
- Broadcast/scalar second operands stay correct (stride-aware operand reads).

## Measurement

1. Snapshot current graph/eager/elementwise/GEMM JSON.
2. Add `dual_mul.comp` (bindings g, a, b, packed output of 2n floats) with
   stride-aware 2D/broadcast operands, plus `bt_dual_mul()` and an offset-row
   view API so Python gets two tensors sharing the packed output.
3. Use it in `Tensor.__mul__` backward; leave all other paths unchanged.
4. Re-profile dispatch count, re-run graph/eager/elementwise/GEMM benchmarks,
   43/43 parity (including gradient gate), MNIST/CIFAR, CTest both builds.
5. FINDINGS with all numbers including negatives. No tolerance change.

## Honesty

If the dispatch saving does not translate into a measurable throughput gain,
record it as a neutral/negative result. Any parity regression blocks the PR.
