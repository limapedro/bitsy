# exp_027 — fused log_softmax kernel

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 5 (fusion) + Phase 6 (kernels)

## Prerequisite

A training step still has 35 dispatches. `nn.log_softmax` is implemented as
`logits - log(logits.exp() @ ones_col)` — four dispatches (exp, GEMM reduction,
log, subtract). This is the largest remaining pure-elementwise/reduction chain
in the forward path, and the campaign lists fused softmax/norm as a target.

## Hypothesis

A fused log_softmax kernel (one workgroup per row: exp, reduce, log, subtract
in registers/shared memory) replaces four dispatches with one while preserving
the fp32 operation order closely enough for the 1e-5 tolerance. Its backward is
`g - exp(y) * sum_cols(g)`, expressible with three existing device ops, so the
autograd graph stays correct.

## Predictions

- Dispatches per step drop by **≥3** (35 → ≤32).
- Captured step improves **≥1.02×** (0.194 → ≤0.190 ms) and eager step does
  not regress; median of several runs reported.
- 43/43 parity incl. the gradient gate; log_softmax output within 1e-5 of the
  unfused path (measured drift reported, not assumed absent).
- Fallback to the unfused path for non-2D or non-contiguous inputs.

## Measurement

1. Snapshot current graph/eager/elementwise/GEMM JSON.
2. Add `log_softmax.comp` + `bt_log_softmax()` (CPU fallback), a Python
   binding, and a `log_softmax()` autograd op with the analytic backward.
3. Make `nn.log_softmax` use it; keep the old path as fallback.
4. Run `tests/parity/run.py` (43 cases) FIRST; only if green continue.
5. Re-profile dispatch count, re-run graph/eager/elementwise/GEMM benchmarks,
   MNIST/CIFAR parity, CTest both builds.
6. FINDINGS with all numbers including negatives. No tolerance change.

## Honesty

The 43-case gate is hard; any regression is reverted. Reports must state
missed targets plainly and not cherry-pick samples.
