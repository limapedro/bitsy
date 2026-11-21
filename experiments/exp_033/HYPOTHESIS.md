# exp_033 — fused softmax-cross-entropy backward

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 5 (fusion) + Phase 6 (kernels)

## Prerequisite

A captured step is ~0.145 ms with ~19 dispatches; PyTorch CUDA graphs take
~0.067 ms with far fewer kernels. The output layer still computes its backward
through a chain: cross-entropy backward (`-g*onehot/B`), log_softmax backward
(`g - exp(y)*sum_cols(g)`), with several elementwise and reduction dispatches.
Algebraically the combined gradient is simply
`dlogits = g * (softmax(logits) - onehot) / B`, which needs one row-parallel
kernel.

## Hypothesis

Replacing the output-layer backward chain with one fused kernel removes
~3–5 dispatches per step without changing the math (softmax uses the same
unnormalized-exp convention as the current fused log_softmax), and improves
captured-graph throughput.

## Predictions

- Dispatches per step drop by **≥3** (19 → ≤16).
- Captured step improves **≥1.05×** (0.145 → ≤0.138 ms) and graph ratio
  improves from ~0.46–0.49 to ≥0.50.
- 43/43 parity incl. gradient gate; loss and gradients within 1e-5 (fp32
  reduction-order drift measured and reported).
- No change to the elementwise/GEMM paths; eager step does not regress.

## Measurement

1. Snapshot current graph/eager/elementwise/gemm JSON.
2. Add `ce_bwd_fused.comp` + `bt_ce_bwd_fused()` (CPU fallback) and a
   `fused_cross_entropy` autograd op that uses the existing fused forward
   (log_softmax + ce_loss) with the new single-dispatch backward.
3. Switch `nn.cross_entropy` to it; keep the old path available as fallback.
4. Run `tests/parity/run.py` (43 cases) FIRST; only if green continue.
5. Re-profile dispatch count, re-run graph/eager/elementwise/gemm benchmarks,
   MNIST/CIFAR, CTest both builds.
6. FINDINGS with all numbers including negatives. No tolerance change.

## Honesty

The 43-case gate is hard; any regression is reverted. Reports state missed
targets plainly and report fp32 drift rather than hiding it.
