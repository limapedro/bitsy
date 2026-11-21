# exp_036 — fused bias+tanh GEMM epilogue

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 6 (fused epilogues, final bounded attempt)

## Prerequisite

A captured step still has ~19 dispatches. `nn.Linear` followed by `nn.Tanh`
issues a GEMM+bias dispatch and a separate tanh dispatch. The campaign lists
fused epilogues; exp_024 fused bias alone. The remaining untried epilogue is
bias+tanh, which is local to the forward pass and easy to verify under the
44-case gate (unlike exp_033's hand-written backward fusion, which broke
training parity).

## Hypothesis

Applying tanh in the GEMM epilogue removes one dispatch per hidden layer while
keeping the same forward values, and the existing autograd backward for tanh
(`g*(1-out²)`) remains valid because the fused output is the same tensor.

## Predictions

- Dispatches per step drop by **≥1** (19 → ≤18).
- Captured step improves slightly (≥1.01×); eager does not regress.
- 44/44 parity incl. the training-scale gate; MNIST/CIFAR loss curves unchanged.
- Only the forward path changes; backward uses existing fused tanh_backward.

## Measurement

1. Snapshot graph/eager/elementwise/gemm JSON.
2. Add `gemm_tiled_bias_tanh.comp` + `bt_matmul_bias_tanh()` (CPU fallback),
   Python binding, and a `Linear.forward_tanh` used by `Sequential` when a
   Linear is immediately followed by Tanh.
3. Run `tests/parity/run.py` (44 cases) FIRST; only if green continue.
4. Re-profile, re-run graph/eager/elementwise/gemm, MNIST/CIFAR, CTest both.
5. FINDINGS with all numbers including negatives. No tolerance change.

## Honesty

The 44-case gate is hard; any regression is reverted. This is a bounded final
fusion attempt — if the gain is negligible the ceiling conclusion stands.
