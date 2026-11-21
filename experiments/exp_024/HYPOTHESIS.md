# exp_024 — fused GEMM+bias epilogue for Linear

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 6 (GEMM ladder: fused epilogues)

## Prerequisite

A training step still has 41 dispatches. `nn.Linear.forward` is
`x @ W + ones @ b` — three dispatches per Linear (two GEMMs plus an add), and
there are two Linears per step. The campaign's GEMM ladder lists fused
epilogues as an untried rung.

## Hypothesis

Fusing the bias add into the GEMM epilogue (`out[i,j] = (A@B)[i,j] + bias[j]`)
removes the separate `ones @ b` matmul and add — two dispatches per Linear,
four per step — without changing the arithmetic order of the matmul itself.
The gradient gate must remain green (backward keeps its current structure;
only forward is fused).

## Predictions

- Dispatches per step drop by **≥4** (41 → ≤37).
- Captured step improves **≥1.03×** (0.220 → ≤0.214 ms) and eager improves
  ≥1.03× (1.75 → ≤1.70 ms); report median of several runs.
- 43/43 parity including the gradient gate stays green; Linear forward output
  matches the unfused path within the existing GEMM tolerance.
- If alignment/bias-shape cases cannot be handled safely, fall back to the
  existing path and record the negative.

## Measurement

1. Snapshot current graph/eager/elementwise/GEMM JSON.
2. Add `gemm_rb_vec4_bias.comp` (vec4 GEMM with bias in epilogue), a
   `bt_matmul_bias()` C/Python API with a CPU fallback, and use it in
   `nn.Linear.forward`.
3. Run `tests/parity/run.py` (43 cases) FIRST; only if green continue.
4. Re-profile dispatch count, re-run graph/eager/elementwise/GEMM benchmarks,
   MNIST/CIFAR parity, CTest both builds.
5. FINDINGS with all numbers including negatives. No tolerance change.

## Honesty

The 43-case gate is a hard gate; any regression is reverted. Reports must state
missed targets plainly and not claim the numerically-unsafe path's numbers.
