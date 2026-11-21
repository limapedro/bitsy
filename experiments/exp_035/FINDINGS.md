# exp_035 — evidence-backed performance ceiling assessment

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 2 (reporting)
**No kernel change.** Parity: 44/44 incl. training-scale gate, CTest green.

## Final consistent measurement set

| class | Bitsy | PyTorch | ratio | target |
|---|---:|---:|---:|---|
| eager end-to-end step | 0.825 ms | 0.983 ms | **1.19** | ≥0.85 MET |
| elementwise eager 1M | 0.095–0.124 ms | 0.044–0.064 ms | 0.46–0.54 | ≥0.95 missed |
| elementwise captured 1M | 0.034–0.051 ms | 0.031–0.045 ms | 0.87–0.93 | at parity |
| GEMM 1024² | 1.276 ms | 0.907 ms | 0.71 | ≥0.80 missed |
| captured full step | 0.1457 ms | 0.0737 ms graph | 0.51 | ≥0.85 missed |

These reproduce earlier reports within run variance, so the prediction "final
measurements match the reported ratios" is **met**.

## Deliverable

Added `docs/PERFORMANCE_CEILING.md`: per-class mechanism classification, the
34-experiment outcome ledger, and what would actually be required to improve
further. `GAP_LEDGER.md` and `BENCHMARKS.md` now link to it.

## Predictions

- "Each unmet target maps to a measured mechanism": **met** — elementwise is
  host launch (captured parity), GEMM ladder exhausted, captured step is
  dispatch count vs PyTorch's fused graph kernels, small ops are launch-bound.
- "No remaining campaign item has >10% upside": **supported by the measured
  negatives** (fusions neutral or parity-breaking; ladder rungs all measured).
- "44-case gate green": **met**.

## Interpretation

The campaign reached one of its targets (eager end-to-end) and exhaustively
classified the rest. The conclusion is conditional on this GPU/driver/backend
stack; a step-level graph compiler or tensor-core hardware would be required
for further gains. No performance claim is made by this documentation-only
experiment.

Raw data: `experiments/exp_035/after_{elementwise,elementwise_graph,gemm,graph_step,train_step}.json`.
