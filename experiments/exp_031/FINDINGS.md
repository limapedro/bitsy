# exp_031 — elementwise three-way timing (launch-overhead classification)

**Date:** 2026-11-21
**Scope:** Phase 3 three-way timing + Phase 5 submission. No kernel, autograd
or tolerance change. Parity: 43/43 incl. gradient gate, CTest 1/1 both builds.

## Implementation

Added `benchmarks/ops/elementwise_graph.py` (dataclass config, no argparse):
for each op (add, mul, relu) and size (4096, 1M), measure Bitsy eager
(`flush()+wait()` per op), Bitsy capture+replay, PyTorch eager, and PyTorch
CUDA graph; report median per-op wall time, both ratios, and GPU clock. This
follows the campaign's own three-way methodology (isolated/captured execution)
that `elementwise.py`'s per-op sync does not.

## Measured ratios (median of two runs)

| n | op | Bitsy eager | Bitsy captured | PyTorch graph | eager ratio | captured ratio |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | add | 0.058–0.062 ms | 0.0100–0.0107 ms | 0.0084–0.0085 ms | 0.17–0.22 | 0.79–0.85 |
| 4096 | relu | 0.052–0.056 ms | 0.0109–0.0116 ms | 0.0082–0.0087 ms | 0.20–0.27 | 0.71–0.80 |
| 1M | add | 0.097–0.100 ms | 0.0478–0.0498 ms | 0.0479–0.0491 ms | 0.44–0.46 | **0.96–1.03** |
| 1M | mul | 0.099–0.100 ms | 0.0500–0.0501 ms | 0.0441–0.0491 ms | 0.44–0.47 | **0.88–0.98** |
| 1M | relu | 0.083–0.098 ms | 0.0345–0.0374 ms | 0.0306–0.0329 ms | 0.32–0.41 | **0.88–0.89** |

(One 4096/mul run measured a 0.0229 ms PyTorch graph outlier giving a 2.39×
ratio; the second run was 0.0080 ms / 0.80×. Reported rather than cherry-picked.)

## Predictions

- "Captured large tensors ≥0.85 of PyTorch CUDA graph": **met** for all three
  1M ops (0.88–1.03×).
- "Eager ratios stay ~0.4–0.76": **met** (0.17–0.47; small ops even lower).
- "No kernel work needed if captured reaches parity": **supported for large
  tensors**; small tensors are still 0.71–0.80×, so a residual launch gap
  remains at 4096 elements.
- "43-case gate green": **met**.

## Interpretation

The elementwise "gap" in the public table is **host launch/sync overhead**,
not kernel bandwidth: with captured execution the same kernels are at parity
with PyTorch CUDA graphs for 1M tensors. This matches exp_013's 1.00× pointwise
capture and the 96%-of-DRAM-ceiling measured in exp_012. Further elementwise
kernel tuning would not move the eager wall ratio; reducing per-op submit+fence
in the default API would. The small-tensor captured residual (~0.75×) is the
remaining launch cost inside graph replay (≈10 µs/op).

Next: the remaining open classes are the default eager submit cost, the
standalone GEMM (~0.72), and the captured full-step graph (~0.43). All future
changes must pass the 43-case gate.

Raw data in `experiments/exp_031/{before_elementwise.json,after_elementwise_graph.json}`.
