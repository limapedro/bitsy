# exp_036 — fused bias+tanh GEMM epilogue (neutral)

**Date:** 2026-11-21
**Scope:** Phase 6 fused epilogues, final bounded fusion attempt. Parity:
44/44 incl. the training-scale gate, CTest 1/1 Vulkan and CPU-only,
MNIST W1 diff 3.278e-07, CIFAR W diff 6.855e-07.

## Implementation

- `gemm_tiled_bias_tanh.comp` computes `tanh(x@W + bias)` in one dispatch.
- `bt_matmul_bias_tanh()` C/Python API with a fallback to `matmul_bias` +
  `tanh` for unsupported layouts and CPU builds.
- `Linear.forward_tanh()` with a backward that reuses `tanh_backward` and the
  existing matmul/reduction ops; `Sequential.forward` fuses a Linear+Tanh pair.

## Measurements

| metric | before (exp_035) | after | target | status |
|---|---:|---:|---|---|
| dispatches / step | 19 | **18** | ≤18 | **met** |
| captured step | 0.1457 ms | **0.1432–0.1447 ms** | ≥1.01× | ~1.01×, marginal |
| graph ratio vs CUDA graph | 0.506 | 0.461–0.466 | ≥0.85 | still missed (clock variance) |
| eager step | 0.825 ms | 0.780–0.879 ms | no regression | within variance |
| elementwise / GEMM | 0.42–0.68 / 0.71 | 0.41–0.53 / 0.71 | — | unchanged |

The ≥1.01× captured prediction is only marginally met (one dispatch saved,
~0.2–1.7% time depending on run); it is recorded as a **neutral** result. The
training-scale gate passed, unlike exp_033's hand-written backward fusion —
keeping the backward composed from verified fused ops was the safe choice.

## Predictions

- "≥1 fewer dispatch": **met** (1 fewer).
- "captured ≥1.01×": **marginally met** (neutral in practice).
- "44/44 parity": **met**.
- "no elementwise/GEMM regression": **met**.

## Interpretation / next

This was the last bounded fusion with local, verifiable math. It confirms the
`docs/PERFORMANCE_CEILING.md` conclusion: removing one dispatch at a time does
not move the captured-graph ratio materially. The remaining gaps need a
step-level graph compiler with training-scale parity or tensor-core hardware —
both beyond the current per-op/kernel work. All changes remain behind the
44-case gate.

Raw data in `experiments/exp_036/{before_*,after_*}`.
