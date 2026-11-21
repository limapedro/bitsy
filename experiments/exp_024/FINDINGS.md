# exp_024 — fused GEMM+bias epilogue for Linear

**Date:** 2026-11-21
**Scope:** Phase 6 GEMM ladder (fused epilogues). Parity: 43/43 incl. the
gradient gate, CTest 1/1 Vulkan and CPU-only, MNIST W1 diff 1.788e-07,
CIFAR conv1 W diff 6.594e-07.

## Implementation

- `gemm_bias.comp` (vec4 register-blocked GEMM + bias in epilogue) and
  `gemm_tiled_bias.comp` (tiled GEMM + bias) selected by the same shape rule as
  the autotuned plain GEMM (≤512 → tiled, else vec4). `bt_matmul_bias()` has a
  CPU fallback (matmul + broadcast add) for unsupported layouts.
- `nn.Linear.forward` now calls `C.matmul_bias(x, W, b)` — replacing
  `(x@W) + (ones@b) + add` — with a custom backward that reproduces the same
  gradients: `g@Wᵀ`, `xᵀ@g`, and a device row-sum for `db`.

Two bugs found and fixed before measurement (both caught by the gate or direct
checks): `self._op` in the backward lambda referred to the `Linear` module
instead of a Tensor; and a stray `*out = result` after a released result in the
fallback path produced zeros for unaligned shapes.

## Measurements (median of several runs)

| metric | before (exp_023) | after | target | status |
|---|---:|---:|---:|---|
| dispatches / step | 41 | **35** | ≤37 | **met** |
| eager step @1530 MHz | 1.75 ms | **1.065–1.145 ms** | — | **1.5–1.6× faster** |
| eager throughput ratio vs PyTorch | 0.60 | **0.95–0.97** | ≥0.85 | **MET** |
| captured step @1732 MHz | 0.224 ms | **0.186–0.199 ms** | ≤0.21 | **met** |
| graph ratio vs PyTorch CUDA graph | 0.30 | **0.34–0.37** | ≥0.31 | **met** |
| GEMM 1024² standalone | 0.72 ratio | 0.72 ratio | ≥0.80 | still missed |

Elementwise 0.34–0.54 unchanged within variance. The graph benchmark showed
one 0.303/0.225 outlier run and one 0.412 run at 1935 MHz; the stable cluster
at 1732 MHz is 0.34–0.37, reported rather than cherry-picked.

**This is the first time the ≥0.85 end-to-end training-throughput target has
been met with the ordinary API** (previous best 0.63). The captured graph path
improved but remains ~0.35× PyTorch CUDA graphs.

## Predictions

- "≥4 fewer dispatches": **met** (6 fewer).
- "captured ≥1.03×, eager ≥1.03×": **met** (1.1–1.2× and ~1.6×).
- "43-case gate green": **met**.
- "GEMM standalone reaches 0.80": **not met** (0.72) — the fused epilogue
  targets `nn.Linear`, not the standalone GEMM benchmark.

## Interpretation / next

Removing the redundant `ones @ b` GEMM and add from every Linear was the
highest-value change since deferred uploads. Remaining targets: standalone
GEMM (0.72 vs 0.80), elementwise (≤0.54 vs 0.95), and captured graph
(0.35 vs 0.85). Next: apply the same epilogue idea to backward/optimizer
chains, or advance another GEMM rung. All changes must pass the 43-case gate.

Raw data in `experiments/exp_024/{before_*,after_*,after_step_profile.json}`.
