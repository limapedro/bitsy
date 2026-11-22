# exp_039 — FINDINGS: small GPT-2 (~2M params) parity and performance

**Date:** 2026-11-22. Hypothesis: `HYPOTHESIS.md`. Oracle: PyTorch 2.11.0+cu126,
CUDA 12.6, GTX 1660 Super. Reproduce: `python benchmarks/models/gpt2_tiny.py`.

## Model

GPT-2-style decoder: token embedding + final LayerNorm + linear head, 4 blocks
of (LayerNorm → per-head Q/K/V/O projections → masked causal attention →
residual → LayerNorm → GELU MLP → residual). Config: vocab 1024, d_model 192,
n_layer 4, n_head 4, d_ff 768, block 32, batch 2, seq 8 =
**2,162,688 parameters**. Weights come from one NumPy RNG stream copied into
both frameworks, so logits/gradients are directly comparable. This is a
GPT-2-style model, not a HuggingFace checkpoint loader.

## Numerics (CUDA oracle, copied weights)

| quantity | Bitsy | PyTorch | max abs diff | tolerance | result |
|---|---:|---:|---:|---:|---|
| loss | 6.870110512 | 6.870110512 | 0.0 | 1e-3 | pass |
| logits | — | — | 1.16e-06 | 1e-4 | pass |
| embed weight grad | — | — | 5.07e-07 | 1e-3 | pass |
| head weight grad | — | — | 2.38e-07 | 1e-3 | pass |
| q0 weight grad | — | — | 4.28e-08 | 1e-3 | pass |

All predictions met; the measured diffs are far tighter than the stated 1e-4.
One AdamW step was applied in both frameworks and the run exits 0.

## Performance (median of 10 iters after 2 warmups, correct sync, live clock)

- Bitsy Vulkan forward: **25.42 ms** (spread 14.75–79.68 ms), clock 1950 MHz.
- PyTorch CUDA forward: **14.69 ms** (spread 9.80–24.77 ms).
- Ratio torch/Bitsy = **0.58** — Bitsy is ~1.7× slower.

**Honest miss:** no performance parity is claimed. The gap comes from many small
per-head matmuls and a Python-composed attention path, each with submit/fence
overhead, versus PyTorch's fused scaled-dot-product attention. This is
consistent with the accepted ceiling in `docs/PERFORMANCE_CEILING.md`.

## Bugs found and fixed (see exp_037 FINDINGS for root causes)

1. `log_softmax` silently ignored columns beyond 256 → wrong CE loss; fixed.
2. Register-blocked GEMM misread transposed-view strides → zero embed grad; fixed.
3. Non-differentiable transpose in attention severed K-projection gradients →
   embed grad off 1.3e-1, q0 off 3e-3; fixed with differentiable `Tensor.transpose`.

Each was found by this experiment's forward/gradient comparison, not hidden.

## Negatives / honesty

- Attention uses per-head projection Parameters and a dense block-causal mask;
  a single fused `MultiheadAttention` API is not implemented. The per-head
  decomposition is a parity-scaffolding choice.
- The dense (batch*seq)² attention and one-hot embedding scale poorly; this is
  acceptable for a 2M-param model at seq=8 but would be prohibitive at long
  context. No claim is made about larger models.
- Full 48/48 parity gate and CTest 100% remain green after all changes.
