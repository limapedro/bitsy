# exp_037 — primitives needed for GPT-2 (GELU, LayerNorm, softmax, embedding, AdamW)

**Date:** 2026-11-21
**Owner:** feature-completion phase (new objective)

## Prerequisite

Two correctness bugs block 100% parity: `Tensor.relu()` backward is the
identity (no zeroing for non-positive outputs) and `__truediv__` backward omits
the denominator gradient. Both must be fixed before sequence models can match
PyTorch. `docs/FEATURE_INVENTORY.md` marks GELU, LayerNorm, softmax, Embedding,
and AdamW/Adam as missing. A ~2M-parameter GPT-2 cannot be built or compared with
PyTorch without them. Existing primitives: linear (with fused bias epilogue),
tanh/sigmoid/relu/exp/log, matmul, reductions, cross_entropy, SGD.

## Hypothesis

Adding these five primitives as differentiable ops following the existing
`Tensor._op` pattern, with Vulkan kernels where a native op is needed and a
correct CPU fallback, will let a small GPT-2 run forward/backward/step on
Bitsy with numeric parity versus PyTorch CUDA.

## Predictions

- GELU matches PyTorch's default (erf) within rtol/atol 1e-4 on fp32.
- LayerNorm matches PyTorch nn.LayerNorm (eps=1e-5) within 1e-4 for forward,
  gradients, and weight/bias grads.
- Softmax matches PyTorch softmax within 1e-5 and gradients within 1e-4.
- Embedding lookup matches PyTorch nn.Embedding exactly in forward and its
  sparse weight gradient matches (dense comparison) within 1e-5.
- AdamW state update matches PyTorch torch.optim.AdamW (betas, eps, weight
  decay) within 1e-5 for parameters and exp_avg/exp_avg_sq.
- All 44 existing parity cases stay green plus new op-level cases.

## Measurement

1. Inventory exact patterns from the explorer report.
2. Implement each primitive + new parity cases in `tests/parity/`.
3. Run per-op parity against PyTorch CUDA on this machine; record numbers.
4. FINDINGS with all results including negatives; no tolerance relaxed without
   an ARCHITECTURE_LOG entry.

## Honesty

"100% parity" means every implemented function matches PyTorch within the
documented tolerance on the tested shapes; the full PyTorch API is thousands of
overloads and is out of scope. Any mismatch is reported, not hidden.
