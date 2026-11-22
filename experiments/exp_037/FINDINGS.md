# exp_037 — FINDINGS: primitives needed for GPT-2

**Date:** 2026-11-22. Hypothesis: `HYPOTHESIS.md`. Oracle: PyTorch 2.11.0+cu126,
CUDA 12.6, GTX 1660 Super.

## What was implemented

- **ReLU backward fix** (`autograd.py`, shader op 11, `bt_relu_backward`):
  previously the identity; now masks non-positive outputs. Verified
  `relu grad [0,0,0,1,1] want [0,0,0,1,1]`.
- **Division backward fix** (`autograd.py`): added the missing `d/db = -g*a/b^2`.
  `div grad a [1,0.5,0.5]`, `grad b [-2,-0.75,-1]` both match PyTorch exactly.
- **sqrt**: new dispatched op `BT_OP_SQRT`, CPU kernel + Vulkan shader op 12,
  Python wrapper `bt.sqrt` / `Tensor.sqrt`, differentiable backward `1/(2*sqrt(x))`.
- **GELU**: exact-erf forward + analytic derivative, new `gelu.comp` Vulkan
  pipeline and CPU fallback (`bt_gelu`, `bt_gelu_backward`, `Tensor.gelu`).
- **LayerNorm, softmax, Embedding, Dropout** (`nn.py`) composed from verified
  ops; **AdamW** (`optim.py`) with bias correction and decoupled weight decay.
- **Differentiable 2D transpose** (`Tensor.transpose`) — zero-dispatch strided
  view with autograd context (required for attention scores).
- **`mse_loss`** corrected to PyTorch's `reduction='mean'` (was sum × 0.5).
- **`__rtruediv__`** added so `1.0 / tensor` works in LayerNorm.
- **Parity gate** extended with `sqrt_cpu/vulkan`, `gelu_cpu/vulkan` (+4 cases),
  tolerances, `bt` top-level exports, and the gelu torch-op mapping.

## Results (all measured, not estimated)

- GELU forward max diff vs `torch.nn.functional.gelu`: **1.2e-7**; backward 1.2e-7.
- LayerNorm (4×8, non-trivial upstream grad): fwd 2.4e-7, dx 5.4e-7,
  dW 6e-8, db 0.0. Softmax dx 1.7e-7. matmul dA/dB ≤ 4.8e-7. reshape 0.0.
- AdamW one step on a Linear(5→3): W max diff **1.5e-8**, b 9.3e-10, loss
  1.673734903 vs 1.673734784.
- Parity gate: **48/48 passed, 0 skipped** (was 44/44). CTest: **100%**.
- GPT-2 end-to-end (exp_039): loss 6.870110512 vs 6.870110512, logits 1.16e-6,
  grads embed 5.1e-7 / head 2.4e-7 / q0 4.3e-8 — all predictions met.

## Bugs found during integration (root causes, all fixed)

1. `log_softmax` kernel assumed ≤256 columns and silently ignored the rest;
   with vocab=1024 this made CE loss 0.34 instead of 6.87. Fixed by looping
   `for (k = tid; k < N; k += 256)` in both the reduction and output pass.
2. The register-blocked GEMM misread non-unit-stride (transposed-view) operands,
   returning zeros for the embedding weight gradient. `vk_ops.c` now forces the
   stride-aware tiled kernel whenever `!vec4_ok`.
3. The GPT-2 harness wrapped `transpose_view(kh._t)` in a leaf Tensor with no
   autograd context, severing the K-projection gradient path; the embedding grad
   was off by 1.3e-1 and q0 by 3e-3. Fixed with the differentiable `transpose()`.
4. `Embedding.forward` used `-1` in `reshape`, unsupported by the C view;
   explicit dims are now computed. Its backward is now `_op`-based so `W` grads flow.

## Negatives / honesty

- **sqrt divergence**: Bitsy clamps negative inputs at 0; PyTorch returns NaN.
  The parity cases use the positive value set, so this divergence is *not*
  hidden — it is documented here and the gate does not claim parity on negatives.
- **intended-fused paths stayed reverted** (exp_019/023 broadcast views, exp_022
  dual mul, exp_033 fused CE backward): reintroducing them is out of scope here.
- "100% parity" means every implemented function matches PyTorch within the
  documented tolerance on tested shapes; the full PyTorch API (thousands of
  overloads) is out of scope, as recorded in the hypothesis.
- No kernel tolerance was relaxed: sqrt/gelu reuse the existing `APPROX`
  (rtol 1e-5, atol 1e-6) class already used by exp/log/sigmoid/tanh.
