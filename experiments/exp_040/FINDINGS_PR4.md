# exp_040 — FINDINGS (PR #4: lean Python nn/optim composition layer)

Status: PR #4 of the v0.2 rewrite. Builds on PR #1-3 (compact core + autograd).

## What landed
- `python/bitsy_v2/` — a lean Python library that is **pure composition over the
  compact C primitives**: Tensor (lazy ops + autograd hooks), nn (Linear,
  activations, Sequential, LayerNorm, Embedding, RNNCell/LSTM cells, RNN/LSTM),
  functional (mse/cross_entropy), optim (SGD, AdamW), and a self-contained ctypes
  binding. **Zero new C lines** except one bug fix below.
- `benchmarks/compact/model_parity.py` — MLP + LayerNorm + RNNCell forward, loss,
  all gradients, and SGD/AdamW steps vs **real PyTorch CUDA**.
- Fixed a genuine transpose-backward bug in `src/rt/rt.c`: it reshaped the
  gradient instead of transposing it (passed the all-ones gradcheck by
  coincidence, failed on non-uniform Linear weight grads).

## Evidence (Windows MSVC + Vulkan 1.4, GTX 1660 Super)
```
$ python benchmarks/compact/model_parity.py
  [PASS] forward_logits (2.9e-7), loss (0.0), grad_l1W/l1b/l2W/l2b (<1.2e-7)
  [PASS] layernorm (2.4e-7), sgd_step (0.0), adamw_step (6e-8)
  [PASS] rnn_cell_fwd (7.5e-9)
  10/10 bitsy_v2 model-parity cases passed
$ python benchmarks/compact/gradcheck_vs_torch.py  -> 25/25 (unchanged)
$ python benchmarks/compact/parity_vs_torch.py     -> 27/27 (unchanged)
$ python tests/parity/run.py                       -> 48/48 (unchanged)
$ ctest --test-dir build-rtvk                      -> 3/3 100%
$ python tools/count_core_lines.py                 -> compact=1460 (budget 4096)
```

## Key result
The architectural thesis is validated: **all layers are Python compositions over
a fixed C primitive set** — adding Linear, LayerNorm, RNN/LSTM, losses, SGD and
AdamW added **zero new kernels and ~1 net C line** (a bug fix). The core stays at
1,460 lines while the Python surface grows. Numerics match PyTorch CUDA to fp32
tolerance on a real MLP + RNNCell.

## Bugs found and fixed
1. **Transpose backward** reshaped instead of transposed the upstream grad —
   masked by the uniform-grad gradcheck, caught by non-uniform Linear grads.
2. Several Python wiring bugs (handle-vs-Tensor confusion, CE sum/mean, missing
   `__neg__`, RNNCell weight layout) — fixed in the composition layer only.

## Honesty / what is NOT in PR #4
- Not all 285 map functions are implemented — the common ones (MLP/LN/Embedding/
  RNN/LSTM/losses/optim) are; many rare ops remain unimplemented.
- Embedding forward is implemented via one-hot gemm; backward correctness not
  yet covered by a parity case here.
- No speed numbers yet (capture/replay + perf tuning are PR #5/#6).
- Legacy core (5710 lines) retained as reference; budget enforces the 1460-line
  compact core.
