# Feature parity matrix — Bitsy vs PyTorch (fp32, this GPU)

Two kinds of parity are tracked, because "good" means both:
- **Numeric**: Bitsy output vs real PyTorch CUDA, fp32, within documented
  tolerance (EXACT for simple elementwise; rtol/atol 1e-5 for matmul and
  transcendentals; 1e-3 for GPT-2 loss). Source of truth: parity harness.
- **Speed**: ratio `pytorch_ms / bitsy_ms`. 1.00 = as fast as PyTorch. Target on
  this GPU (GTX 1660 Super) is ≥ 1.00 for every row. Sourced from benchmarks
  with warmup, correct sync, median + spread, GPU clock logged.

Status: **done** (numeric + speed evidence), **numeric-only** (numbers match,
speed not at target), **planned** (no implementation on this branch), **n/a**.

> This branch is a skeleton. The numbers below are the **inherited baseline**
> measured on the previous architecture and recorded in committed reports
> (`BENCHMARKS.md`, `tests/parity/reports/latest.json`,
> `benchmarks/results/*.json` in the pre-skeleton history). They establish the
> starting point the v0.2 skeleton must meet or beat; they are not claims about
> code on this branch.

## Core tensor ops

| API | Numeric | max abs diff (baseline) | Speed ratio (baseline) | Status |
|---|---|---|---|---|
| add / sub / mul / div | EXACT | 0 | 0.41–0.53 | numeric-only |
| relu / sigmoid / tanh / exp / log | APPROX 1e-5 | ≤ 1.2e-7 | 0.41–0.53 | numeric-only |
| sqrt | APPROX | ≤ 1e-6 | not measured | numeric-only |
| gelu (erf) | APPROX | 1.2e-7 | not measured | numeric-only |
| matmul (fp32) | rtol/atol 1e-5 | ≤ 4.8e-7 (grad), 1.16e-6 (logits) | 0.71 | numeric-only |
| softmax / log_softmax | APPROX | ≤ 3e-8 | fused, not isolated | numeric-only |
| reduce_sum | APPROX | ≤ 1e-6 | not isolated | numeric-only |
| transpose (2D view) | EXACT | 0 | 0 (free view) | done |
| cross_entropy | APPROX | loss 0.0 (GPT-2) | fused | numeric-only |

## Autograd & optimizers

| API | Numeric evidence | Speed | Status |
|---|---|---|---|
| backward (reverse graph) | gradcheck ≤ 1e-6; 48/48 gate | per-step tracked below | numeric-only |
| SGD | MNIST loss/acc identical | captured step ~0.46 | numeric-only |
| AdamW | 1-step W 1.5e-8, b 9.3e-10 | not isolated | numeric-only |

## Layers & models

| Component | Numeric (max diff) | Speed ratio | Status |
|---|---|---|---|
| Linear (+bias) | W grad ≤ 6e-8 | part of step | numeric-only |
| LayerNorm | ≤ 5.4e-7 | not isolated | numeric-only |
| Embedding | forward 0 | dense one-hot matmul | numeric-only |
| ReLU/Tanh/Sigmoid/GELU modules | as ops above | as ops above | numeric-only |
| RNN (B3 T4 I5 H6) | h 5.9e-6, dx 2.2e-7 | 0.15 | numeric-only |
| LSTM (B3 T4 I5 H6) | h 2.1e-6, c 3.1e-6 | 0.04 | numeric-only |
| GPT-2 small (2,162,688 params) | loss 0.0, logits 1.16e-6, grads ≤ 5.1e-7 | 0.58 | numeric-only |
| MNIST MLP | W1 1.5e-7, loss/acc identical | step ~0.46 | numeric-only |
| CIFAR CNN | logits 1.2e-5 | not isolated | numeric-only |

## Planned surface for v0.2 (not yet implemented)

| Area | Items | Status |
|---|---|---|
| nn layers | Conv1d/3d, pooling, BatchNorm, GRU, MultiheadAttention, Transformer | planned |
| dtypes | fp16/bf16 storage (fp32 remains main) | planned |
| data/serialization | Dataset/DataLoader, safetensors, DLPack | planned |
| distributed | DataParallel / collective ops | out of scope until core is fast |

## Interpretation

- **Numeric parity is effectively achieved** for the implemented subset: every
  implemented function matches PyTorch within its tolerance (48/48 op cases +
  all model numerics). This is the "100% parity on tested shapes" claim, with
  the full PyTorch API (thousands of overloads) explicitly out of scope.
- **Speed parity is NOT achieved**: ratios 0.04–0.71 mean Bitsy is slower than
  PyTorch everywhere measured. This is the gap the v0.2 skeleton exists to
  close. The progress image tracks each ratio rising toward 1.0.
- Slow paths share one cause: Python-level per-op dispatch + Vulkan
  submit/fence overhead versus PyTorch's fused cuDNN/CUBLAS kernels. The
  skeleton's answer is lazy fusion + captured command-buffer replay.
