# exp_038 — FINDINGS: RNN and LSTM layers with PyTorch parity

**Date:** 2026-11-22. Hypothesis: `HYPOTHESIS.md`. Oracle: PyTorch 2.11.0+cu126,
CUDA 12.6, GTX 1660 Super. Reproduce:
`python benchmarks/models/rnn_lstm_parity.py`.

## Implementation

`RNNCell`, `RNN`, `LSTMCell`, `LSTM` in `python/bitsy/nn.py`, composed entirely
from already-verified differentiable ops (matmul, sigmoid, tanh, mul, add).
No new C kernels were needed. Time steps are selected with differentiable
selection matmuls so input gradients flow back through the sequence. Both the
forward hidden/cell states and the per-step outputs are returned.

Weights are copied from a PyTorch module into the Bitsy module by the harness
(PyTorch packs the four LSTM gates into one matrix; the harness splits it into
four Parameters). This is parity scaffolding, not a public API difference.

## Numerics (B=3, T=4, I=5, H=6, copied weights, CUDA oracle)

| quantity | max abs diff | tolerance | result |
|---|---:|---:|---|
| RNN hidden h | 5.90e-06 | 1e-5 | pass |
| RNN input dx | 2.18e-07 | 1e-5 | pass |
| RNN dW_ih | 1.30e-06 | 1e-5 | pass |
| LSTM hidden h | 2.06e-06 | 1e-5 | pass |
| LSTM cell c | 3.07e-06 | 1e-5 | pass |
| LSTM input dx | 7.73e-08 | 1e-5 | pass |

All predictions met. Full gate stays 48/48; CTest 100%.

## Performance (median of 20 iters after 3 warmups, live clock logged)

| model | Bitsy fw ms | PyTorch fw ms | ratio (torch/Bitsy) | clock |
|---|---:|---:|---:|---:|
| RNN | 1.749 (spread 1.39–4.78) | 0.255 (0.19–0.54) | 0.146 | 1530 MHz |
| LSTM | 5.360 (4.25–7.33) | 0.215 (0.17–0.69) | 0.040 | 1530 MHz |

**Honest miss:** Bitsy is 6.9× slower than PyTorch on RNN and ~25× on LSTM at
this tiny size. The cause is per-step Python-level dispatch of many small ops
(four gate matmuls + elementwise per step), each with submit/fence overhead,
against PyTorch's single fused cuDNN kernel. This is a measured performance
result, not a numeric failure; it is consistent with the accepted ceiling in
`docs/PERFORMANCE_CEILING.md`. No speed target is claimed.

## Negatives / honesty

- The LSTM gate unpacking is a harness convenience; a public API matching
  PyTorch's packed `weight_ih_l0` layout is not implemented.
- The per-step time selection uses a dense (batch × batch*time) one-hot matmul;
  this is correct but wasteful and contributes to the performance miss.
- No gradient is computed for the Python-side one-hot selection matrices
  (they are constant data), which matches PyTorch's treatment of indices.
