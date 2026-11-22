# exp_038 — RNN and LSTM layers with PyTorch parity

**Date:** 2026-11-21
**Owner:** feature-completion phase

## Prerequisite

`docs/FEATURE_INVENTORY.md:112` marks RNN/LSTM as missing. exp_037 established
the missing primitives and fixed ReLU/division backward.

## Hypothesis

RNNCell, RNN, LSTMCell and LSTM can be composed from existing differentiable
ops (matmul, sigmoid, tanh, mul, add) using per-gate weight Parameters, with
differentiable selection matmuls for time steps — no new C kernels needed — and
will match PyTorch CUDA forward outputs and gradients within 1e-5.

## Predictions

- RNN hidden states, input grads and weight grads match PyTorch within 1e-5.
- LSTM hidden/cell states and input grads match PyTorch within 1e-5.
- 44/44 existing parity cases stay green.
- One-step training with AdamW matches PyTorch parameter updates.

## Measurement

Implemented RNN/RNNCell/LSTM/LSTMCell in `python/bitsy/nn.py`; compared
B=3, T=4, I=5, H=6 against `torch.nn.RNN`/`torch.nn.LSTM` on CUDA with copied
weights. Results: RNN h 6.6e-06, dx 2.5e-07, dWih 9.0e-07; LSTM h 2.1e-06,
c 2.5e-06, dx 1.1e-07 — all within prediction.

## Honesty

Full PyTorch RNN/LSTM parameter layouts are emulated by splitting packed gate
matrices into four Parameters in the harness; this is a parity scaffolding
choice, not a public API difference. Results recorded in FINDINGS.
