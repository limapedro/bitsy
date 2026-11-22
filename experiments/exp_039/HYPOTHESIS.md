# exp_039 — small GPT-2 (~2M params) forward/backward/step parity and performance

**Date:** 2026-11-21
**Owner:** feature-completion phase

## Prerequisite

exp_037 supplied GELU, LayerNorm, softmax, embedding and AdamW; exp_038
verified recurrent layers. `docs/FEATURE_INVENTORY.md:113` marks attention and
Transformer as missing. The objective asks for a ~2M-parameter GPT-2 compared
against PyTorch for numerics and performance.

## Hypothesis

A GPT-2-style decoder (embedding + LayerNorm + causal multi-head attention +
GELU MLP + final projection) can be composed from the existing ops using
per-head projection Parameters and a block-causal mask, and will match PyTorch
CUDA logits, loss and gradients within 1e-5, with measured step throughput.

## Predictions

- Forward logits match PyTorch within 1e-4 (accumulated fp32 differences).
- Cross-entropy loss and all parameter gradients match within 1e-4.
- One AdamW step matches PyTorch parameter updates within 1e-4.
- Step throughput is measured and reported honestly, not claimed as parity.

## Measurement

Build `benchmarks/models/gpt2_tiny.py` with a dataclass config (~2M params),
copy weights into a PyTorch GPT-2 equivalent, compare forward/loss/gradients/
one optimizer step, and benchmark Bitsy Vulkan vs PyTorch CUDA wall time with
warmup, median and clock logging.

## Honesty

The model is a GPT-2-style decoder, not a byte-for-byte HF checkpoint loader;
parity is measured against an equivalent PyTorch implementation with identical
weights. Any mismatch is reported in FINDINGS.
