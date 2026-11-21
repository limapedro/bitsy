# Benchmarks

## M2 elementwise results (2026-11-21)

Measured with `python benchmarks/ops/elementwise.py` on the GTX 1660 Super,
torch 2.11.0+cu126 / CUDA 12.6, Vulkan 1.4.363. Median of 60 iterations after
10 warm-ups. Both sides allocate an output per op and synchronize every
iteration. **Device time via timestamp queries is not implemented yet**, so
only wall time is reported; this understates neither side but makes small-op
launch overhead visible.

| op | elements | Bitsy ms | PyTorch ms | ratio (target >= 0.95) | Bitsy GiB/s | PyTorch GiB/s |
|---|---:|---:|---:|---:|---:|---:|
| add | 4096 | 0.0772 | 0.0132 | 0.17 | 0.6 | 3.5 |
| mul | 4096 | 0.0703 | 0.0246 | 0.35 | 0.7 | 1.9 |
| relu | 4096 | 0.0942 | 0.0167 | 0.18 | 0.3 | 1.8 |
| add | 1,048,576 | 0.1551 | 0.0553 | 0.36 | 75.6 | 211.9 |
| mul | 1,048,576 | 0.1514 | 0.0537 | 0.35 | 77.4 | 218.2 |
| relu | 1,048,576 | 0.1616 | 0.0421 | 0.26 | 48.3 | 185.6 |

**Targets are missed.** The §7.2 bandwidth-bound target is >= 0.95 and >= 90%
of peak; the best result here is 0.36. The caching allocator experiment
(`experiments/exp_001`) improved ratios from ~0.07 to ~0.17–0.36 by removing
per-op `vkAllocateMemory`, and documented what remains: fence/submission
overhead on small tensors and shader/bandwidth efficiency on large ones. Raw
results: `benchmarks/results/elementwise.json`,
`experiments/exp_001/before.json`, `experiments/exp_001/after.json`.

## Comparison target

PyTorch CUDA on the **same** GTX 1660 Super, in the same machine, with warm-up,
correct synchronization on both sides, and median/spread reported over many
iterations. Both device time (GPU timestamp queries) and wall time (including
launch and dispatch overhead) are reported; device time is pending for M10.

## Targets

Ratio = Bitsy throughput ÷ PyTorch CUDA throughput (higher is better).

| Class | Examples | Target |
|---|---|---|
| Bandwidth-bound elementwise | add, mul, relu, gelu, copy, cast | ≥ 0.95, and ≥ 90% of peak memory bandwidth on large tensors |
| Reductions / softmax / norms | sum, mean, softmax, layer_norm | ≥ 0.90 |
| Indexing / data movement | gather, scatter, index_select, cat, transpose | ≥ 0.90 |
| GEMM (FP32) | matmul, linear, bmm | ≥ 0.80 initially, stretch ≥ 0.95 |
| GEMM (FP16 storage, FP32 accumulate) | same | ≥ 0.80 |
| Convolution | conv2d fwd/bwd | ≥ 0.70 initially, stretch ≥ 0.90 |
| Small-op latency | tiny tensors, launch overhead | ≤ 1.2× PyTorch's per-op overhead |
| End-to-end training | MLP, small CNN, small transformer | ≥ 0.85 of PyTorch's step time throughput |

## Honesty rules

- Never cherry-pick shapes, bench only favorable sizes, inflate numbers through
  unfair setup, or disable the competition's optimizations.
- Report achieved fraction of hardware peak alongside the ratio versus PyTorch.
- Where Bitsy is below target, say so and link the experiment explaining it.
- Without tensor cores on this GPU, PyTorch's CUDA path also uses regular CUDA
  cores; results are not extrapolated to matrix-hardware GPUs.

## Recorded context template

Every result records: GPU clocks, driver version, Vulkan version, PyTorch/CUDA/
cuDNN versions, temperature/throttling state, tensor shapes and dtypes, median
and spread, and the benchmark command/commit.
