# Benchmarks

For function-by-function status, numeric differences and the same ratios in a
tracker form, see [docs/PROGRESS.md](docs/PROGRESS.md). The full API inventory is
[docs/FEATURE_INVENTORY.md](docs/FEATURE_INVENTORY.md).

## Elementwise results (updated 2026-11-21, exp_003)

`python benchmarks/ops/elementwise.py` on the GTX 1660 Super, torch
2.11.0+cu126 / CUDA 12.6, Vulkan 1.4.363. Median of 60 iterations after 10
warm-ups. Each iteration calls `bt.flush()` (submit + fence wait) on the Bitsy
side and `torch.cuda.synchronize()` on the PyTorch side. Bitsy device time comes
from Vulkan timestamp queries; PyTorch device time is not measured, so only
wall time is compared.

| op | elements | Bitsy ms | PyTorch ms | Bitsy device ms | ratio (target ≥ 0.95) | Bitsy GiB/s | PyTorch GiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| add | 4096 | 0.0783 | 0.0244 | 0.0020 | 0.31 | 0.6 | 1.9 |
| mul | 4096 | 0.1107 | 0.0307 | 0.0021 | 0.28 | 0.4 | 1.5 |
| relu | 4096 | 0.0862 | 0.0266 | 0.0021 | 0.31 | 0.4 | 1.1 |
| add | 1,048,576 | 0.1784 | 0.0868 | 0.0483 | 0.49 | 65.7 | 135.1 |
| mul | 1,048,576 | 0.1963 | 0.0889 | 0.0518 | 0.45 | 59.7 | 131.9 |
| relu | 1,048,576 | 0.1999 | 0.0773 | 0.0492 | 0.39 | 39.1 | 101.1 |

**Targets are missed** (best ratio 0.49 vs ≥ 0.95). Single-op wall time did
not improve with batching; ratio changes are dominated by PyTorch's run-to-run
variance (PyTorch add 1M was 0.0553 ms in the previous run). Timestamp queries
now show the split: for add 1M the GPU kernel is 0.048 ms of the 0.178 ms wall
time. See `experiments/exp_001` (allocator) and `experiments/exp_003`
(batching/timestamps) for before/after snapshots.

## GEMM results (updated 2026-11-21, exp_003)

`python benchmarks/ops/gemm.py`, median of 30 iterations after 5 warm-ups,
`bt.flush()` + `torch.cuda.synchronize()` per iteration. `BT_GEMM_TILED` selects
the kernel.

| n | kernel | Bitsy ms | PyTorch ms | Bitsy device ms | ratio (target ≥ 0.80) | Bitsy GFLOPS |
|---:|---|---:|---:|---:|---:|---:|
| 256 | tiled | 0.185 | 0.045 | 0.070 | 0.24 | 182 |
| 1024 | tiled | 4.824 | 1.128 | 4.443 | 0.23 | 445 |

Naive baseline at n=1024 was 10.224 ms / 0.09 ratio (`experiments/exp_002`).
**GEMM target missed** (0.23 vs ≥ 0.80). Recent naive numbers are in
`experiments/exp_003/before_gemm.json`; the tiled kernel is still the faster one.

## Batched submissions (exp_003)

`python benchmarks/ops/batch_chain.py`: 20 chained adds of 1M elements.

| mode | wall ms | device ms | note |
|---|---:|---:|---|
| eager (`BT_VK_NO_BATCH=1`) | 2.773 | 0.048 | device time is the last submit only |
| batched (default) | 1.277 | 0.942 | device time sums all 20 kernels |

Batching is **2.17x faster** for a multi-op chain. This is the path to
end-to-end speed; single-op benchmark ratios are unchanged because one op still
means one submit. Raw JSON: `experiments/exp_003/chain_eager.json`,
`chain_batched.json`.

## Comparison target

PyTorch CUDA on the **same** GTX 1660 Super, in the same machine, with warm-up,
correct synchronization on both sides, and median/spread reported over many
iterations. Bitsy device time is reported from GPU timestamp queries; PyTorch
device time measurement is still pending, so device time is never compared
against PyTorch wall time.

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
