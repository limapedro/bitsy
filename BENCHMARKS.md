# Benchmarks

For function-by-function status, numeric differences and the same ratios in a
tracker form, see [docs/PROGRESS.md](docs/PROGRESS.md). The full API inventory is
[docs/FEATURE_INVENTORY.md](docs/FEATURE_INVENTORY.md).

## Elementwise results (updated 2026-11-21, exp_004)

`python benchmarks/ops/elementwise.py` on the GTX 1660 Super, torch
2.11.0+cu126 / CUDA 12.6, Vulkan 1.4.363. Median of 60 iterations after 10
warm-ups. Each iteration calls `bt.flush()` + `bt.wait()` (async submit, then
host completion) on the Bitsy side and `torch.cuda.synchronize()` on the PyTorch
side. Bitsy device time comes from Vulkan timestamp queries; PyTorch device time
is not measured, so only wall time is compared.

| op | elements | Bitsy ms | PyTorch ms | Bitsy device ms | ratio (target ≥ 0.95) | Bitsy GiB/s | PyTorch GiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| add | 4096 | 0.0689 | 0.0365 | 0.0019 | 0.53 | 0.7 | 1.3 |
| mul | 4096 | 0.1137 | 0.0334 | 0.0020 | 0.29 | 0.4 | 1.4 |
| relu | 4096 | 0.1142 | 0.0400 | 0.0020 | 0.35 | 0.3 | 0.8 |
| add | 1,048,576 | 0.1522 | 0.0556 | 0.0479 | 0.36 | 77.0 | 211.0 |
| mul | 1,048,576 | 0.1746 | 0.0621 | 0.0521 | 0.36 | 67.1 | 188.6 |
| relu | 1,048,576 | 0.1840 | 0.0429 | 0.0492 | 0.23 | 42.5 | 182.1 |

**Targets are missed** (best ratio 0.53 vs ≥ 0.95). Ratio movement between runs
is dominated by PyTorch's small-op timing variance (PyTorch add 1M: 0.0868 ms in
exp_003, 0.0556 ms here). See `experiments/exp_001` (allocator), `exp_003`
(batching/timestamps), `exp_004` (async/fusion) for before/after snapshots.

## GEMM results (updated 2026-11-21, exp_008)

`python benchmarks/ops/gemm.py`, median of 30 iterations after 5 warm-ups,
`bt.flush()` + `bt.wait()` + `torch.cuda.synchronize()` per iteration.
`BT_GEMM_VARIANT=0..3` selects naive / tiled / register-blocked /
register-blocked+split-K; the default is an autotuned shape threshold.

| n | kernel | Bitsy ms | PyTorch ms | Bitsy device ms | ratio (target ≥ 0.80) | Bitsy GFLOPS |
|---:|---|---:|---:|---:|---:|---:|
| 256 | tiled (autotune) | 0.134 | 0.042 | 0.070 | 0.31 | 250 |
| 1024 | register-blocked vec4 + double buffer (autotune) | 1.178 | 0.913 | 1.024 | **0.78** | 1823 |

Before exp_008 (16×16 tiled): 0.177 ms / 0.27 at 256² and 4.329 ms / 0.22 at
1024². exp_008 register blocking was 3.54× faster at 1024² (496 → 1754
GFLOPS). exp_009 vec4 + double buffering added another 1.04× (1754 → 1823
GFLOPS) but did not meet its ≥1.08× prediction. Split-K was slower (1.558 ms,
ratio 0.57) and is kept as a documented negative result. **GEMM target still
missed** (0.78 vs ≥ 0.80). Raw snapshots: `experiments/exp_008/` and
`experiments/exp_009/`.

## Batched and async submissions

`python benchmarks/ops/batch_chain.py`: 20 chained adds of 1M elements.

| mode | wall ms | device ms | note |
|---|---:|---:|---|
| eager (`BT_VK_NO_BATCH=1`, exp_003) | 2.773 | 0.048 | device time is the last submit only |
| batched + fence wait (exp_003) | 1.277 | 0.942 | 2.17x faster than eager |
| batched + timeline semaphore (exp_004) | 1.251 | 0.935 | ~2% vs previous batch; async prediction not met |

## Pointwise fusion (exp_004)

`python benchmarks/ops/fusion.py`: 20 `add + relu` pairs of 1M elements.

| path | wall ms | device ms |
|---|---:|---:|
| separate `add` then `relu` | 2.365 | 1.915 |
| fused `add_relu` | 1.200 | 0.935 |

**1.97x faster** with fusion; first fused pointwise kernel, not yet a general
fusion pass. Raw JSON: `experiments/exp_004/fusion.json`.

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
