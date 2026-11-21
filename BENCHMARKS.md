# Benchmarks

For function-by-function status, numeric differences and the same ratios in a
tracker form, see [docs/PROGRESS.md](docs/PROGRESS.md). The full API inventory is
[docs/FEATURE_INVENTORY.md](docs/FEATURE_INVENTORY.md).

## Elementwise results (updated 2026-11-21, exp_011)

`python benchmarks/ops/elementwise.py` on the GTX 1660 Super, torch
2.11.0+cu126 / CUDA 12.6, Vulkan 1.4.363. Median of 60 iterations after 10
warm-ups. Each iteration calls `bt.flush()` + `bt.wait()` (async submit, then
host completion) on the Bitsy side and `torch.cuda.synchronize()` on the PyTorch
side. Bitsy device time comes from Vulkan timestamp queries; PyTorch device time
is not measured, so only wall time is compared.

| op | elements | Bitsy ms | PyTorch ms | Bitsy device ms | clock MHz | ratio (target ≥ 0.95) | Bitsy GiB/s | PyTorch GiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| add | 4096 | 0.0678 | 0.0290 | 0.0018 | 1530 | 0.43 | 0.7 | 1.6 |
| mul | 4096 | 0.0741 | 0.0325 | 0.0020 | 1530 | 0.44 | 0.6 | 1.4 |
| relu | 4096 | 0.0600 | 0.0253 | 0.0018 | 1530 | 0.42 | 0.5 | 1.2 |
| add | 1,048,576 | 0.1059 | 0.0522 | 0.0423 | 1530 | 0.49 | 110.7 | 224.5 |
| mul | 1,048,576 | 0.1091 | 0.0520 | 0.0431 | 1530 | 0.48 | 107.5 | 225.4 |
| relu | 1,048,576 | 0.0883 | 0.0391 | 0.0297 | 1530 | 0.44 | 88.5 | 199.6 |

**Targets are missed** (best ratio 0.49 vs ≥ 0.95). Every result in this table
was measured at 1530 MHz and logs its clock. exp_012 added a vec4 aligned fast
path: a 20-add chain now moves 240 MiB in 0.858 ms device time = **293 GB/s,
96% of the measured 304 GB/s ceiling**, while single-op wall ratios stay
0.43–0.49 because ~36 µs/op of submit+fence overhead dominates. See
`experiments/exp_012/FINDINGS.md`.

## Captured execution (exp_013)

20 chained adds of 1M elements, 1860 MHz (both paths measured in the same run),
median per chain:

| path | ms per chain | ratio vs PyTorch graph |
|---|---:|---:|
| Bitsy re-record each iteration | 1.080 | — |
| Bitsy capture + replay | **0.899** | **1.00×** |
| PyTorch eager | 0.913 | — |
| PyTorch CUDA graph replay | 0.900 | 1.00× |

Capture+replay is 1.20× faster than re-recording, and **matches PyTorch CUDA
graph replay exactly (ratio 1.00)**. This applies only to the captured path;
the ordinary eager API still shows the ratios in the tables above. Details in
`experiments/exp_013/FINDINGS.md`.

## GEMM results (updated 2026-11-21, exp_008)

`python benchmarks/ops/gemm.py`, median of 30 iterations after 5 warm-ups,
`bt.flush()` + `bt.wait()` + `torch.cuda.synchronize()` per iteration.
`BT_GEMM_VARIANT=0..3` selects naive / tiled / register-blocked /
register-blocked+split-K; the default is an autotuned shape threshold.

| n | kernel | Bitsy ms | PyTorch ms | Bitsy device ms | ratio (target ≥ 0.80) | Bitsy GFLOPS |
|---:|---|---:|---:|---:|---:|---:|
| 256 | tiled (autotune) | 0.134 | 0.042 | 0.070 | 0.31 | 250 |
| 1024 | register-blocked vec4 + double buffer (autotune) | 1.265 | 0.903 | 1.028 | **0.71** | 1698 |

Before exp_008 (16×16 tiled): 0.177 ms / 0.27 at 256² and 4.329 ms / 0.22 at
1024². exp_008 register blocking was 3.54× faster at 1024² (496 → 1754
GFLOPS). exp_009 vec4 + double buffering added another 1.04× (1754 → 1823
GFLOPS) but did not meet its ≥1.08× prediction. exp_010 added an on-disk
autotune cache and a 16-deep K-step variant; the k16 variant was **1.59×
slower** than the 8-deep kernel and is never selected. Ratio varies 0.69–0.78
between runs because the GPU clock ranged 1530–1950 MHz; both figures are
reported rather than cherry-picking the faster run. Split-K was also slower
(1.558 ms, ratio 0.57) and is kept as a documented negative result. **GEMM
target still missed.** Raw snapshots: `experiments/exp_008/`, `exp_009/`,
`exp_010/`.

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

## End-to-end training step (updated 2026-11-21, exp_014)

`python benchmarks/models/train_step_bench.py`: MLP 784→64→10, batch 64,
synthetic data, 20 warmup + 200 timed steps, median of per-step wall time.
Bitsy's implicit readback sync and PyTorch's `synchronize()`+`item()` are both
inside the timed region. Deferred transfers are now the default;
`BITSY_DEFERRED_UPLOAD=0` restores the synchronous baseline.

| mode | clock | Bitsy ms/step | PyTorch ms/step | PyTorch/Bitsy throughput | stage allocs / 200 steps |
|---|---:|---:|---:|---:|---:|
| baseline (`=0`) | 1530 MHz | 9.575 | 1.215 | 0.127 | 0 (old path) |
| deferred (default) | 1530 MHz | 3.220 | 1.200 | **0.373** | **0** |

**2.97× faster** at matched clock (a second pair gave 2.83× at a lower deferred
clock), and zero staging allocations during the timed steps. **The ≥0.85
end-to-end target is still missed (0.34–0.37)**: the remaining ~3.1 ms/step is
per-op Python/ctypes dispatch and autograd construction, not transfer sync.
Numeric parity is unchanged — MNIST/CIFAR bitsy losses are bit-identical
between modes (see `experiments/exp_014/FINDINGS.md`).

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
