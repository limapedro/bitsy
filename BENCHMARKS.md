# Benchmarks

**Campaign closed — ceiling accepted (2026-11-21).** The user accepted the
evidence-backed conclusion in `docs/PERFORMANCE_CEILING.md` (exp_035) after 36
experiments. Targets still missed are stated plainly below; no further
performance work is planned on this GPU/backend stack.

For function-by-function status, numeric differences and the same ratios in a
tracker form, see [docs/PROGRESS.md](docs/PROGRESS.md). The full API inventory is
[docs/FEATURE_INVENTORY.md](docs/FEATURE_INVENTORY.md).

## Model parity and performance (2026-11-22, exp_037/038/039)

Small models compared against PyTorch CUDA with identical copied weights;
`python benchmarks/models/rnn_lstm_parity.py` and
`python benchmarks/models/gpt2_tiny.py`. Median of 20/10 iterations after
warmup, correct device sync, live GPU clock logged. Full distributions:
`benchmarks/results/rnn_lstm_parity.json`, `benchmarks/results/gpt2_tiny.json`.

| model | params | forward bitsy ms | forward torch ms | ratio torch/bitsy | clock MHz |
|---|---:|---:|---:|---:|---:|
| RNN (B3 T4 I5 H6) | — | 1.75 (1.39–4.78) | 0.26 (0.19–0.54) | 0.15 | 1530 |
| LSTM (B3 T4 I5 H6) | — | 5.36 (4.25–7.33) | 0.22 (0.17–0.69) | 0.04 | 1530 |
| GPT-2 small | 2,162,688 | 25.42 (14.75–79.68) | 14.69 (9.80–24.77) | 0.58 | 1950 |

**All three are honest performance misses** — Bitsy is slower. Numeric parity
passes for all three (RNN h 5.9e-6, LSTM h 2.1e-6/c 3.1e-6, GPT-2 loss 0.0/
logits 1.16e-6, grads ≤5.1e-7). Cause: Python-level dispatch of small ops and
submit/fence overhead versus fused cuDNN kernels. Consistent with the accepted
ceiling; no speed target is claimed. Details: `experiments/exp_038/FINDINGS.md`,
`experiments/exp_039/FINDINGS.md`.

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
| deferred (default, exp_014) | 1530 MHz | 3.220 | 1.200 | 0.373 | **0** |
| deferred + device unbroadcast (exp_016) | 1530 MHz | 2.626–2.911 | 1.078–1.080 | 0.37–0.41 | **0** |
| + fused transpose views / SGD (exp_017) | 1530 MHz | 2.460–2.469 | 1.010–1.143 | 0.41–0.47 | **0** |
| + dedicated reduction kernel (exp_018) | 1530 MHz | **1.898–1.955** | 1.065–1.071 | **0.55–0.56** | **0** |

Deferred transfer was **2.97× faster** than the synchronous baseline at matched
clock; exp_016's device-side unbroadcast reduction removed a host sync from
every backward and improved the default step a further **1.1–1.2×**; exp_017's
zero-dispatch transpose views and fused SGD step brought it to 2.46 ms
(ratio **0.41–0.47**, the best yet), with zero staging allocations throughout.
**The ≥0.85 end-to-end target is still missed**: the remaining ~1.9 ms/step is
per-op Python/ctypes dispatch and autograd construction. exp_018 replaced
ones-vector GEMM reductions with a dedicated kernel, cutting the eager step
from 2.46 to ~1.93 ms (1.26–1.30×, ratio 0.55–0.56 — best yet). Numeric parity
is unchanged — MNIST report identical, CIFAR W diff 6.4e-07 (see
`experiments/exp_014`, `exp_016`–`exp_018/FINDINGS.md`).

## Captured full training step (updated 2026-11-21, exp_015–018)

`python benchmarks/models/graph_step_bench.py`: the whole MLP step (forward +
backward + SGD) captured and replayed 100× per trial, 10 trials, median of
per-step wall time. Inputs are static buffers (CUDA-graph semantics). Captured
output is bit-identical to eager (abs diff 0.0).

| path | ms/step | clock |
|---|---:|---:|
| Bitsy eager (default API, exp_014) | 3.220 | 1530 MHz |
| Bitsy captured + replayed (exp_015) | 0.294–0.355 | 1732–1935 MHz |
| Bitsy captured + replayed (exp_016, single chunk) | 0.296–0.334 | 1658–1732 MHz |
| Bitsy captured + replayed (exp_017, fused) | 0.257 | 1732 MHz |
| Bitsy captured + replayed (exp_018, reductions) | **0.234–0.246** | 1732 MHz |
| PyTorch eager | 1.20 | 1530 MHz |
| PyTorch CUDA graph | **0.067–0.068** | 1732–1935 MHz |

Capture is **~8–10× faster than eager Bitsy**, but **graph throughput is only
0.28× PyTorch** — the ≥0.85 end-to-end target and the ≥0.80 intermediate gate
are both **missed**. exp_016 removed the mid-backward CPU gradient readback;
exp_017 cut dispatches 66 → 46 (transpose views + fused SGD) taking the step
to 0.257 ms; exp_018 replaced ones-vector GEMM reductions with a dedicated
kernel, taking it to **0.234–0.246 ms** and the graph ratio to 0.277–0.287.
The ≥0.29 stretch ratio was just missed. Remaining cost: ~19 scalar elementwise
and 10 GEMM dispatches. See `experiments/exp_015`–`exp_018/FINDINGS.md`.

## End-to-end gradient-flow gate (exp_020)

`tests/parity/grad_flow.py` is now a **required** parity case (43/43 total)
and runs in CI: one MLP step in Bitsy vs PyTorch CUDA/CPU, comparing loss, all
four parameter gradients and all post-step parameters. It was proven to fail
on a deliberately corrupted gradient even when loss still matched, closing the
gap that let exp_019's bad broadcast view through.

With that gate in place, `ones * grad` was replaced by a dedicated kernel that
writes a contiguous full-shape gradient (no unsafe stride-0 view). Captured
step 0.240 → **0.219 ms**, graph ratio 0.278 → **0.304**, eager step 1.669 →
**1.57 ms**, eager ratio 0.62. Dispatch count is unchanged at 41 (1:1 kernel
replacement); the graph ≥0.31 stretch ratio was narrowly missed. See
`experiments/exp_020/FINDINGS.md`.

## End-to-end training step (updated 2026-11-21, exp_024)

`nn.Linear.forward` now uses a fused GEMM+bias epilogue (tiled for small shapes,
register-blocked vec4 for large) instead of `(x@W) + (ones@b) + add`, with a
matching custom backward. Dispatches per step 41 → **35**.

| path | ms/step | ratio vs PyTorch | target |
|---|---:|---:|---|
| Bitsy eager (1530 MHz) | **1.065–1.145** | **0.95–0.97** | ≥0.85 **MET** |
| PyTorch eager | 1.01–1.11 | — | — |
| Bitsy captured graph (1732 MHz) | 0.186–0.199 | 0.34–0.37 | still missed |
| PyTorch CUDA graph | 0.067–0.068 | — | — |

The eager end-to-end training-throughput target is **met for the first time**
(previous best 0.63): removing the redundant `ones @ b` GEMM and add from each
Linear was worth ~1.6×. The captured graph path improved from 0.30 to 0.34–0.37
but remains well below 0.85. Standalone GEMM (0.72) and elementwise (≤0.54)
targets are still missed. See `experiments/exp_024/FINDINGS.md`.

## exp_036 — fused bias+tanh epilogue (neutral)

The final bounded fusion applied tanh in the Linear GEMM epilogue: dispatches
19 → **18**, captured step 0.1457 → 0.1432–0.1447 ms (~1%, neutral), eager
0.780–0.879 ms (ratio 1.14–1.29), graph ratio 0.46, GEMM/elementwise unchanged.
44/44 parity incl. the training-scale gate; MNIST/CIFAR unchanged. This
confirms the ceiling conclusion in `docs/PERFORMANCE_CEILING.md`. See
`experiments/exp_036/FINDINGS.md`.

## exp_034 — training-scale gradient gate (CI)

`tests/parity/train_scale.py` is now a required case (batch 64, 784→64→10), so
the suite is **44/44** and CI checks training-scale loss, gradients and
post-step parameters. Proven to fail on an injected wrong gradient. It
immediately exposed a real bug: `ce_loss.comp` summed only the first 256 of 640
elements (loss 0.9907 vs 2.4377); fixed by looping over all elements. MNIST/CIFAR
loss curves and W1 diffs unchanged after the fix. No performance change claimed.
See `experiments/exp_034/FINDINGS.md`.

## exp_033 — fused softmax-cross-entropy backward (reverted)

A fused output-layer backward kernel cut dispatches 19 → 14 and captured step
0.145 → 0.137–0.143 ms, and the 43-case gate passed — **but end-to-end training
parity broke** (MNIST W1 diff 0.70, CIFAR 0.49 vs ~3e-07). The change was fully
reverted and the exp_032 baseline restored (43/43, MNIST 3.3e-07). This is the
second case where the small-config gradient gate missed a training-scale
gradient error; see `experiments/exp_033/FINDINGS.md`.

## exp_032 — lazy device-time timestamps (neutral)

Device-time timestamp collection is now on demand, removing an unconditional
`vkGetQueryPoolResults(WAIT_BIT)` + reset from every wait for callers that never
read device time. Captured step 0.156 → 0.145–0.166 ms (ratio 0.46–0.49),
eager step 0.894–0.899 ms (ratio 1.23–1.24), elementwise/GEMM unchanged within
variance. The ≥1.1× request was not met: `elementwise.py`/`gemm.py` request
device time each iteration, so they cannot benefit. First `device_time_ms()`
call after enabling returns 0 by design (documented). Parity 43/43. See
`experiments/exp_032/FINDINGS.md`.

## exp_031 — elementwise three-way timing (launch overhead)

`benchmarks/ops/elementwise_graph.py` follows the campaign's own captured
three-way methodology. At 1M elements Bitsy captured/replayed ops reach
**0.88–1.03×** PyTorch CUDA graphs (add 0.96–1.03, mul 0.88–0.98, relu
0.88–0.89), while eager per-op wall ratios remain 0.17–0.47. At 4096 elements
captured ratios are 0.71–0.85. The elementwise gap in the eager table is
therefore **host submit/fence overhead, not kernel bandwidth**; no kernel
change was made. See `experiments/exp_031/FINDINGS.md`.

## exp_030 — packed f16 GEMM (rejected)

The last GEMM ladder rung was tried and removed: a tiled `f16vec2` shared-memory
GEMM was **~3× slower** at 1024² (3.508 ms vs ~1.0–1.3 ms) and produced wrong
results on a K=16 check even after fixing a tile-index bug. No code was kept;
the tree is unchanged from exp_029. The ≥1.05× / ≥0.80 predictions were not
met. See `experiments/exp_030/FINDINGS.md`.

## exp_029 — fused SGD update (neutral)

All four parameter updates now issue in one dispatch via a dedicated 12-binding
layout. Dispatches per step 22 → **19**, but captured time is unchanged
(0.153–0.159 ms vs 0.156) — the single kernel launches `max_workgroups × 4`
workgroups with an if-chain, offsetting the saved submit. The ≥1.02× prediction
was **not met**. Eager ratio remains 1.11–1.17 (target met); graph ratio 0.43–0.45
and standalone GEMM/elementwise targets remain missed. Parity 43/43 incl.
gradient gate; MNIST W1 diff 3.3e-07. See `experiments/exp_029/FINDINGS.md`.

## exp_028 — fused cross-entropy loss reduction

`nn.cross_entropy` now composes the fused log_softmax with a single fused
loss-reduction kernel (`onehot*log_softmax` reduced over both axes), with
backward `-g*onehot/B`. Dispatches per step 30 → **22** (8 fewer, including
the old reduction backwards). Captured step 0.180 → **0.156 ms** (1.15×), graph
ratio 0.366 → **0.425–0.542** (wide range from PyTorch graph timing variance;
Bitsy stable at 0.156 ms), still below 0.85. Eager 0.906–1.042 ms (ratio
1.11–1.24, target met). GEMM (0.68) and elementwise (≤0.75) targets remain
missed. Parity 43/43 incl. gradient gate; MNIST W1 diff 3.3e-07 (within 1e-5).
See `experiments/exp_028/FINDINGS.md`.

## exp_027 — fused log_softmax

`nn.log_softmax` is now one fused dispatch (one workgroup per row: exp, shared
reduction, log, subtract) with an analytic backward `g - exp(y)*sum_cols(g)`.
Dispatches per step 35 → **30**. Captured step 0.194 → **0.179–0.182 ms**
(1.07×), graph ratio 0.343 → **0.366–0.368**; eager 0.896–0.980 ms (ratio
1.09–1.11, target met). Standalone GEMM (0.71) and elementwise (≤0.54) targets
remain missed. Parity 43/43 incl. gradient gate; MNIST W1 diff 3.3e-07 (small
fp32 drift from the fused reduction order, within 1e-5). See
`experiments/exp_027/FINDINGS.md`.

## exp_026 — median GEMM tuning and 64x128 tile

`gemm_tune.py` now takes the **median of 3 samples** per variant; two
consecutive runs selected the same winners, removing the exp_025 noise flip.
A new 64×128×8 variant (8) won 1024² in both runs with medians 0.982 / 1.001 ms
versus 1.069 / 1.016 ms for the previous winner (1.02–1.09×), and is now the
autotune choice at both shapes. The ≥1.05× prediction was met in one run and
not the other (reported as a range); the GEMM ratio is still **0.72–0.76**,
below the 0.80 target. Standalone 1024² 1.270 ms, captured 0.194 ms (ratio
0.343), eager 0.979 ms (ratio 1.17, target met), parity 43/43. See
`experiments/exp_026/FINDINGS.md`.

## exp_025 — 128x64 tile aspect ratio (negative result)

A 128×64×8 register-blocked variant (variant 7) was measured against the
current winners: 256² 0.148 vs 0.129 ms and 1024² 1.091 vs **1.005 ms** —
1.09–1.15× slower. It is never auto-selected and is kept as a negative-result
variant. The 1024² GEMM ratio remains 0.68–0.76 (target ≥0.80). The tuner's
single-sample noise also flipped its 1024² winner to variant 6 by 0.3%; since
exp_021 measured that variant as 21% slower, the cache was restored to the
established winner rather than accepting a noise-driven regression. See
`experiments/exp_025/FINDINGS.md`.

## exp_023 — zero-dispatch broadcast views rejected again

The exp_019 broadcast-view optimisation was retried after exp_022 fixed view
offsets. The isolated 42 op cases passed but the mandatory gradient gate failed
(42/43 overall, identical loss, wrong parameter gradients), so it was reverted
again; the API remains removed with a warning in the header. The exp_022 offset
fix was not the root cause. Final tree is unchanged from exp_022: captured
0.224 ms (ratio 0.300), eager 1.75 ms (ratio 0.604), parity 43/43. The earlier
0.213 ms / 0.312 from the broken path is **not** claimed. See
`experiments/exp_023/FINDINGS.md`.

## exp_022 — dual-output mul rejected; view-offset fix

A fused mul-backward kernel with a packed two-output buffer was tried. Isolated
checks matched the unfused muls, but the mandatory 43-case gradient gate failed
(loss identical, all parameter gradients wrong), so the path was reverted. The
investigation found and fixed a real bug: `bt_tensor_copy_to_host` and the
fused `sgd_step`/`reduce_sum`/`broadcast_grad`/`dual_mul` shaders ignored
`tensor->offset`, so non-zero-offset views were read from element 0 — the likely
root cause of exp_019's broadcast-view failure. No performance change (normal
tensors have offset 0): captured 0.220 ms (ratio 0.303), eager 1.56 ms (ratio
0.627), parity 43/43. See `experiments/exp_022/FINDINGS.md`.

## exp_021 — GEMM L2 swizzle (negative result, kept for honesty)

A 4×4 group-swizzled tile order (`BT_GEMM_VARIANT=6`) was measured against the
current vec4 winner in one controlled tuner run: 256² 0.227 vs **0.172** ms and
1024² 1.193 vs **0.985** ms — **1.21–1.32× slower**. The autotune cache keeps
variant 4; the swizzled kernel is retained only as a measured negative result.
The 1024² ratio remains 0.68–0.76 across clock-varying runs, still below 0.80.
See `experiments/exp_021/FINDINGS.md`.

## exp_019 — negative result (kept for honesty)

A zero-dispatch broadcast view for reduction gradients was tried. It improved
captured step time to 0.213 ms and graph ratio to 0.312, **but broke end-to-end
training** (MNIST loss 1.87 vs PyTorch 1.33, accuracy 0.434 vs 0.896) while the
42-case op harness still passed. It was reverted and the API removed; only the
bit-equivalent fused `tanh_backward` kernel was kept (dispatches 43 → 41, no
measurable captured-graph gain). Every reported number above is from the
parity-safe path. See `experiments/exp_019/FINDINGS.md`.

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
