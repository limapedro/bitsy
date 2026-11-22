# Progress tracker — implementation, numerics, performance

**Last updated:** 2026-11-22
**Repository tag:** `v0.5.0-cifar` (feature-completion work unreleased on a branch)
**Oracle / target:** PyTorch 2.11.0+cu126, CUDA 12.6, NVIDIA GeForce GTX 1660 SUPER

> Rule: no claim in this document is made without a committed report or a
> command output in this repository. `BENCHMARKS.md` and
> `tests/parity/reports/*.json` are the source of truth. Where a target is
> missed, the miss is stated rather than hidden.

Note: committed JSON reports still record the compiled version string
`0.2.0-bitsy` because the version macro was not advanced with later tags. This
is tracked as issue #20; it does not affect the numeric or timing values.

---

## 0. Feature completion + model parity (2026-11-22, exp_037/038/039)

New work toward "100% parity" for the functions needed by GPT-2/LSTM/RNN.
"100% parity" here means every implemented function matches PyTorch within the
documented tolerance on tested shapes; the full PyTorch API (thousands of
overloads) is out of scope. All results below have committed JSON/command
evidence; misses are stated, not hidden.

**Correctness fixes** (exp_037): ReLU backward was the identity, division
backward omitted `d/db` — both fixed. `log_softmax` ignored columns >256
(wrong CE loss at vocab 1024) — fixed. The register-blocked GEMM misread
transposed-view strides (zero embedding grad) — now falls back to the
stride-aware tiled kernel. Attention now uses a differentiable transpose.

**New primitives** (exp_037): `sqrt`, exact-erf `gelu` (+backward), `LayerNorm`,
`softmax`, `Embedding`, `Dropout` (eval/identity), `AdamW`, differentiable 2D
`transpose`. `mse_loss` corrected to PyTorch's mean reduction.

| API | Evidence (max abs diff vs PyTorch CUDA) | Result |
|---|---|---|
| GELU fwd/bwd | 1.2e-7 | pass |
| LayerNorm fwd/dx/dW/db | 2.4e-7 / 5.4e-7 / 6e-8 / 0 | pass |
| softmax dx | 1.7e-7 | pass |
| AdamW one-step W/b | 1.5e-8 / 9.3e-10 | pass |
| sqrt / gelu parity cases | 4 new cases | pass |

**Model parity** (exp_038/039, copied weights, CUDA oracle):

| model | params | loss / logits diff | grad diffs | result |
|---|---:|---|---|---|
| RNN (B3 T4 I5 H6) | — | h 5.9e-6 | dx 2.2e-7, dWih 1.3e-6 | pass (tol 1e-5) |
| LSTM (B3 T4 I5 H6) | — | h 2.1e-6, c 3.1e-6 | dx 7.7e-8 | pass (tol 1e-5) |
| GPT-2 small | 2,162,688 | loss 0.0, logits 1.16e-6 | embed 5.1e-7, head 2.4e-7, q0 4.3e-8 | pass |

**Performance — honest misses** (median, warmup, synced, live clock logged;
full distributions in `benchmarks/results/*.json`):

| model | Bitsy ms | PyTorch ms | ratio torch/Bitsy | clock |
|---|---:|---:|---:|---:|
| RNN forward | 1.75 (1.39–4.78) | 0.26 (0.19–0.54) | 0.15 | 1530 MHz |
| LSTM forward | 5.36 (4.25–7.33) | 0.22 (0.17–0.69) | 0.04 | 1530 MHz |
| GPT-2 forward | 25.42 (14.75–79.68) | 14.69 (9.80–24.77) | 0.58 | 1950 MHz |

Bitsy is slower on all three; cause is per-step/per-head Python-level dispatch
of small ops with submit/fence overhead versus fused cuDNN kernels. No speed
target is claimed; consistent with `docs/PERFORMANCE_CEILING.md`.

**Gate:** `python tests/parity/run.py` → **48/48 passed, 0 skipped** (was 44/44);
CTest → 100%. Details: `experiments/exp_037..039/FINDINGS.md`.

---

## 1. Surface status at a glance

Counting group-level rows in [FEATURE_INVENTORY.md](FEATURE_INVENTORY.md):

| Status | Rows | Meaning |
|---|---:|---|
| done | 9 | implemented and parity-verified where applicable |
| partial | 11 | subset of the group implemented |
| missing | 55 | no implementation |

This is a group count, not a function count. The implemented surface is
enumerated below.

---

## 2. Implemented functions — numeric parity vs PyTorch

All results below are from `python tests/parity/run.py` on this machine:
**40/40 cases passed, 0 skipped**, on cpu_ref and Vulkan, comparing against
PyTorch CPU and CUDA. Source: `tests/parity/reports/latest.json`.

| API | C symbol | Backends | Tolerance mode | Evidence | Result |
|---|---|---|---|---|---|
| `add` | `bt_add` | cpu_ref, Vulkan | exact | latest.json, 17 add cases | 0 mismatches (incl. scalar, empty, broadcast, ±inf/NaN/±0, 4096 elems) |
| `sub` | `bt_sub` | cpu_ref, Vulkan | exact | latest.json | 0/91 mismatches on both backends |
| `mul` | `bt_mul` | cpu_ref, Vulkan | exact | latest.json | 0/91 mismatches on both backends |
| `div` | `bt_div` | cpu_ref, Vulkan | exact | latest.json | 0/91 mismatches; Vulkan uses fp64 intermediate for correct rounding |
| `relu` | `bt_relu` | cpu_ref, Vulkan | exact | latest.json | 0/91 mismatches on both backends |
| `add_relu` (fused) | `bt_add_relu` | cpu_ref, Vulkan | exact | latest.json | 0/91 mismatches on both backends |
| `sigmoid` | `bt_sigmoid` | cpu_ref, Vulkan | rtol 1e-5, atol 1e-6 | latest.json | 0/91 mismatches on both backends |
| `tanh` | `bt_tanh` | cpu_ref, Vulkan | rtol 1e-5, atol 1e-6 | latest.json | 0/91 mismatches on both backends |
| `exp` | `bt_exp` | cpu_ref, Vulkan | rtol 1e-5, atol 1e-6 | latest.json | 0/91 mismatches on both backends |
| `log` | `bt_log` | cpu_ref, Vulkan | rtol 1e-5, atol 1e-6 | latest.json | 0/91 mismatches on both backends (positive inputs) |
| `matmul` (2D) | `bt_matmul` | cpu_ref, Vulkan | rtol 1e-5, atol 1e-5 | latest.json | 0 mismatches on (4,5)x(5,6) and (32,64)x(64,32) |
| `transpose` (2D) | `bt_transpose` | cpu_ref, Vulkan | exact (data movement) | unit test + used in backprop | matches PyTorch `.t()` in CTest and training |
| `view`/reshape | `bt_tensor_view` | both (no dispatch, shares storage) | exact | CTest + MNIST/CIFAR training | no numeric transform; used for flatten |
| `conv2d` fwd + dX/dW/dB | `bt_conv2d*` | cpu_ref, Vulkan | direct max-abs diff | `tests/parity/reports/conv_check.json` | fwd ≤ 4.8e-7, dW ≤ 7.6e-6, dB ≤ 2.9e-6, dX ≤ 1.9e-6 |

### Gradient and end-to-end numeric parity

| Check | Evidence | Result |
|---|---|---|
| MLP gradient parity (loss, W grad, b grad) | `tests/parity/reports/gradcheck.json` | 1.19e-07, 5.96e-08, 1.19e-07 |
| MNIST MLP, 10 epochs | `benchmarks/results/mnist_mlp_parity.json` | max epoch loss diff 2.78e-07, accuracy identical, final W1 diff 1.49e-07 |
| CIFAR-10 CNN, 3 epochs | `benchmarks/results/cifar_cnn_parity.json` | max epoch loss diff 1.64e-07, accuracy identical each epoch, final conv1 W diff 7.0e-07, logit diff 1.24e-05, argmax agreement 1.0 |

The small nonzero values are expected fp32 accumulation-order differences; the
exact-mode ops are bit-identical apart from NaN payloads.

---

## 3. Implemented functions — performance vs PyTorch CUDA

Ratio = Bitsy throughput ÷ PyTorch throughput (higher is better). Wall time
includes `bt.flush()` + `bt.wait()` (async submit, then host completion) per
iteration on the Bitsy side and `torch.cuda.synchronize()` on the PyTorch side.
Bitsy device time is from Vulkan timestamp queries; PyTorch device time is not
measured, so it is not compared.
Sources: `benchmarks/results/elementwise.json`, `benchmarks/results/gemm.json`,
`BENCHMARKS.md`.

| Op | Shape | Bitsy ms | PyTorch ms | Bitsy device ms | Ratio | Target | Gap |
|---|---|---:|---:|---:|---:|---:|---|
| add | 4096 | 0.0689 | 0.0365 | 0.0019 | 0.53 | ≥0.95 | miss 0.42 |
| mul | 4096 | 0.1137 | 0.0334 | 0.0020 | 0.29 | ≥0.95 | miss 0.66 |
| relu | 4096 | 0.1142 | 0.0400 | 0.0020 | 0.35 | ≥0.95 | miss 0.60 |
| add | 1,048,576 | 0.1522 | 0.0556 | 0.0479 | 0.36 | ≥0.95 / ≥90% peak | miss 0.59 |
| mul | 1,048,576 | 0.1746 | 0.0621 | 0.0521 | 0.36 | ≥0.95 / ≥90% peak | miss 0.59 |
| relu | 1,048,576 | 0.1840 | 0.0429 | 0.0492 | 0.23 | ≥0.95 / ≥90% peak | miss 0.72 |
| GEMM (naive) | 1024² | 12.722 | 0.908 | 12.464 | 0.07 | ≥0.80 | miss 0.73 |
| GEMM (tiled) | 1024² | 4.329 | 0.950 | 4.088 | 0.22 | ≥0.80 | miss 0.58 |
| GEMM (rb vec4 + double buffer, autotune) | 1024² | 1.289 | 0.893 | 1.035 | 0.69 | ≥0.80 | miss 0.11 |
| GEMM (tiled, autotune) | 256² | 0.134 | 0.042 | 0.070 | 0.31 | ≥0.80 | miss 0.49 |

Achieved throughput: large elementwise peaks at ~77 GiB/s vs PyTorch ~211 GiB/s;
register-blocked vec4 GEMM reaches 1666–1823 GFLOPS depending on GPU clock
(1530–1950 MHz) versus PyTorch 2350–2400 GFLOPS at n=1024 (up from 496 GFLOPS
with the tiled kernel). exp_010's 16-deep K-step variant was 1.59× slower and
is excluded by the measured autotune cache.

**No performance target is met.** Conv2d performance has not been benchmarked
at all — only its numerics. Ratios vary between snapshots mainly because
PyTorch's small-op timings move (add 1M: 0.0868 ms in exp_003 → 0.0556 ms here).

**Batched multi-op workload** (`benchmarks/ops/batch_chain.py`, 20 adds of 1M
elements): eager submit-per-op 2.773 ms → batched 1.277 ms (2.17x) → batched
with timeline semaphore 1.251 ms (~2% more). The async prediction (≥1.2x) was
**not met** because the chain already had one submission.

**Fusion** (`benchmarks/ops/fusion.py`, 20 add+relu pairs of 1M elements):
separate add then relu 2.365 ms → fused `add_relu` 1.200 ms, a **1.97x** speedup,
with device time 1.915 ms → 0.935 ms. First fused kernel, not a general pass.

Experiment trail:
- `experiments/exp_001` — caching allocator: 4.2–5.7× lower wall time.
- `experiments/exp_002` — tiled SGEMM: 2.36× faster than naive at n=1024.
- `experiments/exp_003` — batched submissions (2.17× on a 20-op chain) + timestamps.
- `experiments/exp_004` — timeline async (prediction not met), fused add+relu (1.97×).

---

## 4. Tolerance policy actually in force

Source: `tests/parity/tolerances.py`.

| Class | Ops | Mode | Values |
|---|---|---|---|
| Exact | add, add_relu, sub, mul, div, relu, data movement | `exact` | bit-identical fp32 patterns; NaN payloads ignored |
| ULP-class (current bound) | sigmoid, tanh, exp, log | `rtol_atol` | rtol 1e-5, atol 1e-6 |
| Reduction/GEMM | matmul | `rtol_atol` | rtol 1e-5, atol 1e-5 |
| Conv | conv2d + backward | direct max-abs diff in `conv_check.py` | no formal tolerance entry yet |

The ULP distributions against a float64 reference required by the original
tolerance policy are **not measured yet**; the rtol/atol bounds above are the
current, documented stand-in. Any loosening requires an
`ARCHITECTURE_LOG.md` entry.

---

## 5. Open issues that block "professional grade"

1. **Performance:** every measured ratio is below target. Batched submissions,
   timestamp queries, timeline-semaphore async submission and a first fused
   kernel are done (exp_003/004); still needed: general fusion pass,
   register-blocked and autotuned GEMM, vectorized loads, and overlapping host
   work with GPU execution across independent batches.
2. **C autograd:** gradients currently come from an interim Python graph; hooks,
   `no_grad`, and double backward are missing.
3. **Correctness:** intermittent wrong logits on very large Vulkan eval batches
   (chunked eval is the verified path). Recorded in `ARCHITECTURE_LOG.md` and
   `DEVIATIONS.md`.
4. **Surface:** 55 missing feature groups in `FEATURE_INVENTORY.md`, led by
   indexing/data movement, reductions, normalization/attention, DataLoader, and
   checkpoint interop.
5. **Hygiene:** stale version string in reports (issue #20), no validation-layer
   integration in CI, no sanitizers or fuzzing yet.

---

## 6. How to refresh this document

```powershell
python tests/parity/run.py                 # 42/42, overwrites reports/latest.json
python tests/parity/gradcheck.py           # gradient parity
python tests/parity/conv_check.py          # conv forward/backward parity
python benchmarks/ops/elementwise.py       # elementwise ratios + device time
$env:BT_GEMM_TILED="0"; python benchmarks/ops/gemm.py   # naive GEMM baseline
$env:BT_GEMM_TILED="1"; python benchmarks/ops/gemm.py   # tiled GEMM
$env:BT_VK_NO_BATCH="1"; python benchmarks/ops/batch_chain.py  # eager chain
$env:BT_VK_NO_BATCH="0"; python benchmarks/ops/batch_chain.py  # batched chain
python benchmarks/ops/fusion.py             # fused add+relu vs separate
python benchmarks/models/mnist_mlp.py      # MNIST loss curves
python benchmarks/models/cifar_cnn.py      # CIFAR loss curves (needs data/cifar_subset.npz; see tools/export_cifar.py)
```

Then update the numbers in this file and in `BENCHMARKS.md` together, and keep
`experiments/exp_NNN/` before/after snapshots untouched.

---

## 7. Next priorities (in order)

1. Fix the large-batch Vulkan eval anomaly with a dedicated synchronization/race
   experiment.
2. General pointwise fusion beyond `add+relu` (bias+activation, softmax+mask);
   re-benchmark end-to-end training steps with fused kernels.
3. Register-blocked/vectorized GEMM with per-device autotuning (new `exp_NNN`).
4. Native reduction kernels (`sum`, `mean`, `max`, `argmax`) so log_softmax and
   normalization no longer need matmul-based stand-ins.
5. Move autograd into C; add hooks and `no_grad`.
6. Fill the missing API groups from `FEATURE_INVENTORY.md`, earliest first:
   indexing/data movement → reductions → normalization/attention → optim/schedulers
   → DataLoader → checkpoint/DLPack interop.
