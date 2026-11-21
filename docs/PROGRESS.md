# Progress tracker — implementation, numerics, performance

**Last updated:** 2026-11-21
**Repository tag:** `v0.5.0-cifar`
**Oracle / target:** PyTorch 2.11.0+cu126, CUDA 12.6, NVIDIA GeForce GTX 1660 SUPER

> Rule: no claim in this document is made without a committed report or a
> command output in this repository. `BENCHMARKS.md` and
> `tests/parity/reports/*.json` are the source of truth. Where a target is
> missed, the miss is stated rather than hidden.

Note: committed JSON reports still record the compiled version string
`0.2.0-bitsy` because the version macro was not advanced with later tags. This
is tracked as issue #20; it does not affect the numeric or timing values.

---

## 1. Surface status at a glance

Counting group-level rows in [FEATURE_INVENTORY.md](FEATURE_INVENTORY.md):

| Status | Rows | Meaning |
|---|---:|---|
| done | 7 | implemented and parity-verified where applicable |
| partial | 10 | subset of the group implemented |
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

Ratio = Bitsy throughput ÷ PyTorch throughput (higher is better). All numbers
are wall time with sync per iteration; device timestamp queries are not
implemented yet. Sources: `benchmarks/results/elementwise.json`,
`benchmarks/results/gemm.json`, `BENCHMARKS.md`.

| Op | Shape | Bitsy ms | PyTorch ms | Ratio | Target | Gap |
|---|---|---:|---:|---:|---:|---|
| add | 4096 | 0.0772 | 0.0132 | 0.17 | ≥0.95 | miss 0.78 |
| mul | 4096 | 0.0703 | 0.0246 | 0.35 | ≥0.95 | miss 0.60 |
| relu | 4096 | 0.0942 | 0.0167 | 0.18 | ≥0.95 | miss 0.77 |
| add | 1,048,576 | 0.1551 | 0.0553 | 0.36 | ≥0.95 / ≥90% peak | miss 0.59 |
| mul | 1,048,576 | 0.1514 | 0.0537 | 0.35 | ≥0.95 / ≥90% peak | miss 0.60 |
| relu | 1,048,576 | 0.1616 | 0.0421 | 0.26 | ≥0.95 / ≥90% peak | miss 0.69 |
| GEMM (naive) | 1024² | 10.224 | 0.908 | 0.09 | ≥0.80 | miss 0.71 |
| GEMM (tiled) | 1024² | 4.337 | 0.961 | 0.22 | ≥0.80 | miss 0.58 |
| GEMM (tiled) | 256² | 0.146 | 0.046 | 0.31 | ≥0.80 | miss 0.49 |

Achieved throughput: elementwise peaks at ~77 GiB/s vs PyTorch ~212 GiB/s;
tiled GEMM reaches 495 GFLOPS vs PyTorch 2234 GFLOPS at n=1024.

**No performance target is met.** Conv2d performance has not been benchmarked
at all — only its numerics. Unmeasured ops are deliberately absent from this
table rather than assumed fast.

Experiment trail:
- `experiments/exp_001` — caching allocator: 4.2–5.7× lower wall time.
- `experiments/exp_002` — tiled SGEMM: 2.36× faster than naive at n=1024.

---

## 4. Tolerance policy actually in force

Source: `tests/parity/tolerances.py`.

| Class | Ops | Mode | Values |
|---|---|---|---|
| Exact | add, sub, mul, div, relu, data movement | `exact` | bit-identical fp32 patterns; NaN payloads ignored |
| ULP-class (current bound) | sigmoid, tanh, exp, log | `rtol_atol` | rtol 1e-5, atol 1e-6 |
| Reduction/GEMM | matmul | `rtol_atol` | rtol 1e-5, atol 1e-5 |
| Conv | conv2d + backward | direct max-abs diff in `conv_check.py` | no formal tolerance entry yet |

The ULP distributions against a float64 reference required by the original
tolerance policy are **not measured yet**; the rtol/atol bounds above are the
current, documented stand-in. Any loosening requires an
`ARCHITECTURE_LOG.md` entry.

---

## 5. Open issues that block "professional grade"

1. **Performance:** every measured ratio is below target. Needs batched
   submissions/timeline semaphores, GPU timestamp queries, register-blocked and
   autotuned GEMM, vectorized loads, and pointwise fusion.
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
python tests/parity/run.py                 # 40/40, overwrites reports/latest.json
python tests/parity/gradcheck.py           # gradient parity
python tests/parity/conv_check.py          # conv forward/backward parity
python benchmarks/ops/elementwise.py       # elementwise ratios
$env:BT_GEMM_TILED="0"; python benchmarks/ops/gemm.py   # naive GEMM baseline
$env:BT_GEMM_TILED="1"; python benchmarks/ops/gemm.py   # tiled GEMM
python benchmarks/models/mnist_mlp.py      # MNIST loss curves
python benchmarks/models/cifar_cnn.py      # CIFAR loss curves (needs data/cifar_subset.npz; see tools/export_cifar.py)
```

Then update the numbers in this file and in `BENCHMARKS.md` together, and keep
`experiments/exp_NNN/` before/after snapshots untouched.

---

## 7. Next priorities (in order)

1. Fix the large-batch Vulkan eval anomaly with a dedicated synchronization/race
   experiment.
2. Batched command submission + timestamp queries, then re-benchmark; publish
   device time instead of only wall time.
3. Register-blocked/vectorized GEMM with per-device autotuning (new `exp_NNN`).
4. Native reduction kernels (`sum`, `mean`, `max`, `argmax`) so log_softmax and
   normalization no longer need matmul-based stand-ins.
5. Move autograd into C; add hooks and `no_grad`.
6. Fill the missing API groups from `FEATURE_INVENTORY.md`, earliest first:
   indexing/data movement → reductions → normalization/attention → optim/schedulers
   → DataLoader → checkpoint/DLPack interop.
