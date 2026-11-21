# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (TOGISHI Phase 5, exp_013)

- `bt_vulkan_capture_begin/end/replay` record-once command buffers with
  timeline replay (`bt.capture_begin()`, `capture_end()`, `replay(n)` in Python).
- 20-add chain: 1.080 → 0.899 ms/chain (1.20×); Bitsy replay versus PyTorch
  CUDA graph replay is **1.00×** — first measured performance-parity result,
  limited to the captured path. The ≥1.5× prediction was not met (GPU work is
  now the floor). Default eager ratios unchanged; parity 42/42. Details in
  `experiments/exp_013/FINDINGS.md`.

### Added (TOGISHI Phase 6, exp_012)

- vec4/aligned elementwise fast path with linear indexing and a safe scalar
  fallback for broadcast/strided/odd shapes.
- Large-tensor device bandwidth 269–284 GB/s; 20-add chain reaches 293 GB/s
  (96% of measured DRAM ceiling). Wall ratios remain 0.43–0.49 because
  submit+fence dominates; the ≥1.3× prediction was not met. Parity 42/42.
  Details in `experiments/exp_012/FINDINGS.md`.

### Added (TOGISHI Phase 5, exp_011)

- Per-run GPU clock logging in `gemm.py`/`elementwise.py` via
  `benchmarks/common/clock.py`.
- Hazard-tracked per-buffer Vulkan barriers replacing the unconditional global
  barrier; transfer barriers unchanged.
- Validated on-disk pipeline-cache load path (persistence blocked by driver
  621.01, documented in `experiments/exp_011/FINDINGS.md`).
- `benchmarks/ops/async_chain.py`: 1.10× from submitting 8 chains before one
  wait (below the 1.2× prediction). Elementwise ratios 0.34–0.54 at 1530 MHz;
  parity 42/42.

### Added (TOGISHI Phase 6, exp_010)

- On-disk GEMM autotune cache (`benchmarks/results/gemm_autotune.txt`) written
  by `benchmarks/ops/gemm_tune.py` and read by the Vulkan backend; verified by
  profiler trace names.
- 16-deep K-step vec4 GEMM variant: measured **1.59× slower** than the 8-deep
  kernel and kept only as a documented negative result.
- GEMM ratio remains 0.69–0.78 across GPU clocks 1530–1950 MHz; target 0.80
  still missed. Parity 42/42. Details in `experiments/exp_010/`.

### Added (TOGISHI Phase 6, exp_009)

- vec4 global loads and double-buffered shared tiles for the register-blocked
  GEMM, with stride checks and a safe scalar fallback (`BT_GEMM_VARIANT=4`).
- 1024² GEMM 1.224 → 1.178 ms (1754 → 1823 GFLOPS, ratio 0.77 → 0.78). The
  ≥1.08× prediction and ≥0.80 target are recorded as missed in
  `experiments/exp_009/FINDINGS.md`. Parity 42/42.

### Added (TOGISHI Phase 6, exp_008)

- GEMM ladder: register-blocked 128×128×8 kernel, conflict-free shared memory,
  split-K variant (negative result), and shape-based autotune.
- 1024² GEMM: 4.329 → 1.224 ms (3.54×), 496 → 1754 GFLOPS, ratio 0.22 → 0.77;
  target 0.80 still missed. Parity 42/42. Raw variant snapshots in
  `experiments/exp_008/`.

### Added (TOGISHI Phase 2, exp_007)

- Opt-in `hg.profiler`: per-dispatch GPU timestamps, CPU enqueue time,
  bytes/FLOPs versus calibrated ceilings, Chrome-trace JSON, and counters for
  dispatches/barriers/submits/allocations.
- Three-way timing and PyTorch kernel profiling scripts plus ranked
  `GAP_LEDGER.md`. Top gap: GEMM 155.5 µs (3.7% of FP32 peak) versus cuBLAS
  `volta_sgemm_128x32_nn` + split-K.
- No framework optimization in this phase; parity re-verified at 42/42.

### Added (M10 experiment, exp_004)

- Timeline-semaphore async submission: `bt_vulkan_flush()` no longer waits;
  `bt_vulkan_wait()` is the explicit host completion point and also collects
  timestamp results.
- Fused `bt_add_relu` kernel on both backends; 20-pair chain is 1.97x faster
  than separate add + relu (`experiments/exp_004/fusion.json`).
- Parity suite extended to 42 cases including exact fused-op checks. Async
  submission alone did not meet its speed prediction (2% on an already-batched
  chain) and is documented as such.

### Added (M10 experiment)

- Batched Vulkan submissions: dispatches are recorded until `bt_vulkan_flush()`
  or host readback; rotating descriptor sets preserve per-dispatch bindings.
- GPU timestamp queries with `bt_vulkan_device_time_ms()` when the device
  exposes `timestampComputeAndGraphics`.
- `experiments/exp_003` with hypothesis, before/after snapshots and findings:
  20-op chain 2.17x faster (2.773 → 1.277 ms). Single-op ratios remain below
  target and are reported honestly in `BENCHMARKS.md` / `docs/PROGRESS.md`.

### Added (docs)

- `docs/FEATURE_INVENTORY.md`: full PyTorch-grade API surface with per-group status.
- `docs/PROGRESS.md`: implemented functions with numeric differences, benchmark
  ratios versus targets, training parity, tolerance policy, and next priorities.

### Added (M5 experiment)

- `experiments/exp_002`: hypothesis-first 16x16x16 shared-memory tiled SGEMM.
  Measured 2.36x faster at n=1024 (210 -> 495 GFLOPS); ratio 0.09 -> 0.22,
  still below the GEMM target. Raw before/after snapshots and findings included.
- `benchmarks/ops/gemm.py` with published results in `BENCHMARKS.md`.

### Added (M8 in progress)

- NCHW `Conv2d` with dX/dW/dB kernels on cpu_ref and Vulkan, a separate conv
  dispatcher, and a committed conv gradcheck (`tests/parity/conv_check.py`).
- CIFAR-10 CNN parity script and result: 3 epochs with loss curves and accuracy
  matching PyTorch CUDA, final conv1 weight diff 7e-7, logit diff 1.2e-5.
- `tools/export_cifar.py` regenerates the gitignored HF CIFAR subset.
- Known issue recorded: intermittent large-batch Vulkan eval anomaly; chunked
  evaluation is used and the anomaly is not claimed fixed.

### Added (M4/M5/M7 in progress)

- `bt_matmul`, `bt_transpose` and `bt_log` on cpu_ref and Vulkan; 40/40 parity
  cases across ten ops and both backends.
- Interim Python autograd/nn/optim surface with device-native compute:
  `Tensor.backward`, `Linear`, `Tanh`, `Sequential`, `mse_loss`,
  `cross_entropy`, `SGD`.
- Committed gradient parity (`tests/parity/reports/gradcheck.json`) and MNIST
  training parity (`benchmarks/results/mnist_mlp_parity.json`): 10 epochs with
  loss curves and accuracy matching PyTorch CUDA.

### Added (M2 in progress)

- Vulkan compute backend on the GTX 1660 Super: device selection, embedded
  SPIR-V shader, device buffers with staging transfers and fence waits.
- Eight fp32 elementwise ops on both backends, with parity against PyTorch
  CPU and CUDA (19/19 cases, including 9 Vulkan cases).
- `benchmarks/ops/elementwise.py` and the first honest wall-time comparison
  against PyTorch CUDA in `BENCHMARKS.md` (targets currently missed).
- `experiments/exp_001`: caching allocator, 5–6x lower latency, findings kept
  including the before snapshot.

## [0.2.0-bitsy] - 2026-11-21

### Changed

- Full rename: Hagane → Bitsy. GitHub repo `limapedro/bitsy`, public headers
  `include/bitsy/bt_*.h`, C ABI `bt_*` / `BT_*`, shared library `bitsy.dll` /
  `libbitsy.so`, Python import `import bitsy as bt`. The previous exact-name
  repo was preserved as `limapedro/bitsy-legacy-2023` (nothing deleted).
- Code style now follows tinygrad / geohot: 2-space indentation in C and
  Python, dense one-line functions, section banners, and short comments that
  explain *why*. Rules and source links are in `ARCHITECTURE_LOG.md`.
- Re-verified after both changes: CTest 1/1, parity 13/13 bit-exact against
  PyTorch CPU and CUDA on the GTX 1660 SUPER.

## [0.1.0-m0] - 2026-11-21

### Added

- Repository conventions, license, PyTorch attribution, and the initial
  architecture decision log (M0 "Anvil").
- CMake + Ninja build and Windows/Linux CI, including a `glslc` to SPIR-V
  helper for the Vulkan shaders that arrive in M2.
- C unit-test harness and the dataclass-configured parity harness skeleton.
- Initial `add` operator on the `cpu_ref` backend with PyTorch broadcasting
  semantics; 13/13 parity cases pass bit-exactly against PyTorch CPU and CUDA.
- M0 status report (`docs/m0_status.md`) and committed parity report.
