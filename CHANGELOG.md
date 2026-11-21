# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Rejected (TOGISHI Phase 5/6, exp_023)

- Zero-dispatch stride-0 broadcast views were retried after the exp_022 offset
  fix. Isolated 42 op cases passed but the mandatory gradient gate failed
  (42/43, identical loss, wrong parameter gradients), so the change was
  reverted and the API removed again with a header warning. The offset fix was
  not the root cause. Final tree unchanged from exp_022: captured 0.224 ms /
  ratio 0.300, eager 1.75 ms / ratio 0.604, parity 43/43. Details in
  `experiments/exp_023/FINDINGS.md`.

### Fixed (TOGISHI Phase 5/6, exp_022)

- `bt_tensor_copy_to_host` and the fused `sgd_step`/`reduce_sum`/
  `broadcast_grad`/`dual_mul` shaders ignored `tensor->offset`; added
  `bt_vk_readback_offset()` and offset push constants so non-zero-offset views
  read correctly. Likely root cause of exp_019's broadcast-view gradient bug.
- A fused dual-output mul backward was tried and **reverted**: isolated outputs
  matched but the 43-case gradient gate failed with identical loss and wrong
  gradients. Kept as a negative-result variant. No perf change: captured 0.220
  ms (ratio 0.303), eager 1.56 ms (ratio 0.627), parity 43/43. Details in
  `experiments/exp_022/FINDINGS.md`.

### Added (TOGISHI Phase 6, exp_021)

- `gemm_rb_vec4_swz` (variant 6): 4×4 group-swizzled tile order, measured and
  **rejected** — 1.32× slower at 256², 1.21× slower at 1024² than the current
  vec4 winner. Kept as a negative-result variant; autotune still selects
  variant 4. 1024² GEMM ratio remains 0.68–0.76 vs ≥0.80. Parity 43/43 incl.
  gradient gate, CTest both builds. Details in `experiments/exp_021/FINDINGS.md`.

### Added (TOGISHI Phase 2/6, exp_020)

- **Required end-to-end gradient-flow gate** (`tests/parity/grad_flow.py`) in
  `run.py` (43/43) and both CI jobs; compares loss, all parameter gradients and
  post-step parameters against PyTorch. Proven to catch a corrupted gradient
  whose loss still matched.
- Dedicated `broadcast_grad` kernel replacing `ones * grad` with a contiguous
  full-shape gradient (no unsafe stride-0 view). Captured step 0.240 → 0.219 ms
  (ratio 0.304), eager 1.669 → 1.57 ms (ratio 0.62). Dispatch count unchanged
  at 41; graph ≥0.31 just missed. Parity 43/43, MNIST/CIFAR within tolerance.
  Details in `experiments/exp_020/FINDINGS.md`.

### Added (TOGISHI Phase 6, exp_019)

- Fused `tanh_backward` kernel (`g * (1 - t*t)`, one dispatch) with a CPU path,
  wired into autograd. Dispatches/step 43 → 41; no measurable captured-graph
  gain (0.240 ms, ratio 0.278), eager 1.669 ms (ratio 0.569). Parity 42/42,
  MNIST W1 diff 1.788e-07.
- **Reverted/removed:** zero-dispatch broadcast views for reduction gradients.
  They improved captured time to 0.213 ms (ratio 0.312) but broke training
  parity (MNIST loss 1.87 vs 1.33, W1 diff 0.149) while the isolated 42-case
  harness stayed green. End-to-end training parity is now a required gate for
  autograd/view changes. See `experiments/exp_019/FINDINGS.md`.

### Added (TOGISHI Phase 6, exp_018)

- `reduce_sum.comp` + `bt_reduce_sum()` C/Python API with an autograd-aware
  `reduce_sum_op` backward; ones-vector GEMM reductions in `sum`,
  `log_softmax`, `cross_entropy`, `mse_loss` and `_unbroadcast` replaced.
- Tiled GEMMs per step 20 → 10, total dispatches 46 → 43. Eager step 2.46 →
  1.898–1.955 ms (1.26–1.30×, ratio 0.55–0.56 — best yet); captured step
  0.257 → 0.234–0.246 ms, graph ratio 0.259 → 0.277–0.287 (≥0.29 just missed).
  Parity 42/42, CTest both builds, MNIST identical, CIFAR W diff 6.4e-07.
  Details in `experiments/exp_018/FINDINGS.md`.

### Added/Changed (TOGISHI Phase 5/6, exp_017)

- `bt_tensor_transpose_view()` / `C.transpose_view()`: zero-dispatch 2D
  transpose view; matmul backward now uses it instead of materialising a
  transposed buffer.
- Fused `sgd_step.comp` + `bt_sgd_step()` API: `w - lr*grad` in one dispatch
  with lr as a push constant (CPU-only path uses existing ops).
- Dispatch count per MLP step 66 → 46 (30.3%); captured step 0.296–0.334 →
  0.257 ms (1.15–1.30×), graph ratio 0.20–0.23 → 0.259 (≥0.28 prediction
  missed); eager step 2.626 → 2.46 ms, ratio 0.41–0.47 (best yet). Parity
  42/42, CTest both builds, MNIST/CIFAR drift < 2e-7. Details in
  `experiments/exp_017/FINDINGS.md`.

### Changed (TOGISHI Phase 6, exp_016)

- Device-side unbroadcast reductions for 2D gradients replace the CPU roundtrip
  in backward (CPU fallback retained for >2D). Capture-phase submit count drops
  2 → 1 (single-chunk MLP capture); the default eager step improves 3.220 →
  2.626–2.911 ms (1.1–1.2×, ratio 0.37–0.41). Captured graph throughput does
  NOT improve (0.20–0.23 vs PyTorch CUDA graphs; ≥0.28 prediction missed) —
  ~66 small dispatches per step now dominate. Parity 42/42, MNIST identical,
  CIFAR unchanged; reduction drift 1.07e-7 relative. Details in
  `experiments/exp_016/FINDINGS.md`.

### Added (TOGISHI Phase 5, exp_015)

- Full training-step capture/replay: captures are split into command-buffer
  chunks at host readbacks (the interim autograd reduces bias gradients on the
  CPU), with resources pinned until the final wait. A single buffer deadlocked
  on unexecuted timestamp queries; a recycling version was bit-wrong and was
  rejected. Replay after the final wait requires re-capture (documented).
- Captured step is bit-identical to eager (abs diff 0.0) and runs in
  0.294–0.355 ms/step vs 3.220 ms eager (~9–11×), but only 0.19–0.23× of
  PyTorch CUDA graph throughput — both end-to-end targets missed.
  New `benchmarks/models/graph_step_bench.py`. Parity 42/42. Details in
  `experiments/exp_015/FINDINGS.md`.

### Changed (TOGISHI Phase 4/5, exp_014)

- Deferred transfers are now the **default**: uploads/readbacks use a pooled
  host-visible staging buffer and enqueue the copy instead of flushing and
  waiting per call. `BITSY_DEFERRED_UPLOAD=0` restores the old synchronous path.
- Training-step wall time 9.575 → 3.220 ms/step at matched 1530 MHz (**2.97×**),
  0 staging allocations over 200 timed steps. The ≥0.85 end-to-end throughput
  target is still missed (0.34–0.37); remaining cost is per-op dispatch/autograd
  overhead. Parity 42/42; MNIST/CIFAR Bitsy losses bit-identical between modes.
  New `benchmarks/models/train_step_bench.py` and `bt.stage_alloc_count()`.
  Details in `experiments/exp_014/FINDINGS.md`.

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
