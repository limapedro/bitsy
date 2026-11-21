# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
