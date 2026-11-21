# Roadmap

This file is kept in sync with the milestones in the project brief. Each
milestone ends with all tests green, a parity report, benchmarks where relevant,
updates to this file and `ARCHITECTURE_LOG.md`, and a tagged release.

Status legend: **done** / **in progress** / **planned**.

| # | Name | Focus | Status |
|---|---|---|---|
| M0 | Anvil | Repository, CMake, CI, conventions, empty Python package, parity harness skeleton, first `add` on `cpu_ref`; renamed to Bitsy and restyled | **done** (`v0.2.0-bitsy`, supersedes `v0.1.0-m0`) |
| M1 | Ingot | Tensor core on `cpu_ref`: storage, views, strides, broadcasting, creation, elementwise, reductions, full parity | planned |
| M2 | Forge | Vulkan bring-up: device, allocator, streams, pipelines, shader build, transfers, first elementwise kernels | **in progress** |
| M3 | Quench | Core op coverage on Vulkan: reductions, indexing, movement, sorting, cumulative, RNG parity | planned |
| M4 | Strike | Autograd: graph, engine, backward formulas, hooks, `no_grad`, version counters, gradcheck | **in progress** (interim Python graph + gradcheck passing) |
| M5 | Temper | GEMM: naive → tiled → register-blocked, batched, FP16-storage, fused epilogues | **in progress** (naive kernel + parity done; tuning pending) |
| M6 | Blade | Python library: CPython extension, full `Tensor` surface, NumPy/DLPack interop, wheels | planned |
| M7 | Whetstone | `nn` and `optim`: modules, layers, losses, init, optimizers, schedulers, fused updates | **in progress** (minimal Linear/Sequential/SGD/CE done) |
| M8 | Edge | Convolution and pooling: direct, im2col+GEMM, Winograd/FFT, small CNN | planned |
| M9 | Grain | Normalization, attention, sequence models, small transformer | planned |
| M10 | Hamon | Fusion and overhead: pointwise fusion, launch reduction, allocator tuning | planned |
| M11 | Temper II | Checkpoint compatibility, safetensors, DLPack, autocast, gradient checkpointing | planned |
| M12 | Katana | Hardening: fuzzing, sanitizers, validation layers, docs, release 1.0 | planned |

## Current milestone: M0 "Anvil"

Exit criteria:

- [x] Repository created and public-facing docs written
- [x] Core architectural decisions logged with dates
- [x] CMake + Ninja build for Windows and Linux
- [x] CI workflow running on Windows and Linux
- [x] C unit-test harness wired into CTest
- [x] Empty `bitsy` Python package imports
- [x] Dataclass-configured parity harness skeleton
- [x] Tolerance table with recorded reasoning
- [x] `add` implemented on `cpu_ref` and matching PyTorch (forward) — 13/13 cases, bit-exact
- [x] M0 status report and tag `v0.1.0-m0`
- [x] Rename Hagane → Bitsy everywhere (repo, docs, headers, C ABI, Python import)
- [x] tinygrad/geohot style refactor with explanatory comments
- [x] Re-verify after rename + restyle: CTest 1/1, parity 13/13, tag `v0.2.0-bitsy`

See `docs/m0_status.md` for the evidence and known limitations. The prior exact
name `bitsy` was preserved as `limapedro/bitsy-legacy-2023`, not deleted.

## Current work: M2 "Forge"

- [x] Vulkan instance/device selection on the GTX 1660 Super
- [x] Device buffers, staging upload/readback, one command stream with fence waits
- [x] Build-time GLSL → SPIR-V embedded into the shared library
- [x] 8 fp32 elementwise kernels (add, sub, mul, div, relu, sigmoid, tanh, exp)
- [x] Parity vs PyTorch CUDA: 19/19 cases including 9 Vulkan cases
- [x] First benchmark against CUDA, published honestly in `BENCHMARKS.md`
- [x] exp_001 caching allocator: 5–6x faster, ratios 0.20–0.45 (target still missed)
- [ ] Timestamp queries, validation layers in CI, asynchronous streams
- [ ] M2 status report and tagged release

Performance targets are not met; see `experiments/exp_001/FINDINGS.md`.

## Current work: M4/M5/M7 — GEMM, gradients, MNIST

- [x] `bt_matmul` / `bt_transpose` on cpu_ref and Vulkan, 3 parity cases
- [x] `bt_log` on both backends for cross-entropy
- [x] 40/40 parity cases across 10 ops and both backends
- [x] Gradient parity vs PyTorch CUDA: loss diff 1.2e-7, W grad 6.0e-8 (`tests/parity/reports/gradcheck.json`)
- [x] MNIST MLP: 10 epochs, loss curves and accuracy identical to PyTorch CUDA, final W1 diff 1.5e-7 (`benchmarks/results/mnist_mlp_parity.json`)
- [ ] Move autograd into C; replace unbroadcast CPU roundtrip with a reduction kernel
- [ ] Tiled/autotuned GEMM experiment; CIFAR/text/GAN/quantization parity
