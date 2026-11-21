# Roadmap

This file is kept in sync with the milestones in the project brief. Each
milestone ends with all tests green, a parity report, benchmarks where relevant,
updates to this file and `ARCHITECTURE_LOG.md`, and a tagged release.

Status legend: **done** / **in progress** / **planned**.

| # | Name | Focus | Status |
|---|---|---|---|
| M0 | Anvil | Repository, CMake, CI, conventions, empty Python package, parity harness skeleton, first `add` on `cpu_ref` | **in progress** |
| M1 | Ingot | Tensor core on `cpu_ref`: storage, views, strides, broadcasting, creation, elementwise, reductions, full parity | planned |
| M2 | Forge | Vulkan bring-up: device, allocator, streams, pipelines, shader build, transfers, first elementwise kernels | planned |
| M3 | Quench | Core op coverage on Vulkan: reductions, indexing, movement, sorting, cumulative, RNG parity | planned |
| M4 | Strike | Autograd: graph, engine, backward formulas, hooks, `no_grad`, version counters, gradcheck | planned |
| M5 | Temper | GEMM: naive → tiled → register-blocked, batched, FP16-storage, fused epilogues | planned |
| M6 | Blade | Python library: CPython extension, full `Tensor` surface, NumPy/DLPack interop, wheels | planned |
| M7 | Whetstone | `nn` and `optim`: modules, layers, losses, init, optimizers, schedulers, fused updates | planned |
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
- [x] Empty `hagane` Python package imports
- [x] Dataclass-configured parity harness skeleton
- [x] Tolerance table with recorded reasoning
- [ ] `add` implemented on `cpu_ref` and matching PyTorch (forward)
- [ ] M0 status report and tag `v0.1.0-m0`
