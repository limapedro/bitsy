# Bitsy

[![CI](https://github.com/limapedro/bitsy/actions/workflows/ci.yml/badge.svg)](https://github.com/limapedro/bitsy/actions/workflows/ci.yml)

Bitsy is a deep learning framework written from scratch in **C11** with a
**Vulkan compute** backend and a Python front end that mirrors PyTorch.
The goal is that PyTorch code moves to Bitsy with minimal edits
(`import bitsy as bt`). The M0 bring-up surface is intentionally tiny:

```python
import numpy as np
import bitsy as bt                    # no PyTorch dependency at runtime

a = bt.Tensor.from_numpy(np.array([1., 2., 3.], dtype=np.float32))
b = bt.Tensor.from_numpy(np.array([4., 5., 6.], dtype=np.float32))
print(bt.add(a, b).numpy())            # [5. 7. 9.]
```

The full `bt.tensor` / `bt.nn` / `bt.optim` surface lands in M6/M7.

## Status

**M0 "Anvil" — foundations complete (`v0.2.0-bitsy`, renamed from `v0.1.0-m0`
"Hagane").** Bitsy is not yet usable for training. See
[docs/m0_status.md](docs/m0_status.md) for the evidence,
[ROADMAP.md](ROADMAP.md) for the milestone plan, and
[CHANGELOG.md](CHANGELOG.md) for released changes. The earlier `bitsy` repo on
this account was preserved as `bitsy-legacy-2023`; nothing was deleted.

## Two goals

Everything in the project serves two measurable goals:

1. **Numeric parity.** For any supported operation, module or training step,
   Bitsy's outputs match PyTorch's within the tolerances in
   `tests/parity/tolerances.py`. Every claim is verified by running both.
2. **Performance parity.** Bitsy's Vulkan backend matches PyTorch's CUDA
   backend on the *same GPU on the same machine*. Results are reported honestly
   in [BENCHMARKS.md](BENCHMARKS.md), including misses.

## Design at a glance

- C11 core, no C++ in the core library.
- Refcounted **storage** separated from **tensor** views (storage, offset,
  sizes, strides, dtype, device, `requires_grad`).
- A **dispatcher** keyed by `(op, backend, dtype)` so backends can override
  individual operators.
- Backends: `cpu_ref` (naive, independent correctness oracle) and `vulkan`
  (the performance target).
- Dynamic **autograd** graph with per-op backward functions.
- Python package `bitsy`, conventionally imported as `bt`, that mirrors
  PyTorch's API and does **not** depend on PyTorch at runtime.

## Repository map

| Path | Contents |
|---|---|
| `include/bitsy/` | Public C headers |
| `src/core/` | Storage, tensor, dtype, shape/stride math, dispatcher |
| `src/autograd/` | Graph, engine, backward formulas |
| `src/ops/` | Backend-independent op definitions |
| `src/backends/cpu_ref/` | Naive reference kernels |
| `src/backends/vulkan/` | Device, allocator, streams, pipelines, shaders |
| `python/bitsy/` | Python package |
| `python/ext/` | CPython C extension binding |
| `tests/unit/` | C unit tests |
| `tests/parity/` | Bitsy-vs-PyTorch comparison harness |
| `tests/golden/` | Small reference data and deterministic generators |
| `benchmarks/` | Operators, models, and committed results |
| `experiments/` | Hypothesis-first tuning experiments (`exp_NNN`) |
| `tools/` | Build and development helpers |

## Building

Requirements: CMake 3.20+, Ninja, a C11 compiler, and (for Vulkan) the Vulkan
SDK 1.3+ with `glslc`.

Windows, from a Visual Studio developer prompt:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBT_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Linux:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBT_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Vulkan is opt-in during the early milestones (`-DBT_ENABLE_VULKAN=ON`).

## Parity testing

The harness needs PyTorch installed (CPU builds are sufficient; CUDA is used
when present). It is configured with dataclasses, not command-line flags:

```powershell
$env:PYTHONPATH = "python"
python tests/parity/run.py
```

It writes a machine-readable report to `tests/parity/reports/latest.json`.

## Code style

Dense, small and direct, following the style of tinygrad / geohot: 2-space
indentation in C and Python, one-line functions where they read cleanly,
section banners, and short comments that explain **why** a line exists. The
rules and the tinygrad sources they came from are recorded in
[ARCHITECTURE_LOG.md](ARCHITECTURE_LOG.md) (2026-11-21 entry).

## Conventions and decisions

- [docs/FEATURE_INVENTORY.md](docs/FEATURE_INVENTORY.md) — full PyTorch-grade API surface
- [docs/PROGRESS.md](docs/PROGRESS.md) — implemented functions, numeric differences, measured ratios vs targets
- [ARCHITECTURE_LOG.md](ARCHITECTURE_LOG.md) — locked architectural decisions
- [DEVIATIONS.md](DEVIATIONS.md) — intentional differences from PyTorch
- [BENCHMARKS.md](BENCHMARKS.md) — performance results and methodology
- [NOTICE](NOTICE) — attribution to the PyTorch project

## License

BSD-3-Clause. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
