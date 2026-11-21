# Hagane (鋼)

Hagane is a deep learning framework written from scratch in **C11** with a
**Vulkan compute** backend and a Python front end that mirrors PyTorch.
The goal is that PyTorch code moves to Hagane with minimal edits:

```python
import hagane as hg          # instead of `import torch`

x = hg.tensor([1.0, 2.0, 3.0])
y = hg.tensor([4.0, 5.0, 6.0])
print(x + y)                 # tensor([5., 7., 9.])
```

## Status

**M0 "Anvil" — foundations (in progress).** Hagane is not yet usable for
training. See [ROADMAP.md](ROADMAP.md) for the milestone plan and current
state, and [CHANGELOG.md](CHANGELOG.md) for released changes.

## Two goals

Everything in the project serves two measurable goals:

1. **Numeric parity.** For any supported operation, module or training step,
   Hagane's outputs match PyTorch's within the tolerances in
   `tests/parity/tolerances.py`. Every claim is verified by running both.
2. **Performance parity.** Hagane's Vulkan backend matches PyTorch's CUDA
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
- Python package `hagane`, conventionally imported as `hg`, that mirrors
  PyTorch's API and does **not** depend on PyTorch at runtime.

## Repository map

| Path | Contents |
|---|---|
| `include/hagane/` | Public C headers |
| `src/core/` | Storage, tensor, dtype, shape/stride math, dispatcher |
| `src/autograd/` | Graph, engine, backward formulas |
| `src/ops/` | Backend-independent op definitions |
| `src/backends/cpu_ref/` | Naive reference kernels |
| `src/backends/vulkan/` | Device, allocator, streams, pipelines, shaders |
| `python/hagane/` | Python package |
| `python/ext/` | CPython C extension binding |
| `tests/unit/` | C unit tests |
| `tests/parity/` | Hagane-vs-PyTorch comparison harness |
| `tests/golden/` | Small reference data and deterministic generators |
| `benchmarks/` | Operators, models, and committed results |
| `experiments/` | Hypothesis-first tuning experiments (`exp_NNN`) |
| `tools/` | Build and development helpers |

## Building

Requirements: CMake 3.20+, Ninja, a C11 compiler, and (for Vulkan) the Vulkan
SDK 1.3+ with `glslc`.

Windows, from a Visual Studio developer prompt:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DHG_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Linux:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DHG_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Vulkan is opt-in during the early milestones (`-DHG_ENABLE_VULKAN=ON`).

## Parity testing

The harness needs PyTorch installed (CPU builds are sufficient; CUDA is used
when present). It is configured with dataclasses, not command-line flags:

```powershell
$env:PYTHONPATH = "python"
python tests/parity/run.py
```

It writes a machine-readable report to `tests/parity/reports/latest.json`.

## Conventions and decisions

- [ARCHITECTURE_LOG.md](ARCHITECTURE_LOG.md) — locked architectural decisions
- [DEVIATIONS.md](DEVIATIONS.md) — intentional differences from PyTorch
- [BENCHMARKS.md](BENCHMARKS.md) — performance results and methodology
- [NOTICE](NOTICE) — attribution to the PyTorch project

## License

BSD-3-Clause. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
