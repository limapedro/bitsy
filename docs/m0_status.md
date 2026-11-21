# M0 "Anvil" — Status Report

**Date:** 2026-11-21
**Release:** `v0.1.0-m0`

## What passes

| Check | Evidence |
|---|---|
| C11 core builds with MSVC + Ninja | `cmake --build build` → `Linking C shared library hagane.dll`, no warnings |
| C unit tests | `ctest --test-dir build --output-on-failure` → `1/1 Test #1: unit_core ... Passed` |
| Python package imports | `import hagane` → version `0.1.0-m0` |
| `add` on `cpu_ref` vs PyTorch CPU and CUDA | `python tests/parity/run.py` → **13/13 parity cases passed** |
| CI on Windows (MSVC) | GitHub Actions PR #5: `Windows (MSVC + Ninja) pass 1m7s` |
| CI on Linux (gcc, software Vulkan available) | GitHub Actions PR #5: `Linux (gcc + Ninja, software Vulkan available) pass 27s` |
| Vulkan validation layers | Not applicable yet: no Vulkan code exists in M0 |

Parity report environment (from `tests/parity/reports/latest.json`):

- Windows-10-10.0.26300-SP0, Python 3.10.11
- torch 2.11.0+cu126, CUDA 12.6, NVIDIA GeForce GTX 1660 SUPER
- hagane 0.1.0-m0

Cases covered for `add`: scalar+scalar, vector, special values (inf, -inf,
NaN, ±0), extreme magnitudes, row broadcasting, two-axis broadcasting with a
size-1 dimension, scalar RHS, size-1 expansion, empty tensors (0,2),
non-power-of-two (7,13), 4096-element vectors, broadcast special values, and a
shape-mismatch error case. The comparison is bit-exact on float32 bit patterns
(NaN payloads excluded) for Hagane vs PyTorch CPU, PyTorch CUDA vs PyTorch CPU,
and Hagane vs PyTorch CUDA.

## What fails

Nothing in M0 scope. Out of scope and deliberately not claimed:

- Vulkan backend: starts in M2.
- Full `Tensor`/`nn`/`optim` Python surface: starts in M6/M7.
- Dtypes other than float32 on `cpu_ref`: start in M1.
- Autograd, GEMM, convolution, RNG parity: M3–M5.

## Performance

No benchmarks were run: M0 has no Vulkan kernels. The full target table and
honesty rules are in `BENCHMARKS.md`. No §7.2 ratio is claimed.

## Known deviations

See `DEVIATIONS.md`. The only active ones are the transitional ctypes shim and
the M0 float32/cpu_ref-only bring-up surface.

## Next steps (M1 "Ingot")

1. Extend the tensor core with views (`view`, `reshape`, `transpose`,
   `permute`, `expand`, `squeeze`, `narrow`, slicing) and version counters.
2. Implement creation ops, the full elementwise set, and reductions on
   `cpu_ref`, each with a declarative parity entry.
3. Cross-check every new op against PyTorch CPU before adding any Vulkan code.
