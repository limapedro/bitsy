# Deviations from PyTorch

This file lists every place where Bitsy intentionally or unavoidably differs
from PyTorch. Temporary deviations are marked with the milestone that removes
them. Anything not listed here is intended to match PyTorch's semantics.

## Temporary (bring-up)

- **Python binding (M0–M5, removed in M6).** Early parity work calls the C core
  through a thin `ctypes` shim (`bitsy._ctypes`) rather than a CPython C
  extension. The public `bt.Tensor` surface arrives in M6. This shim is test
  tooling, not the final API. See `ARCHITECTURE_LOG.md`, 2026-11-21.
- **Limited dtype/device surface (M0).** The bring-up API accepts float32
  tensors on `cpu_ref` only. The dtype and device enums exist in the C core and
  the full surface arrives in M1/M2. Unsupported combinations return
  `BT_ERR_NOT_IMPLEMENTED` rather than producing wrong answers.

## Permanent / hardware-driven

- **No C++ in the core.** PyTorch's ATen is C++; Bitsy mirrors its design in
  C11. This is an implementation difference, not an API difference.
- **BF16 has no native hardware path on the target GPU.** BF16 is stored as
  packed `uint16` with explicit round-to-nearest-even conversion.
- **No Tensor Cores.** The GTX 1660 Super has no matrix hardware, so both
  Bitsy and PyTorch's CUDA baseline use regular CUDA cores. Performance
  results must not be extrapolated to tensor-core GPUs.
- **Vulkan 1.2 feature baseline.** Headers are Vulkan 1.3, but only 1.2 features
  exposed by Turing are required. Optional extensions are queried at runtime
  and never assumed.

## Known semantic differences

None recorded yet. As parity work uncovers cases where bit-exact reproduction
is infeasible, they are documented here with the numeric justification and a
link to the relevant architecture-log entry.
