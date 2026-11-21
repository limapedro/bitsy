# exp_016 — Findings: device-side unbroadcast reductions

**Date:** 2026-11-21
**Scope:** Phase 6 reductions + Phase 5 submission. No kernel, tolerance or
benchmark-method change. Parity: 42/42, CTest 1/1 Vulkan and CPU-only.

## Implementation

- `_unbroadcast()` (autograd.py) now sums 2D gradients with device `matmul`
  against a ones vector plus `view`, instead of a `g.numpy()` CPU roundtrip.
  The existing CPU path is kept for >2D gradients (conv) and unsupported axis
  patterns, and is documented as such.
- Numeric check on the real `(64,10) -> (64,1)` bias gradient: max absolute
  difference from NumPy's pairwise sum is **9.54e-7** (max relative **1.07e-7**),
  well inside the campaign's 1e-5 tolerance. This is a fp32 reduction-order
  change, not bit-identical to the old CPU path.
- Capture-phase profiler: submit count dropped from **2 to 1** (the remaining
  submit is `bt_vk_capture_begin()` flushing pre-existing uploads), confirming
  the MLP step now captures into a **single command-buffer chunk** with no
  mid-backward host sync.

## Measurements

| path | before | after | change |
|---|---:|---:|---:|
| Eager Bitsy step (1530 MHz) | 3.220 ms | **2.626–2.911 ms** | 1.11–1.23× faster |
| Eager throughput ratio vs PyTorch | 0.34–0.37 | **0.37–0.41** | improved |
| Captured Bitsy step | 0.294–0.355 ms | 0.296–0.334 ms | **no improvement** |
| Captured graph ratio vs PyTorch CUDA graph | 0.19–0.23 | 0.20–0.23 | **no improvement** |

Elementwise 0.41–0.59 and GEMM 1024² 0.75 at 1530 MHz (unchanged within noise).
MNIST report numerically identical to the committed one; CIFAR final conv1 W
diff 6.855e-07 (4D gradients still use the CPU fallback).

## Predictions

- "Captured step ≥1.3× faster, ratio ≥0.28": **not met** (no measurable
  change). Removing one `vkQueueSubmit` per replay saves only a few µs because
  the final wait is shared; the extra device reduction dispatches offset it.
- "Single-chunk capture": **met** (submit count 2 → 1 at capture).
- "Parity within tolerance": **met** (42/42; MNIST identical; CIFAR unchanged;
  reduction diff 1.07e-7 relative).
- ">2D keeps CPU fallback": **met** by construction; CIFAR exercise confirms.

## Interpretation / next

The device reduction is a **mixed result**: it removes a host sync from every
backward and modestly speeds up the default eager step (1.1–1.2×, ratio
0.37–0.41), and it enables single-chunk capture, but it does **not** improve
captured graph throughput. The reason the graph path is still ~0.20× PyTorch is
now clear: ~66 small Vulkan dispatches per step (plus their barriers) dominate;
saving one submit is negligible against that. The next work should attack that
directly — fuse the backward/optimizer op chains in the interim autograd (or
record a fused step kernel) so the captured graph has far fewer dispatches,
then re-measure. Any such change needs its own `exp_NNN` and full parity
re-verification.

Raw data in `experiments/exp_016/{before_*,after_*}`.
