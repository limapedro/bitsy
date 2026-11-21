# exp_013 — Findings: record-once command buffers and timeline replay

**Date:** 2026-11-21
**Scope:** Phase 5 submission overhead. No kernel or tolerance changes.
Parity: 42/42, CTest 1/1 Vulkan and CPU-only.

## Implementation

- Added `bt_vulkan_capture_begin/end/replay` (Python: `bt.capture_begin()`,
  `capture_end()`, `replay(n)`). A dedicated primary command buffer is recorded
  once with `SIMULTANEOUS_USE`, then submitted `n` times with one timeline
  signal per submission; `bt_vulkan_wait()` is the single host completion point.
- During capture, `bt_vk_flush()` is a no-op so the captured buffer owns all
  recording. Input tensors must stay alive until the wait completes (documented
  in the header).
- A correctness check replaying a captured 5-add chain 10 times produced
  exactly the same bits as re-recording it (max diff 0).

## Replay benchmark (20 adds × 1M elements, 1860 MHz)

Both Bitsy replay and PyTorch graph were measured in the same process/run (GPU
clock 1860 MHz at the final sample), so the replay/graph ratio is a fair
same-clock comparison. The ordinary elementwise/GEMM tables below were measured
separately at 1530 MHz.

| path | ms per chain |
|---|---:|
| Bitsy re-record each iteration | 1.080 |
| Bitsy capture + replay | **0.899** (1.20× faster) |
| PyTorch eager | 0.913 |
| PyTorch CUDA graph replay | 0.900 |

**Bitsy replay / PyTorch CUDA graph = 1.00×** — for a record-once pointwise
chain, captured execution matches PyTorch CUDA exactly. This is the first
measured performance-parity result in the campaign, but it applies only to the
captured/graph execution path, not the ordinary eager API.

## Predictions

- "Replay ≥1.5× faster than re-recording": **not met** (1.20×). The remaining
  time is actual GPU work (~0.86 ms per chain from exp_012), so a larger
  enqueue-only speedup was not physically available.
- "Replay throughput vs PyTorch CUDA graph ≥0.80": **met** (1.00×).
- "Parity stays green": **met**.
- "Captured kernels identical bits": **met**.

Regular elementwise/GEMM ratios are unchanged within run-to-run variance
(elementwise 0.40–0.64, GEMM 0.71 at 1530 MHz) because those benchmarks use the
ordinary eager API. Raw snapshots in `experiments/exp_013/`.

## Interpretation / next

The per-op submit/fence overhead has been *characterized* and a captured path
that matches PyTorch is available, but the default API still pays it. The next
step is to make deferred submission the default behaviour for chains (auto-capture
or stream-like semantics) without changing numeric results, and to measure its
effect on end-to-end training steps. That needs its own `exp_NNN`.
