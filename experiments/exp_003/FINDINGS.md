# exp_003 — Findings: batched submissions and timestamp queries

**Date:** 2026-11-21
**Decision:** keep batching as the default; keep the eager toggle for comparison.

## What was tried

Vulkan work is no longer submitted per op. Dispatches are recorded into one
command buffer with a rotating descriptor set per dispatch, and submitted only
when the host reads data (readback) or calls the new `bt_vulkan_flush()` API.
GPU timestamp queries (`vkCmdWriteTimestamp`, enabled when the device limit
`timestampComputeAndGraphics` is true) measure each dispatch; the last batch's
device time is available as `bt_vulkan_device_time_ms()`.

`BT_VK_NO_BATCH=1` restores submit-per-op for A/B measurement.

## Measurements

### 20-op pointwise chain (`benchmarks/ops/batch_chain.py`, 1M elements)

| mode | wall ms | reported device ms |
|---|---:|---:|
| eager (one submit per op) | 2.773 | 0.048 (last op only) |
| batched (one submit) | 1.277 | 0.942 (sum of 20 kernels) |

**2.17x faster wall time**, meeting the < 5 ms prediction. In batched mode the
timestamp queries sum all 20 kernels (0.942 ms), leaving ~0.34 ms of the
1.277 ms wall time as submit/fence/host overhead; in eager mode the ~0.94 ms of
actual GPU work is spread across 20 submits and the wall time is 2.773 ms.

### Single ops, now with fair sync and device time

| op | n | before wall | after wall | device time | ratio before → after |
|---|---:|---:|---:|---:|---:|
| add | 4096 | 0.0772 | 0.0783 | 0.0020 | 0.17 → 0.31 |
| mul | 4096 | 0.0703 | 0.1107 | 0.0021 | 0.35 → 0.28 |
| relu | 4096 | 0.0942 | 0.0862 | 0.0021 | 0.18 → 0.31 |
| add | 1M | 0.1551 | 0.1784 | 0.0483 | 0.36 → 0.49 |
| mul | 1M | 0.1514 | 0.1963 | 0.0518 | 0.35 → 0.45 |
| relu | 1M | 0.1616 | 0.1999 | 0.0492 | 0.26 → 0.39 |
| GEMM tiled | 1024² | 4.337 | 4.824 | 4.443 | 0.22 → 0.23 |

Single-op wall time did **not** improve: one op still means one submit either
way, and the explicit flush call adds a tiny host cost. The ratio movement is
dominated by run-to-run variance in PyTorch's small-op timings (e.g. PyTorch
add 1M was 0.0553 ms before and 0.0868 ms now). Device time now quantifies the
gap: for add 1M the kernel takes 0.048 ms while wall time is 0.178 ms.

### Correctness

CTest 1/1; parity 40/40; gradcheck loss 1.19e-07, W grad 5.96e-08; conv
forward ≤ 4.8e-7 and gradients ≤ 7.6e-6 on both backends. No parity regression.

## Interpretation

The batching hypothesis is supported for multi-op workloads (2.17x), which is
the path to end-to-end speed. The single-op comparison shows why the headline
ratios barely move: the benchmark measures one op per iteration, so one submit
per iteration remains. Timestamp queries are now the honest way to separate
kernel time from launch overhead.

Performance targets remain missed. Next steps, each its own experiment:
asynchronous submissions without per-op fence waits (timeline semaphores),
larger tiles/register blocking for GEMM, and fusing pointwise chains into one
dispatch. Raw before/after JSON snapshots are in this folder.
