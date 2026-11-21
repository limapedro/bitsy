# exp_001 — Findings: caching device buffers

**Date:** 2026-11-21
**Decision:** keep the change.

## What was tried

`bt_vk_alloc` previously called `vkCreateBuffer` + `vkAllocateMemory` on every
tensor allocation (including every elementwise output) and destroyed them on
release. It now recycles device-local buffers from 24 power-of-two size-class
free lists; `bt_vk_free` returns a buffer to its class, and
`bt_vk_allocator_shutdown` destroys the cache at process exit.

## Measurements (same command, same GPU, before/after snapshots in this folder)

| case | before ms | after ms | speedup | before ratio vs CUDA | after ratio |
|---|---:|---:|---:|---:|---:|
| add 4096 | 0.3854 | 0.0772 | 5.0x | 0.07 | 0.17 |
| mul 4096 | 0.4040 | 0.0703 | 5.7x | 0.09 | 0.35 |
| relu 4096 | 0.3941 | 0.0942 | 4.2x | 0.09 | 0.18 |
| add 1M | 0.6753 | 0.1551 | 4.4x | 0.08 | 0.36 |
| mul 1M | 0.7304 | 0.1514 | 4.8x | 0.07 | 0.35 |
| relu 1M | 0.7167 | 0.1616 | 4.4x | 0.06 | 0.26 |

Parity stayed green: CTest 1/1, parity 19/19 (including all Vulkan cases).

## Interpretation

The hypothesis was correct: driver allocation, not shader execution, dominated
the old wall time. The latency predictions were met (4096-element cases
0.07–0.09 ms < 0.15 ms; 1M cases 0.15–0.16 ms < 0.35 ms). The ratio prediction
was only partly met: four of six cases beat 0.25, while add 4096 reached 0.17
and relu 4096 reached 0.18. PyTorch's small-op wall time varies between runs
(0.013–0.025 ms), so ratios for the smallest tensors are noisy; the median and
spread are kept in the JSON snapshots.

Remaining gap versus the §7.2 target (>= 0.95 for bandwidth-bound ops) is real
and unmet. The fixed cost of `vkQueueSubmit` + fence wait is now the floor for
small tensors (~0.06 ms, still 3x PyTorch). Large-tensor bandwidth is ~100 GiB/s
vs PyTorch ~220 GiB/s, about 45% of peak-equivalent. Next steps, each needing
its own experiment: batched submissions / persistent command buffers, then a
true suballocator and timestamp queries to separate device time from launch
overhead. No cherry-picking: the full table is in `BENCHMARKS.md` and the raw
JSON snapshots are here.
