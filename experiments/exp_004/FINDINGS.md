# exp_004 — Findings: async timeline submission and pointwise fusion

**Date:** 2026-11-21
**Decision:** keep the fused kernel; keep timeline submission but record the
async prediction as **not met**.

## What was tried

1. `bt_vulkan_flush()` now submits with a timeline semaphore signal and no
   fence wait. New `bt_vulkan_wait()` waits for the last submitted value,
   collects timestamp results, and recycles descriptor/query slots. Upload and
   readback wait internally where the host needs the data.
2. Added a fused `bt_add_relu` kernel (`relu(a+b)` in one dispatch) with a
   dedicated GLSL shader and CPU reference, registered on both backends.

## Measurements

### Async chain (20 adds of 1M elements, explicit wait at end)

| submission | before ms | after ms | change |
|---|---:|---:|---:|
| batched + fence wait (exp_003) | 1.277 | — | — |
| batched + timeline, explicit wait | — | 1.251 | 2% faster |

**Prediction not met** (expected ≥1.2x). The chain already had exactly one
submission before this change, so removing the fence wait can only save the
CPU's wait at the end of the chain; the wall time is dominated by 0.94 ms of
GPU work plus the explicit completion wait. Async submission is therefore kept
for its API benefit (CPU can submit before waiting) but is **not** a speedup for
single-batch chains.

### Fusion (20 add+relu pairs of 1M elements)

| path | wall ms | device ms |
|---|---:|---:|
| separate `add` then `relu` | 2.365 | 1.915 |
| fused `add_relu` | 1.200 | 0.935 |

**1.97x faster**, meeting the ≥1.3x prediction. Both paths call `flush()` +
`wait()` once, so the comparison is fair. The fused path halves both dispatch
count and GPU work time.

### Single-op and GEMM ratios (with explicit completion wait)

| case | before ratio | after ratio |
|---|---:|---:|
| add 4096 | 0.31 | 0.53 |
| mul 4096 | 0.28 | 0.29 |
| relu 4096 | 0.31 | 0.35 |
| add 1M | 0.49 | 0.36 |
| mul 1M | 0.45 | 0.36 |
| relu 1M | 0.39 | 0.23 |
| GEMM tiled 1024² | 0.23 | 0.22 |

Ratio movement here is dominated by PyTorch's run-to-run variance (e.g. PyTorch
add 1M was 0.0868 ms in exp_003 and 0.0556 ms now). Bitsy wall times were
roughly flat to slightly better. **No performance target is met.**

### Correctness

CTest 1/1; parity 42/42 (new fused cases included, exact mode, 0 mismatches on
both backends); gradcheck loss 1.19e-07, W grad 5.96e-08; conv gradients ≤7.6e-6.

## Interpretation

Fusion works and is the clearest win in this experiment. Async submission did
not pay off for a workload that was already one batch; its benefit should show
up when the CPU has several independent submissions in flight, which the
current eager autograd does not create. The next step is to make training steps
submit several batches before waiting, and to fuse more common patterns
(bias+activation, softmax+mask). Raw JSON snapshots are in this folder.
