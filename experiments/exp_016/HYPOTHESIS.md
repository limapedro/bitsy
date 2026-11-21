# exp_016 — device-side unbroadcast reductions for single-chunk capture

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 6 (reductions) + Phase 5 (submission)

## Prerequisite

exp_015 made full-step capture work by splitting the capture into command-buffer
chunks at the autograd's CPU-gradient readbacks. The MLP step needs exactly one
such readback (`_unbroadcast` of a `(64,10)` bias gradient to `(64,1)`), so every
replay submits **two** chunks (~36 µs of extra submit+fence) and captured
throughput is only 0.19–0.23× PyTorch CUDA graph throughput.

## Hypothesis

The 2D unbroadcast reduction can be done entirely on device using existing
matmul/view ops (multiply by a ones vector), eliminating the host roundtrip.
The whole MLP step then captures into a **single** command buffer and replays
with one submit, removing the extra submit and the chunk-splitting machinery
from the measured path.

## Predictions

- Captured step wall time improves **≥1.3×** (0.294–0.355 → ≤0.27 ms/step) and
  graph throughput ratio improves from 0.19–0.23 to **≥0.28**.
- The capture produces exactly **one** chunk for the MLP step (verified by the
  profiler submit count or a capture-chunk counter).
- Numerics stay within tolerance: 42/42 parity cases, MNIST/CIFAR loss curves
  matching within rtol/atol 1e-5. The reduction order changes fp32 rounding
  versus NumPy's pairwise sum, so exact equality is **not** expected — any
  drift must be measured and reported, not hidden.
- Higher-dimensional (>2D) unbroadcasts keep the existing CPU fallback; this
  experiment targets the 2D MLP case only and says so.

## Measurement

1. Snapshot current graph/elementwise/GEMM JSON and training reports into
   `experiments/exp_016/`.
2. Replace `_unbroadcast`'s CPU roundtrip with a device reduction for ≤2D
   tensors, keeping the old path for >2D. No kernel or tolerance change.
3. Measure the numeric difference of the new reduction against NumPy on the
   real `(64,10) -> (64,1)` gradient, and re-run graph/elementwise/GEMM
   benchmarks, `tests/parity/run.py`, MNIST/CIFAR parity, CTest Vulkan+CPU.
4. If the ≥1.3× prediction is met, keep the device path; otherwise record the
   negative and classify the remaining cost. Write FINDINGS with all numbers.

## Honesty

No tolerance is relaxed. If fp32 drift breaks parity, revert the device path,
keep the chunked capture, and record the attempt as a negative result. All
public reports must state the measured target status plainly.
