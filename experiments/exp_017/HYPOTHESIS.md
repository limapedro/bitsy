# exp_017 — fuse backward and optimizer op chains

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 5 (fusion) + Phase 6 (kernels)

## Prerequisite

exp_016 removed the backward CPU roundtrip and achieved a single-chunk capture,
but the profiler showed **~66 Vulkan dispatches and 66 barriers per MLP step**.
Captured throughput remains 0.20–0.23× PyTorch CUDA graph throughput and eager
end-to-end 0.37–0.41× PyTorch. Submit count is no longer the bottleneck.

## Hypothesis

A large fraction of those 66 dispatches are short elementwise chains produced
by the interim autograd (mul-backward, tanh-backward, add-accumulation) and the
SGD optimizer (`W - lr*grad` is two dispatches per parameter). Replacing the
hot patterns with fused kernels that do several elementwise operations in one
dispatch will cut dispatch/barrier count substantially without changing the
math.

## Predictions

- Dispatches per MLP step drop by **≥30%** (66 → ≤46), measured by profiler.
- Captured step wall time improves **≥1.3×** (0.296–0.334 → ≤0.25 ms/step) and
  graph throughput ratio improves from 0.20–0.23 to **≥0.28**.
- Eager step improves **≥1.15×** (2.63–2.91 → ≤2.29 ms/step).
- Numeric parity holds: 42/42 cases, MNIST/CIFAR loss curves within tolerance;
  fused kernels must reproduce the existing fp32 operation order (or the drift
  must be measured and reported).

## Measurement

1. Snapshot current graph/eager/elementwise/GEMM JSON and training reports into
   `experiments/exp_017/`.
2. Profile one step with `bt.profile_dump` and count dispatch names to find the
   exact fusion targets (this is measurement, not a code change).
3. Implement the highest-value fusions with dedicated shaders/C ops where the
   existing `add_relu` fused kernel is insufficient; keep unfused fallbacks for
   shapes/patterns that do not match.
4. Re-profile dispatch counts, re-run graph/eager/elementwise/GEMM benchmarks,
   `tests/parity/run.py`, MNIST/CIFAR parity, CTest Vulkan+CPU.
5. FINDINGS with all numbers including negatives. No tolerance changes.

## Honesty

If fusion does not improve throughput, keep it only if it is neutral and
correct, record the negative and classify the remaining cost. Do not claim
parity from a benchmark that skips the fused path. All public reports must state
missed targets plainly.
