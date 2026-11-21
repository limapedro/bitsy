# exp_015 — full-step capture and replay (CUDA-graph equivalent)

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 5 (submission / per-op overhead)

## Prerequisite

exp_013 showed a captured 20-add chain matches PyTorch CUDA graph replay at
**1.00×**, but that required an explicit pointwise capture. exp_014 eliminated
transfer sync and made deferred uploads the default, cutting the training step
9.575 → 3.220 ms/step (2.97×) yet leaving end-to-end throughput at only
0.34–0.37 of PyTorch. The remaining ~3.1 ms/step is Python/ctypes dispatch and
autograd graph construction over many small Vulkan dispatches. No experiment
has yet captured a *whole training step* (forward + backward + optimizer).

## Hypothesis

The whole-step dispatch sequence is static across steps. Capturing forward +
backward + optimizer once into the existing reusable Vulkan command buffer,
holding the input/parameter/intermediate buffers alive for the replay window,
and replaying it N times with one host wait removes both Python dispatch cost
and per-submit overhead. That should bring end-to-end throughput close to the
1.00× already demonstrated for captured pointwise chains.

## Predictions

- Bitsy captured-step throughput is **≥0.80×** PyTorch CUDA graph throughput
  at matched clock (the campaign's end-to-end target is ≥0.85; 0.80 is the
  intermediate gate).
- Captured step is **≥2× faster** than the current default eager Bitsy step
  (~3.22 ms/step).
- Capturing produces the same numeric result as eager execution for a single
  step (loss/weights within tolerance), and the 42-case parity harness stays
  green.
- Input refresh works by copying a new batch into the persistent device buffers
  used at capture time; if no such API exists and adding it is required, that
  is recorded explicitly in FINDINGS.

## Measurement

1. Snapshot current elementwise/GEMM/training JSON into `experiments/exp_015/`.
2. Add `benchmarks/models/graph_step_bench.py` (dataclass config, no argparse):
   MLP step, PyTorch CUDA graph (eager warmup on side stream, capture, N replays,
   one sync) versus Bitsy `bt.capture_begin()`/`capture_end()`/`replay(N)` with
   persistent input/parameter/intermediate buffers kept alive. Median+spread,
   GPU clock, separate processes where needed.
3. Verify captured-step numerics against an eager step on one batch.
4. Re-run `benchmarks/ops/elementwise.py`, `gemm.py`, `tests/parity/run.py`,
   CTest Vulkan+CPU-only.
5. Write FINDINGS including negatives and lifetime/API limitations. Never
   change a tolerance or kernel algorithm in this experiment.

## Honesty

If capture cannot be made correct for a full training step (buffer lifetimes,
readback coupling, or allocator interactions), record that as the result and
classify the remaining gap. All reports must state missed targets plainly and
link the raw evidence. Do not claim parity from a benchmark that replays stale
inputs without saying so.
