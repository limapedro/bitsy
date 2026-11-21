# exp_020 — gradient-flow parity gate and safe broadcast elimination

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 2 (profiling/gates) + Phase 5/6

## Prerequisite

exp_019 proved the 42-case isolated-op harness is insufficient: zero-dispatch
broadcast views passed 42/42 but broke end-to-end training (MNIST loss 1.87 vs
PyTorch 1.33, W1 diff 0.149). The optimisation was reverted, but the `ones *
grad` materialisations remain and cost dispatches and bandwidth.

## Hypothesis

Two deliverables: (1) add a **required end-to-end gradient-flow parity gate**
to the test suite so autograd/view changes cannot regress silently; (2) with
that gate in place, find the exact consumer that misreads a stride-0 broadcast
view and either make it safe or replace the broadcast with a dedicated
full-shape contiguous gradient kernel that keeps parity.

## Predictions

- The new gate fails on the known-bad broadcast view (verified by temporarily
  reintroducing it) and passes on the current tree. It is wired into CI.
- A safe broadcast elimination removes ≥3 dispatches per step
  (43 → ≤40), captured step ≤0.225 ms (from ~0.240) and eager ≤1.60 ms
  (from ~1.67), with the gradient gate and all existing parity green.
- If the stride-0 view cannot be made safe, the dedicated full-shape gradient
  kernel keeps dispatch count unchanged (43) and is retained only if it is
  measurably faster; negatives recorded either way.

## Measurement

1. Snapshot current graph/eager/elementwise/GEMM JSON and training reports.
2. Add `tests/parity/grad_flow.py` (dataclass config, no argparse) comparing a
   one-step MLP update in Bitsy vs PyTorch CUDA on the same batch (loss and
   every parameter gradient/after-step parameter within 1e-5), and wire it into
   `tests/parity/run.py` + CI.
3. Diagnose the first divergent op between broadcast and materialised paths
   with an instrumented run; fix or reject the broadcast view accordingly.
4. Re-run graph/eager/elementwise/GEMM benchmarks, parity, MNIST/CIFAR, CTest.
5. FINDINGS with all numbers including negatives. No tolerance relaxed.

## Honesty

Numeric parity is a hard gate: any change that fails the new gradient gate is
reverted regardless of speed. No result is claimed from the previously broken
path. All public reports state missed targets plainly.
