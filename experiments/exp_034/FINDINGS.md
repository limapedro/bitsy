# exp_034 — training-scale gradient parity gate in CI

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 2 (gates). Locked by the exp_033 consequence in
ARCHITECTURE_LOG.md.

## Implementation

- Added `tests/parity/train_scale.py` (dataclass config, no argparse): one real
  MLP step at the training configuration (batch 64, 784→64→10, lr 0.1, same
  init as the MNIST harness) comparing loss, all four parameter gradients and
  post-step parameters between Bitsy and PyTorch CUDA/CPU at rtol 1e-4/
  atol 1e-5.
- Wired into `tests/parity/run.py` as a required case — total **44/44** — and
  into both CI jobs via the existing `run.py` invocation.

## Bug found by the new gate

The new loss check initially FAILED: Bitsy CE loss 0.9907 vs PyTorch 2.4377.
Root cause: `ce_loss.comp` had only 256 threads and summed just the first 256
of 640 (batch 64 × 10 classes) `onehot*log_softmax` products, dividing by 64.
Numerical check confirmed `-(onehot*lsm).ravel()[:256].sum()/64 == 0.99067`.
Fixed by looping `for (k = tid; k < n; k += 256)` in the shader. After the fix
all nine train-scale checks pass; MNIST/CIFAR loss curves and W1 diffs are
unchanged (3.278e-07 / 7.302e-07). This was a genuine numeric-parity bug that
the previous gates did not cover.

## Gate sensitivity

Injected a wrong gradient (`w1` grad scaled to 0) into the Bitsy step:
`passed=False`, with exactly `grad_w1` failing and the loss still passing — the
same signature as exp_019/exp_033. Restored tree passes 44/44.

## Predictions

- "gate passes on clean tree": **met** (after fixing the discovered CE bug).
- "gate fails on injected bad gradient": **met**.
- "total 44/44, CI green": **met** locally; CI verified by PR.
- "no kernel change": **not met** — the CE loss shader needed a correctness
  fix that the new gate exposed; recorded above.

## Performance (unchanged within variance)

Captured step 0.1443 ms (ratio 0.552 @1590 MHz; ratio varies with PyTorch
graph timing), eager step 0.892 ms (ratio 1.165), GEMM/elementwise unchanged.
No speed claim is made by this experiment.

## Interpretation / next

The training-scale gate now closes the coverage gap that allowed two parity
failures through. Further step-level fusion can proceed under the stronger
gate; all prior negatives remain valid. Open performance gaps: captured graph
~0.46–0.55, standalone GEMM ~0.68, eager elementwise 0.39–0.52.

Raw data in `experiments/exp_034/{train_scale_PASS.json,after_*}`.
