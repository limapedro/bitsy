# exp_034 — training-scale gradient parity gate in CI

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 2 (profiling gates)
**Locked by:** ARCHITECTURE_LOG.md exp_033 consequence

## Prerequisite

exp_019 (broadcast views) and exp_033 (fused CE backward) both passed the
43-case gate — including its small-config gradient-flow case (batch 8,
32→16→8) — while end-to-end training diverged (MNIST W1 diff 0.149 / 0.70).
The gap is gate coverage, not the checks themselves.

## Hypothesis

A required parity case using the **training** configuration (batch 64,
784→64→10, same init and synthetic batch) that compares loss, all four
parameter gradients and post-step parameters against PyTorch CUDA will catch
both known-bad changes. Adding it to `run.py` makes it a CI gate.

## Predictions

- The new gate PASSES on the current (exp_033-reverted) tree.
- It FAILS on a deliberately corrupted gradient path at training scale.
- Total parity cases become 44/44 and both CI jobs remain green.
- No kernel/autograd change is made; performance numbers are unchanged.

## Measurement

1. Add `tests/parity/train_scale.py` (dataclass config, no argparse) with the
   training shapes, comparing loss, grads and post-step params within 1e-5.
2. Wire it into `tests/parity/run.py` (44 cases) and CI (run.py already invoked).
3. Verify on the clean tree and with an injected fault (gradient scaled to 0).
4. FINDINGS with all numbers; no performance claim from this experiment.

## Honesty

The gate must not merely pass — its sensitivity to a known-bad gradient is
demonstrated. Any gate that is too loose to catch exp_033's failure is a miss.
