# exp_029 — fused four-parameter SGD update

**Date:** 2026-11-21
**Owner:** TOGISHI Phase 5 (fusion) + Phase 6 (kernels)

## Prerequisite

After exp_028 the step has 22 dispatches, of which 4 are separate `sgd_step`
updates (one per Linear parameter). The campaign lists multi-tensor optimizer
fusion as a target.

## Hypothesis

A single kernel that updates all four parameters (workgroup.y selects the
parameter, workgroup.x covers its elements) replaces four dispatches with one
without changing the arithmetic `W - lr*grad`.

## Predictions

- Dispatches/step drop by **≥3** (22 → ≤19).
- Captured step improves **≥1.02×** (0.156 → ≤0.153 ms); eager does not regress.
- 43/43 parity incl. the gradient gate; updated parameters within 1e-5.
- CPU fallback and >4-parameter models keep the old path.

## Measurement

1. Snapshot current graph/eager JSON.
2. Add `sgd_fused.comp` (12 storage bindings + a 4-count PC array), dedicated
   descriptor layout/pipeline, `bt_sgd_step_fused()` with a CPU fallback, and
   use it in `Optimizer.step` for ≤4 parameters.
3. Run `tests/parity/run.py` (43 cases) FIRST; only if green continue.
4. Re-profile, re-run graph/eager/elementwise/GEMM, MNIST/CIFAR, CTest both.
5. FINDINGS with all numbers including negatives. No tolerance change.

## Honesty

The 43-case gate is hard; any regression is reverted. Reports state missed
targets plainly and report fp32 drift rather than hiding it.
