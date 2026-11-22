# exp_040 — FINDINGS (PR #9: next API batch + legacy-core cutover plan)

Status: PR #9 of the v0.2 rewrite. Builds on PR #8 (speed parity reached).

## What landed — 29 new public functions, ZERO new C lines
All new APIs are pure Python compositions over the frozen compact primitive set
(`src/rt` unchanged at 2,026 lines). New surface:

| Group | Functions |
|---|---|
| Activations | `silu`, `softplus`, `mish`, `leaky_relu`, `elu`, `selu`, `hardtanh`, `hardswish` |
| Elementary | `square`, `reciprocal`, `rsqrt`, `exp2`, `log2`, `clamp`/`clip`, `lerp` |
| Reductions | `var`, `std`, `logsumexp`, `norm` (L1/L2) |
| Losses | `l1_loss`, `smooth_l1_loss`, `bce_loss`, `bce_with_logits_loss`, `nll_loss` |
| Logic/compare | `logical_and`, `logical_or`, `logical_not`, `logical_xor`, `Tensor.gt/ge/lt/le/eq/ne/where` |

Stability note: `softplus`, `bce_with_logits` and `logsumexp` use numerically stable
forms (`max(x,0)+log(1+exp(-|x|))`, `max+log(sum(exp(x-max)))`), not naive formulas.

## Evidence — `benchmarks/compact/api_parity.py`
New harness, 44 cases (forward + gradient where differentiable) vs PyTorch CUDA:
```
44/44 bitsy_v2 extra-API parity cases passed
  activations: fwd/grad diffs 0 .. 2.4e-7
  losses: fwd/grad diffs 0 .. 3.7e-9
  reductions: 0 .. 3.8e-5 (norm1, within 1e-4 tol)
  logical/comparisons: exactly 0
```
`rsqrt`/`log2` are compared on positive inputs: the C `sqrt` clamps negatives at 0
(documented exp_037 deviation) while PyTorch returns NaN.

## All gates green (no regression)
op 27/27, gradcheck 25/25, model 10/10, **new API 44/44**, legacy 48/48, CTest 3/3.
Replay perf unchanged: 0.9× per-step gap vs PyTorch (bitsy 0.151 vs torch 0.162 ms/step),
clock 1530→1950 MHz. Compact core **2,026 lines** (budget 4,096, 2,070 spare).
`docs/FEATURE_INVENTORY.md` statuses updated for every touched row.

## Legacy-core cutover plan (concrete, NOT executed in this PR)
The 5,710-line legacy core (`src/core`, `src/ops`, `src/backends`, `include/bitsy`)
is retained only as a numeric reference. Removing it is gated on replacing its
one remaining consumer — the 48-case legacy parity harness.

1. **Map legacy cases → v2**: `tests/parity/` cases exercise tensor creation,
   elementwise, reduce, matmul, autograd, SGD on both `cpu_ref` and Vulkan.
   Audit each of the 48 against `bitsy_v2` (core primitives + PR #9 compositions).
   Any case not covered gets a v2 equivalent first.
2. **Add a v2 parity runner** producing the same 48-case report shape, so the
   `48/48` gate can run against `bitsy_v2` with zero loss of coverage.
3. **Flip the gate**: switch CI's legacy parity step to the v2 runner; require
   48/48 green on v2 for both CPU and Vulkan.
4. **Delete legacy sources**: remove `src/core`, `src/ops`, `src/backends`,
   `include/bitsy`, legacy CMake targets, and the legacy Python package.
   Update `tools/count_core_lines.py` to report a single core total (2,026 lines).
5. **Re-baseline**: full gates + replay benchmark after deletion; expect no
   numeric change because v2 was already the production path for all PR #1–#9 work.

Risks: (a) a legacy case depends on an op not yet in v2 — mitigate by step 1 audit;
(b) CPU-device parity — v2's C core has CPU stubs that must be re-verified;
(c) benchmark scripts that still import the legacy package — grep and repoint.
This plan is deliberately a separate PR: deleting 5,710 lines while changing the
gate in the same commit would make regressions hard to bisect.

## Honest scope / remaining work
- Feature inventory still has many `missing` rows (RNG/distributions, conv/pool
  layers, linalg decompositions, serialization, data loading, AMP, distributed,
  FFT/sparse/quantized — explicitly out of scope until the core is solid).
- GEMM tile autotune, shared-memory tiling, subgroup reductions still NOT done.
- Speed parity remains benchmark-scoped (1M fused pointwise chain), not universal.
- Clock not pinned; numeric diffs are honest (1.9e-6 on replay, 0..3.8e-5 here).
