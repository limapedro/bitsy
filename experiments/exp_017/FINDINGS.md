# exp_017 — Findings: fuse backward and optimizer op chains

**Date:** 2026-11-21
**Scope:** Phase 5 fusion + Phase 6 kernels. Parity: 42/42, CTest 1/1 Vulkan
and CPU-only. No tolerance changed.

## Profiler-guided targets

Profiling one MLP step with `bt.profile_dump` gave **66 dispatches / 66
barriers**: 22 scalar elementwise, 20 tiled GEMM, **16 transpose**, 6 vec4
elementwise, 2 register-blocked vec4 GEMM. The 16 transposes came from
`Tensor.__matmul__` eagerly materialising `aT`/`bT` for backward; the SGD step
was `sub(mul(grad, lr))` plus an lr upload per parameter.

## Implementation

- Added `bt_tensor_transpose_view()` (C + Python `C.transpose_view`): a 2D
  transpose as a **strided view sharing storage**, zero Vulkan dispatch. The
  GEMM already consumes tensor strides, so matmul backward now uses views.
- Added a fused `sgd_step.comp` shader and `bt_sgd_step()` C/Python API:
  `out = w - lr*grad` in **one** dispatch with lr as a push constant (no lr
  upload). The CPU-only build keeps the same multiply-then-subtract order via
  existing ops.

## Dispatch and throughput results

| metric | before | after | target | status |
|---|---:|---:|---:|---|
| dispatches / step | 66 | **46** | ≤46 | **met** (30.3% fewer) |
| barriers / step | 66 | 52 | — | improved |
| captured step ms (1732 MHz) | 0.296–0.334 | **0.2566–0.2569** | ≤0.25 | marginally missed |
| captured speedup | — | **1.15–1.30×** (≈1.23× central) | ≥1.3× | not reliably met |
| graph ratio vs PyTorch CUDA graph | 0.20–0.23 | **0.259** | ≥0.28 | missed |
| eager step ms (1530 MHz) | 2.626 | **2.460–2.469** | ≤2.29 | missed |
| eager speedup | — | **1.06–1.07×** vs 2.626 | ≥1.15× | missed |
| eager ratio vs PyTorch | 0.37–0.41 | **0.41–0.47** | — | improved |

The 20 fewer dispatches per replay translate into a real but
smaller-than-predicted gain: 0.334 → 0.257 ms in the best before/after pair
(1.30×), but only ~1.15× against the faster before run. The ≥0.28 graph ratio
and ≥1.15× eager predictions were not met; reported rather than cherry-picked.
Elementwise 0.40–0.57 and GEMM 1024² 0.67 are unchanged within run variance.

## Parity

- 42/42 parity cases; CTest 1/1 on Vulkan and CPU-only (the new SGD op has a
  CPU path).
- MNIST final W1 diff **1.788e-07** (was 1.490e-07) and CIFAR final conv1 W diff
  **6.631e-07** (was 6.855e-07): small run-to-run drift, all loss curves and
  accuracies match at reported precision. Fused SGD preserves the multiply-
  then-subtract fp32 order, so no systematic drift is introduced.

## Interpretation / next

Removing 20 of 66 dispatches and fusing SGD is a real improvement (captured
0.257 ms, eager 2.46 ms, ratio 0.41–0.47 — the best eager figure so far), but
the remaining ~0.19 ms/step of captured graph time is spread over 46 small
dispatches, still ~3.9× PyTorch CUDA graph throughput. The biggest remaining
single class is the **20 tiled GEMM dispatches**, many of which are tiny
reduction matmuls (ones-vector reductions used by the interim autograd).
Next: replace those reductions with a dedicated reduction kernel (and fuse the
loss/softmax backward), then re-measure with full parity. `exp_NNN`.

Raw data: `experiments/exp_017/{before_*,after_*,after_step_profile.json}`.
