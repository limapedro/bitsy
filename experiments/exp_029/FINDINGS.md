# exp_029 — fused four-parameter SGD update (neutral result)

**Date:** 2026-11-21
**Scope:** Phase 5 optimizer fusion + Phase 6 kernels. Parity: 43/43 incl.
gradient gate, CTest 1/1 Vulkan and CPU-only, MNIST W1 diff 3.278e-07,
CIFAR conv1 W diff 7.004e-07.

## Implementation

- `sgd_fused.comp`: one dispatch updates up to four parameters. `workgroup.y`
  selects the parameter, `workgroup.x` covers its elements; 12 storage bindings
  (4× W, 4× grad, 4× output) avoid descriptor-array dynamic indexing, which
  `glslc` rejected. A dedicated descriptor layout/pipeline was added (the
  descriptor pool grew by one set / 12 descriptors).
- `bt_sgd_step_fused()` C/Python API with a CPU fallback and a >4-parameter
  fallback in `Optimizer.step`.
- Three implementation bugs were caught before merge by CTest and the gate:
  a too-small descriptor pool (Vulkan init failed), a pointer-level mistake in
  the C fallback, and a ctypes out-array declared as `void*` instead of
  `void**`. All fixed and re-verified.

## Measurements (median of several runs)

| metric | before (exp_028) | after | target | status |
|---|---:|---:|---:|---|
| dispatches / step | 22 | **19** | ≤19 | **met** |
| captured step @1740 MHz | 0.156 ms | **0.153–0.159 ms** | ≤0.153 | **missed** (neutral) |
| graph ratio vs PyTorch CUDA graph | 0.43–0.54 | 0.43–0.45 | ≥0.85 | still missed |
| eager step | 0.906–1.042 ms | 1.002–1.035 ms | — | no regression |
| eager ratio vs PyTorch | 1.11–1.24 | 1.11–1.17 | ≥0.85 | **met** |

The ≥1.02× captured-step prediction was **not met**: the fused kernel does one
dispatch with `max_workgroups × 4` workgroups and a per-parameter if-chain,
which offsets the saved submit. The change is kept because it is correct and
reduces dispatch count by 3, but it is recorded as a neutral/negative result.

Elementwise 0.25–0.52 and standalone GEMM (median tuner ~0.72) are unchanged
within variance. Parity held with the same fp32 drift as exp_027/028.

## Predictions

- "≥3 fewer dispatches": **met** (3 fewer).
- "captured ≥1.02×": **not met** (neutral).
- "43-case gate green": **met** after fixing the implementation bugs.
- "parameters within 1e-5": **met** (direct comparison max diff < 2e-08).

## Interpretation / next

Dispatch count is not the only cost: workgroup count and branching inside a
fused kernel matter. The remaining 19 dispatches are 5 elementwise, 3
reduce_sum, 6 GEMM variants, 1 vec4 elementwise, 1 log_softmax and others. The
captured graph ratio (~0.43) and the elementwise/GEMM targets remain missed.
Next: packed f16 GEMM or a fundamentally different step-level fusion; all
changes must pass the 43-case gate.

Raw data in `experiments/exp_029/{before_*,after_*,after_step_profile.json}`.
