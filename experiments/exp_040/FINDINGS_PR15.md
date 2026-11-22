# exp_040 — FINDINGS (PR #15: cooperative-matrix blocker + fusion correctness fix + APIs)

Status: PR #15 of the v0.2 rewrite.

## 1. Tensor-core / cooperative-matrix GEMM — BLOCKED by hardware, measured
The objective asked for a `GL_NV_cooperative_matrix` GEMM path. I checked the
actual device (Vulkan 1.4 SDK, driver 621.1.0.0):
```
deviceName = NVIDIA GeForce GTX 1660 SUPER
VK_NV_cooperative_matrix  : ABSENT
VK_KHR_cooperative_matrix : ABSENT
(no cooperative-matrix-related extension exposed at all)
```
**This target has no cooperative-matrix extension**, so a GLSL tensor-core
GEMM cannot be compiled or run here. This is a genuine hardware/driver blocker,
not an untried optimization. Subgroup ops (`subgroupSize=32`, shuffle/ballot/
vote/arithmetic) ARE present, but as documented in PR #14 they duplicate the
broadcast/coalescing the shared-memory kernel already gets free, so they do not
close the ~3× cuBLAS gap. Closing it would need an RTX-class GPU with
`VK_KHR_cooperative_matrix` (or a CUDA path via `VK_NV_cuda_kernel_launch`,
which is a different runtime). **I did not claim the gap was closed — it was not.**

GEMM evidence unchanged from PR #14: n=1024 6.32 ms vs torch 1.80 ms (~3.5×;
clock 1530→1950 MHz this run).

## 2. Real correctness bug found and fixed (important)
While adding `isfinite`, a composed expression that worked when intermediates
were realized step-by-step returned **all zeros** when evaluated lazily in one
expression. Instrumented the compiler:
```
[fuse] emit failed: too many fusion inputs ninst=23
```
Root cause: a fusion holds at most `RT_MAXIN=8` distinct leaf inputs. Deep
composition trees (multiple scalar constants) exceeded it; `emit()` returned -1
and `realize_fused()` **silently left `t->buf` NULL**, which `rt_to_host` read as
zeros. This affected both CPU and Vulkan backends and could have corrupted any
deeply composed user expression.

Fix: `realize_fused()` now retries up to 4 times, materializing the failing
node's direct children into buffers before re-emitting (each child becomes one
leaf input). A final explicit zero buffer is used if even 4 attempts fail, so
`buf` is never NULL. Verified: `isfinite([nan, inf, -inf, 0, 1])` now returns
`[0,0,0,1,1]` on both backends. All gates green after the fix.

## 3. API batch (pure Python)
Added `isnan`, `isinf`, `isfinite`, `nan_to_num`, `greater`, `less`, `equal`,
`hypot`, `copysign`. API parity extended 60 → **69 cases**, all green (including
a special-value section with NaN/±inf). Feature inventory updated.

## Gates (all green)
v2 parity 48/48, op 27/27, gradcheck 25/25, model 10/10, API 69/69,
CTest 2/2. Core **2,090 lines** (budget 4,096, 2,006 spare).
Fused-pointwise replay unchanged (0.9× per-step vs PyTorch, clock 1530→1845).

## Honest scope / remaining work
- **GEMM ~3.5× off cuBLAS** — blocked on this GPU by lack of cooperative-matrix
  extension. Needs RTX-class hardware or a CUDA-kernel path (a decision/hardware
  change, not a code change on this target).
- Full 285-function map still incomplete (RNG, conv/pool, linalg decompositions,
  serialization, data loading, AMP, distributed, FFT/sparse).
- Speed parity benchmark-scoped; single-step forced download still slower
  (0.95 vs 0.72 ms this run); clock not pinned.
