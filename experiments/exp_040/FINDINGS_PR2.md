# exp_040 — FINDINGS (PR #2: GEMM/reduce parity + pointwise extensions)

Status: PR #2 of the v0.2 compact-runtime rewrite. Builds on PR #1.

## What landed
- GEMM with **bias epilogue** + **transpose flags** (transA/transB) in both
  CPU (`gemm_cpu`) and Vulkan (`gemm.comp`) paths.
- Generic reduce with **kind** (sum / mean / max / min) in CPU + `reduce.comp`.
- Pointwise primitives: `pow, maximum, minimum, abs, sign, gt, ge, lt, le, eq, ne, where`
  fused into the one op-program kernel (`fused.comp` + `fused_cpu`).
- `rt_scalar`, `rt_reshape`, full public ABI in `bitsy_rt.h`.
- Autograd rules for the new ops (pow/max/min/abs/where/matmul-with-flags).
- `tests/unit/test_rt2.c` (CPU + Vulkan) and a ctypes parity harness
  `benchmarks/compact/parity_vs_torch.py` comparing against real PyTorch CUDA.

## Evidence (commands + output, Windows MSVC + Vulkan 1.4, GTX 1660 Super)

Unit tests (Vulkan-enabled build, both backends):
```
ctest --test-dir build-rtvk --output-on-failure
  1/3 unit_core ............ Passed
  2/3 unit_compact_core .... Passed   (PR #1)
  3/3 unit_compact_core2 ... Passed   (PR #2: gemm+bias, transA, mean/max/min,
                                       pow/max/min/abs/gt/where — CPU AND Vulkan)
100% tests passed, 0 tests failed out of 3
```

Numerics vs PyTorch CUDA (fp32, ctypes → bitsy_rt Vulkan device):
```
$ python benchmarks/compact/parity_vs_torch.py
  [PASS] elem_add .. elem_le         (11 cases, diff 0 .. 1.9e-6)
  [PASS] unary_exp .. unary_abs      (8 cases, diff 0 .. 1.9e-6)
  [PASS] where                       (0.0)
  [PASS] gemm_bias                   (1.9e-6)
  [PASS] gemm_transA                 (0.0)
  [PASS] reduce_sum/mean/max/min     (<= 1.8e-7)
  [PASS] fusion_dispatch_count = 1
  27/27 compact-runtime vs PyTorch CUDA cases passed
```

Constraints:
```
$ python tools/count_core_lines.py
  v0.2 COMPACT core: 1443 lines (budget 4096, 2653 spare)
$ python tests/parity/run.py
  48/48 parity cases passed. 0 skipped.
```

## Key result
The compact runtime now matches PyTorch CUDA on 27 numeric cases across GEMM
(with bias and transposes), all reduce kinds, and the extended fused pointwise
set — on the actual Vulkan device. This is real cross-framework parity evidence,
not an internal self-check.

## Bugs found and fixed
1. Test expectation error: `where(cond,P,Q)` last element expected 1 but correct
   value is 5 (P[3]=5, cond[3]=1) — fixed the test, implementation was right.
2. CPU-only stub signatures in rt_vk.c didn't match new gemm/reduce signatures —
   fixed.
3. Unused local in `rt_where` — removed.

## Honesty / what is NOT in PR #2
- No speed/benchmark numbers yet — correctness parity only. Capture/replay and
  Vulkan perf tuning are PR #5/#6.
- The full 285-function Python map is still not implemented; this PR adds the
  low-level primitives needed by many of them.
- Legacy core (5710 lines) still retained as reference; budget enforces the
  1443-line compact core.
