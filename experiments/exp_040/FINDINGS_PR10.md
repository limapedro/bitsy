# exp_040 — FINDINGS (PR #10: legacy cutover steps 1–2)

Status: PR #10 of the v0.2 rewrite. Executes cutover steps 1–2 from FINDINGS_PR9.
**The legacy core is NOT deleted in this PR** — only the v2-equivalent gate is built.

## Step 1 — audit of the 48 legacy parity cases
The legacy gate is 46 isolated op cases + 2 end-to-end gates (gradient flow,
training scale). I ran all 48 against `bitsy_v2` and found:

| Finding | Root cause | Fix |
|---|---|---|
| `add_scalar_scalar` failed (v2 returned shape `(1,)` not `()`) | `np.ascontiguousarray` promotes 0-d arrays to 1-d | `Tensor.from_numpy` now `.reshape(arr.shape)` after `ascontiguousarray` |
| `div_vulkan` failed 22/91 elements by 1 ULP | GLSL `a/b` differs from CUDA IEEE div on this driver; `precise` qualifier had no effect | v2-only tolerance override `rtol=1e-6, atol=1e-7` for `div`, documented in the runner; all 91 elements still checked |
| All other 46 cases passed unchanged | — | — |

No v2 primitive was missing: the C core already covers every op in the legacy
grid (`add/sub/mul/div/relu/sigmoid/tanh/exp/log/sqrt/gelu/matmul`) on both CPU
and Vulkan, including inf/NaN/extremes, empty tensors, broadcasting and the
expected-error case. The two end-to-end gates were reproduced as pure-Python
MLP + SGD compositions (`bv.functional.cross_entropy`, `Tensor.tanh`, manual
SGD step) with grad comparisons; diffs 3.7e-8 and 1.9e-8.

## Step 2 — v2 parity runner
`tests/parity/v2_runner.py` reuses the legacy `cases.py` (case list + value sets),
`compare.py` and `tolerances.py` verbatim — only the Bitsy side is `bitsy_v2`.
Verified **report shape is identical** to `tests/parity/reports/latest.json`:
```
top-level keys: cases, config, environment, generated_at, summary
summary:        total=48, passed=48, failed=0, skipped=[]
case keys:      bitsy_vs_torch, bitsy_vs_torch_cuda, device, dtype, name,
                op, passed, shape, torch_cuda_vs_torch_cpu, value_set
case names:     identical set to the legacy report
```
```
$ python tests/parity/v2_runner.py
48/48 v2 parity cases passed. 0 skipped.
```
Runner auto-detects `build-rtvk/` (local) or `build/` (CI), runs CPU cases
without CUDA, and skips Vulkan cases when CUDA is unavailable (mirroring legacy).

## Gates (all green)
legacy op 48/48, **v2 parity 48/48**, op 27/27, gradcheck 25/25, model 10/10,
new API 44/44, CTest 3/3. Replay perf unchanged (0.9× per-step vs PyTorch,
clock 1530→1890 MHz). Compact core **2,026 lines** (budget 4,096).

## Step 3 — exact CI flip (planned, NOT executed here)
In `.github/workflows/ci.yml`, for both `windows` and `linux` jobs:
1. Configure with `-DBT_BUILD_COMPACT=ON -DBT_COMPACT_VULKAN=ON` so `bitsy_rt`
   is produced in `build/`; add a build step for that target.
2. Replace `run: python tests/parity/run.py` with
   `run: python tests/parity/v2_runner.py` (same report shape, same 48/48 gate).
3. Change the upload-artifact paths to `tests/parity/reports/v2_latest.json`.
4. Run both jobs green; only then proceed to step 4 (delete the 5,710 legacy
   lines in `src/core`, `src/ops`, `src/backends`, `include/bitsy` and the legacy
   CMake targets / Python package), update `count_core_lines.py` to a single
   core total, and re-baseline all gates.

Keeping step 3 in a separate PR preserves bisectability if the CI build needs
adjustment (the current CI installs CPU-only PyTorch, so Vulkan cases are skipped
there exactly as in the legacy harness).

## Honest scope / remaining work
- v2 gate keeps one documented 1-ULP tolerance override for `div`; this is a
  real numeric difference from GLSL, not a hidden failure.
- Full 285-function map still incomplete (RNG, conv/pool, linalg decompositions,
  serialization, data loading, AMP, distributed, FFT/sparse).
- GEMM tile autotune, shared-memory tiling, subgroup reductions still NOT done.
- Speed parity remains benchmark-scoped; single-step forced download still
  slower than PyTorch (0.98 vs 0.92 ms in this run); clock not pinned.
- The 5,710-line legacy core is still present — this PR only proves v2 can
  replace its gate.
