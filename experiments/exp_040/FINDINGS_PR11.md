# exp_040 — FINDINGS (PR #11: legacy cutover steps 3–5 COMPLETE)

Status: PR #11 of the v0.2 rewrite. Executes the remaining cutover steps from
FINDINGS_PR9/PR10. **The 5,710-line legacy core is now deleted.**

## Step 3 — CI gate flipped to v2
`.github/workflows/ci.yml`, both Windows and Linux jobs:
- Configure explicitly with `-DBT_BUILD_COMPACT=ON` (kept as a compatibility
  no-op in CMake; the compact runtime is now the only runtime and always built).
- Gate changed from `python tests/parity/run.py` → `python tests/parity/v2_runner.py`.
- Artifacts repointed to `tests/parity/reports/v2_latest.json`.

**Honest note on `BT_COMPACT_VULKAN`**: I did NOT enable it in CI. Both CI jobs
install CPU-only PyTorch, so `v2_runner.py` skips Vulkan cases exactly as the
legacy gate did; enabling compact Vulkan would additionally require the Vulkan
SDK on Windows runners and headers on Linux without adding any tested coverage.
The local Vulkan path remains fully validated (`build-rtvk`, 48/48 with CUDA).

## Step 4 — legacy sources deleted
Removed: `src/core/` (15 files), `src/ops/` (5), `src/backends/` (35),
`include/bitsy/` (11), `python/bitsy/` (10), `cmake/BitsyShader.cmake`,
`tests/unit/test_core.c`, and the five legacy parity harnesses (`run.py`,
`grad_flow.py`, `train_scale.py`, `gradcheck.py`, `conv_check.py`).
`CMakeLists.txt` reduced to the single `bitsy_rt` target + 2 compact C tests.
`tools/count_core_lines.py` now reports one core total; the "legacy reference"
figure is gone.

Verified no remaining legacy imports in active code (`python/bitsy_v2`,
`tests/parity/v2_runner.py`, `benchmarks/compact`, `tools`).

## Step 5 — re-baseline (all green)
Two fresh builds validated:
- CI-equivalent (`build-cisim`, no Vulkan): `ctest` 2/2; `v2_runner.py` 48/48.
- Vulkan dev build (`build-rtvk`): `ctest` 2/2; `v2_runner.py` 48/48.

Gates after cutover:
```
v2 parity        48/48   (legacy gate removed — replaced by this)
op parity        27/27
gradcheck        25/25
model parity     10/10
new API parity   44/44
CTest             2/2   (was 3/3; the deleted unit_core tested the legacy lib)
core           2,026 lines / 4,096 budget (2,070 spare)
```
Replay perf unchanged: 0.9× per-step vs PyTorch CUDA (bitsy 0.159 vs torch
0.183 ms/step), clock 1530→1845 MHz, spread logged.

## Honest scope / remaining work
- Historical benchmark scripts under `benchmarks/models/` and `benchmarks/ops/`
  still `import bitsy` and are now non-functional. Their recorded result JSONs
  remain as evidence. They are not part of the active gate surface and were
  left in place rather than deleted as unrelated historical artifacts; migrating
  or removing them is a follow-up.
- Full 285-function map (`docs/FEATURE_INVENTORY.md`) still incomplete.
- GEMM tile autotune, shared-memory tiling, subgroup reductions still NOT done.
- Speed parity remains benchmark-scoped (1M fused pointwise chain); single-step
  forced download still slower than PyTorch (1.05 vs 0.75 ms in this run);
  clock not pinned.
- CTest count dropped 3→2 by design (legacy C unit test deleted with its lib).
