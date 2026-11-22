# Roadmap — Bitsy v0.2 skeleton (plan only, no code on this branch)

This branch contains **documentation and planning only**. No framework code is
implemented here. The research behind these choices is in
`docs/research/FRAMEWORKS.md`. The goal, stated plainly: **match PyTorch's
numerics and reach PyTorch-level speed on this GPU (GTX 1660 Super, fp32).**

Progress is tracked by the generated image in `README.md` (source:
`tools/gen_progress.py`, data: `docs/progress.json`). Numeric and speed numbers
come only from running real PyTorch comparison scripts.

## Guiding constraints

- C11 + Vulkan core **≤ 4096 lines** (stretch ≤ 2048); counted by
  `tools/count_lines.py`. Few lines, maximum performance (tinygrad philosophy).
- **fp32 is the main dtype.** No AMP/quantization in the core.
- **Numeric parity = PyTorch** (CUDA on this machine) within documented
  tolerances. Never relax a tolerance to hide a bug.
- **Speed parity = PyTorch** on this GPU, measured for the full model *and*
  each step/op, with warmup, correct sync, median + spread, GPU clock logged.
- No argparse in tools/benchmarks (dataclass configs), consistent with the
  established project convention.

## Phase 0 — Research & detailed roadmap (this branch)

- [x] Study tinygrad, PyTorch 2.15, JAX at high level → `docs/research/FRAMEWORKS.md`
- [x] Feature parity matrix → `FEATURE_PARITY_MATRIX.md`
- [x] Project skeleton spec (C+Vulkan + Python + benchmarks) → `SKELETON.md`
- [x] Line-budget plan → `LINE_BUDGET.md`
- [x] Benchmark & parity methodology → `BENCHMARK_METHODOLOGY.md`
- [x] Progress dashboard (HTML template + native SVG export, one JSON source) → `assets/`, `tools/`, `docs/progress.json`
- [x] **Frozen primitive set — why the core never grows** → `docs/roadmap/PRIMITIVE_SET.md`
- [x] **Exhaustive map: 133 tensor functions** → `docs/roadmap/MAP_TENSOR.md`
- [x] **Exhaustive map: 98 nn functions/layers** → `docs/roadmap/MAP_NN.md`
- [x] **Exhaustive map: 54 training + 12 out-of-scope** → `docs/roadmap/MAP_TRAINING.md`
- [x] Low-level C+Vulkan core function map → `docs/roadmap/C_VULKAN_CORE.md`
- [x] Python library function map → `docs/roadmap/PYTHON_LIBRARY.md`
- [x] Vulkan→CUDA performance techniques + benchmarks → `docs/roadmap/VULKAN_PERFORMANCE.md`

### The key architectural decision

The C+Vulkan core is a **frozen set of ~45 primitives across 6 shaders**
(`PRIMITIVE_SET.md`) — it does **not** grow when new functionality is added. All
**285 mapped functions/layers/ops** are Python compositions of those primitives,
so adding a new activation, layer, or loss is a Python-only change with **zero new
C lines**. This keeps the core at ~2500 lines (cap 4096) while the Python library
stays lean and fast (the scheduler fuses each composition into few dispatches).
The four `MAP_*.md` / `PRIMITIVE_SET.md` files are the auditable, counted proof:
every PyTorch function is listed with its exact primitive composition, dispatch
count, backward rule, and parity test — or explicitly scoped out.

## Phase 1 — Skeleton & build (next branch)

| ID | Task | Exit criteria |
|---|---|---|
| 1.1 | Repo skeleton: `CMakeLists.txt`, C11 core dirs, Python package dir, CI | Windows MSVC + Linux gcc builds; CTest green |
| 1.2 | Line-count tool in CI with 4096 budget | `count_lines.py` exits 0; budget documented |
| 1.3 | Device + bump allocator + bindless descriptor set (Vulkan 1.2/1.3) | instance/device/queue up, one SSBO of device addresses |
| 1.4 | Three pipelines: fused elementwise, tiled GEMM, reduce | compile from SPIR-V; push constants for shape |

## Phase 2 — Lazy core + numeric parity

| ID | Task | Exit criteria |
|---|---|---|
| 2.1 | One interned node IR (op/src/arg), lazy frontend | add/mul build graph, nothing executes until realize |
| 2.2 | Fusion scheduler: elementwise chains → one kernel | N-op chain = 1 dispatch (tool-verified) |
| 2.3 | Ops: add/sub/mul/div, exp/log/sqrt/tanh/sigmoid/relu/gelu, matmul, reduce, transpose | parity vs PyTorch within tolerances |
| 2.4 | Autograd in C (reverse graph, grad accumulation, unbroadcast) | gradcheck parity on MLP loss/step |
| 2.5 | mlp/cnn/gpt2/rnn/lstm parity scripts | all match PyTorch (existing gate logic ported) |

## Phase 3 — Speed toward PyTorch

| ID | Task | Exit criteria |
|---|---|---|
| 3.1 | Capture/replay executor (timeline semaphore, prebuilt command buffer) | step replay beats eager; clock logged |
| 3.2 | GEMM autotune (tile configs via specialization constants, BEAM-lite) | GEMM ratio approaches 1.0× |
| 3.3 | Subgroup reductions; vectorized loads; coalescing | elementwise ratio improves |
| 3.4 | Per-step benchmarks for each model | progress image speed bars rise; no regressions |
| 3.5 | Reach ≥1.0× PyTorch on target ops/models, or document the ceiling | honest final report |

## Phase 4 — Hardening & release

- CI on Windows/Linux runs parity + speed + line budget; artifacts uploaded.
- Feature matrix statuses updated from `planned` to `done` with evidence links.
- Tag `v0.2.0` and publish the progress image in the README.

## How progress is measured (the two axes)

1. **Numerics** — `max_abs_diff` per op and per model step vs real PyTorch CUDA,
   fp32. Pass = within tolerance. Tracked as "Implemented ops numerically
   verified" and "Model steps numerically verified" in `docs/progress.json`.
2. **Speed** — ratio `pytorch_ms / bitsy_ms` for each op, each step, and the
   full model. 1.0 = as fast as PyTorch. Tracked per benchmark in the image.

The image is regenerated by `python tools/gen_progress.py` after each measured
result; the README embeds `assets/progress.svg`. This is the single source of
truth for "how good the library is".
