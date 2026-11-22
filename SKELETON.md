# Project skeleton spec — C + Vulkan core, Python library, benchmarks

This is the planned layout for the v0.2 implementation branch. No code exists
on the current documentation branch. Design is derived from
`docs/research/FRAMEWORKS.md`.

## 1. Directory layout

```
bitsy/
  CMakeLists.txt               # C11, Ninja; options: BT_VULKAN, BT_TESTS
  include/bitsy/
    bitsy.h                    # public C ABI (opaque handles)
    bt_tensor.h bt_shape.h bt_dtype.h bt_device.h   # opaque structs
  src/
    core/                      # device, arena allocator, shape/dtype, error
    graph/                     # interned node IR, lazy frontend, views
    sched/                     # topological scheduler + fusion into programs
    codegen/                   # op-program + specialization tables, render
    exec/                      # Vulkan device/pipelines/bindless/capture-replay
    exec/cpu.c                 # reference fallback (same scheduler)
  shaders/                     # fused.comp, gemm.comp, reduce.comp
  python/bitsy/                # ctypes or CPython binding + nn + optim
  tests/parity/                # numerics vs real PyTorch (dataclass config)
  tests/unit/                  # C unit tests (CTest)
  benchmarks/
    ops/                       # per-op numerics + speed vs PyTorch
    models/                    # full-model step speed (mlp/cnn/rnn/gpt2)
    common/clock.py            # GPU clock sampling
  tools/
    count_lines.py             # enforce C+Vulkan ≤ 4096 lines
    gen_progress.py            # regenerate assets/progress.svg
  docs/
    research/FRAMEWORKS.md
    progress.json              # measured numerics + speed data
  FEATURE_PARITY_MATRIX.md
  ROADMAP.md
  README.md                    # embeds assets/progress.svg
```

## 2. C + Vulkan core

- **Language**: C11 (MSVC `/std:c11`, gcc `-std=c11`). Vulkan 1.2 baseline
  (buffer device address for bindless); timeline semaphores for replay.
- **Core layers** (tinygrad's four, in C):
  1. *Frontend/IR*: opaque `bt_node` = `{op, src[], arg, shape, dtype=f32}`.
     Hash-consed. Lazy: ops build nodes; `bt_realize` triggers execution.
  2. *Scheduler*: topo walk; collapse maximal elementwise chains into one
     fused program (`CALL`); matmul/reduce are cut points that materialize a
     buffer. Views (reshape/transpose) are strides, not copies.
  3. *Codegen*: one op-program consumed by `fused.comp`; tiled GEMM and reduce
     parameterized by push/specialization constants. No per-op shaders.
  4. *Execution*: one bindless descriptor set (SSBO of device addresses),
     push constants for shapes, fused/GEMM/reduce pipelines, one primary
     command buffer captured once and replayed via timeline semaphore; CPU
     reference backend shares the scheduler for parity testing.
- **fp32 main dtype** throughout. Accumulation in fp32.

## 3. Python library

- Thin binding (`ctypes` first, CPython extension later) over the C ABI.
- `Tensor` with lazy ops; `nn` modules (Linear, activations, LayerNorm,
  Embedding, RNN/LSTM, attention); `optim` (SGD, AdamW); `functional`
  losses. Functional style (JAX/tinygrad): stateless ops are functions,
  modules only hold parameters — keeps the C surface small and composable.
- `realize()`/`jit()` explicitly exposed so users control capture/replay.

## 4. Benchmarks & parity (the "how good is it" loop)

- Every op and every model step runs **against real PyTorch CUDA**, fp32.
- For each: report `max_abs_diff` (numerics) and `torch_ms/bitsy_ms` (speed),
  with warmup, median+spread, correct sync (`flush`/`wait` vs
  `cuda.synchronize`), and live GPU clock — matching the project's existing
  honesty rules.
- Scripts produce JSON into `benchmarks/results/`; `tools/gen_progress.py`
  reads those plus `docs/progress.json` to regenerate `assets/progress.svg`.
- The full model benchmark reports both the **full step** and **each step**
  (per-layer/per-op) so the slow stage is visible.

## 5. Definition of done for v0.2

1. Core ≤ 4096 lines, builds Windows + Linux, CI green.
2. Every implemented op matches PyTorch within tolerance (parity gate green).
3. Every model (MLP, CNN, RNN, LSTM, GPT-2) matches PyTorch numerics.
4. Speed ratios reported honestly for every op/step/full model; progress image
   updated after each measurement. Target: ≥ 1.0× PyTorch on this GPU; if a
   ceiling is hit, document the mechanism (as prior experiments did) rather
   than hide it.
5. Tag `v0.2.0`, README progress image published.
