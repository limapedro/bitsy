# exp_040 — FINDINGS (PR #1: lazy graph + fused kernel)

Status: PR #1 of the v0.2 compact-runtime rewrite. Hypothesis: HYPOTHESIS.md.

## What landed in PR #1

- `src/rt/rt.c` — interned lazy node IR, op constructors, fusion scheduler
  (`emit` inlines pointwise chains into one op-program), CPU executor, C autograd.
- `src/rt/rt_vk.c` — Vulkan backend: device init, bindless buffer device
  addresses, 3 pipelines (fused/gemm/reduce), push constants, timeline replay hooks.
- `src/rt/shaders/{fused,gemm,reduce}.comp` — one op-program shader (stack
  interpreter), tiled GEMM with specialization constants, generic reduce.
- `include/bitsy_rt/bitsy_rt.h` — public C ABI.
- `tests/unit/test_rt.c` — CPU + Vulkan unit tests.
- `tools/count_core_lines.py` — reports compact core separately from legacy.

## Evidence (commands + output, Windows MSVC + Vulkan 1.4, GTX 1660 Super)

CPU build + unit test:
```
cmake --build build --target bitsy_rt rt_unit_tests
.\build\rt_unit_tests.exe
  [PASS] pointwise chain fuses into 1 dispatch
  [PASS] pointwise numerics        (out = 2,6,12,20,30,42)
  [PASS] matmul numerics           (58,64,139,154)
  [PASS] reduce_sum numerics       (6,15)
  [PASS] autograd d(sum x^2)=2x    (2,4,6,8,10,12)
  OK: 0 failures
```

Vulkan build + unit test (`-DBT_COMPACT_VULKAN=ON`): all shaders compile with
glslc; all CPU checks pass plus:
```
  -- Vulkan backend --
  [PASS] vulkan pointwise chain 1 dispatch
  [PASS] vulkan pointwise numerics
  [PASS] vulkan matmul numerics
  OK: 0 failures
```

Line budget:
```
$ python tools/count_core_lines.py
  v0.2 COMPACT core (budget-enforced): 1220 TOTAL
  legacy core (reference): 5710 TOTAL
  WITHIN BUDGET (2876 lines spare).
```

Legacy parity gate still green (nothing broken):
```
$ python tests/parity/run.py
  48/48 parity cases passed. 0 skipped.
```

## Key result

The central tinygrad idea works in C+Vulkan: a 5-op pointwise chain
`((t*t)+t).relu()` executes in **1 dispatch** on both CPU and GPU, with correct
numerics. This is the mechanism that eliminates the ~36 µs/op submit+fence
overhead that caused the legacy performance miss.

## Bugs found and fixed during PR #1

1. `VkBufferBeginInfo` typo (should be `VkCommandBufferBeginInfo`) in rt_vk.c —
   Windows MSVC rejected it; fixed.
2. GLSL `GL_EXT_shader_int64` extension is not accepted by this glslc; the
   correct name is `GL_EXT_shader_explicit_arithmetic_types_int64`. Fixed all 3 shaders.
3. GLSL runtime-sized array `v[]` cannot be indexed by `int64_t`; cast to `int`.
4. CMake expected shader file names `rt_fused.comp` but files were `fused.comp`;
   aligned names + generated header symbols.
5. `topo_collect` DFS pushed child-before-parent, looping forever and overflowing
   the stack (the crash seen in the prototype). Fixed push order.
6. `reduce_sum` backward used a zeros tensor instead of ones for broadcast;
   produced zero grads. Fixed to allocate a real ones tensor.

## Honesty / what is NOT in PR #1

- This is the *foundation*: only add/sub/mul/div + unary + matmul + reduce +
  transpose + autograd on the compact runtime. The full 285-function map
  (`MAP_*.md`) is not implemented yet — subsequent PRs.
- No JIT capture/replay benchmark yet (hooks only) — PR #5.
- No speed claims: this PR establishes correctness + the 1-dispatch fusion proof.
- The legacy core (5710 lines) is retained as a numeric reference until the
  cutover PR; the 1220-line compact core is what the budget enforces.
