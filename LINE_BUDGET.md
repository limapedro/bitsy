# Line budget — C + Vulkan core ≤ 4096 lines

tinygrad's "fewest lines, maximum performance" is a hard design constraint. We
mirror it: the compiled C+Vulkan core must stay under **4096 lines**, with a
stretch goal of **2048**. `tools/count_lines.py` enumerates the core sources
(`src/`, `include/`) and fails CI above the budget. Docs, tests, benchmarks,
Python, shaders-as-data and the research branch do **not** count toward the C
core budget (shaders are counted in a separate row for transparency).

## Budget allocation (target, lines)

| Component | Target | Role |
|---|---:|---|
| `core/` device + arena + shape/dtype + error | 450 | Vulkan init, bump allocator, opaque types |
| `graph/` IR + lazy frontend + views | 500 | interned node, ops, strides |
| `sched/` scheduler + fusion | 350 | topo walk, elementwise chaining, cut points |
| `codegen/` op-program + specialization tables | 300 | fused program, GEMM/reduce params |
| `exec/` Vulkan pipelines + bindless + capture | 700 | 3 pipelines, one descriptor set, replay |
| `exec/cpu.c` reference backend | 300 | same scheduler, CPU oracle |
| `include/bitsy/` public headers | 250 | opaque ABI |
| **C core subtotal** | **2850** | ≤ 4096 budget (1246 headroom) |
| `shaders/*.comp` (counted separately) | 250 | fused / gemm / reduce |

## Baseline for comparison

The previous (pre-skeleton) core measured **5710 lines** across 60 C/H/GLSL
files (`tools/count_core_lines.py` on the legacy tree), dominated by ~10 GEMM
shader variants and a per-op dispatcher/ops layer. The v0.2 skeleton targets
2850 core lines by replacing those with one op-program kernel, one GEMM, one
reduce, and a lazy graph that owns autograd.

## Rules

1. No new shader per operation — extend the fused op-program.
2. No per-op C dispatch function — ops are node constructors.
3. Every C source added must state which budget row it belongs to.
4. Budget breaches require an ARCHITECTURE_LOG entry with numbers, as with any
   tolerance change.
5. `python tools/count_lines.py` prints per-file and total; CI enforces it.

## Progress tracking

The current line count is recorded in `docs/progress.json` under
`line_budget`, and shown in the README progress image once implementation
begins. On this documentation-only branch the value reflects the *target*
skeleton, not shipped code — the image labels this honestly as "planned".
