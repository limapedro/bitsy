# exp_040 — Bitsy 0.2: tinygrad-inspired minimal core rewrite

**Date:** 2026-11-22
**Status:** hypothesis (written before implementation, per project rule)

## Motivation

After the exp_037–039 parity work, the framework is numeric-correct but far
from PyTorch performance and structurally far from optimal: the compiled core
is **5152 lines** of C/headers/GLSL, with ~10 separate GEMM shader variants, a
separate `ops/` + `dispatcher/` + per-op kernel layer, and eager per-op Vulkan
submits. The user directed a major rewrite following tinygrad's philosophy:
fewest lines, maximum performance, core ≤ 4096 lines, released as **0.2**.

## Sources studied (details in `docs/study/`)

- tinygrad docs + source (`docs/study/TINYGRAD.md`): 4-layer stack
  (frontend → scheduler → lowering → execution), lazy UOp graph, fusion of
  pointwise chains into one kernel, TinyJit driver bypass, BEAM/autotuning,
  arithmetic-intensity / memory-bound reasoning.
- PyTorch 2.15 docs (`docs/study/PYTORCH.md`): design philosophy
  (usability first, simple over easy), autograd mechanics, torch.compile.
- Vulkan 1.4 spec (`docs/study/VULKAN.md`): bindless descriptors, push
  constants, timeline semaphores, persistent command buffers, subgroup ops,
  cooperative matrices, memory model.

## Hypothesis

A from-scratch compact core built around tinygrad's four layers — a lazy UOp
graph, a scheduler that fuses pointwise chains into ONE kernel, a generic
tiled GEMM with specialization constants, and an execution engine that records
command buffers once and replays them — can:

1. Implement the parity-relevant subset (elementwise, matmul, reductions,
   gelu/sigmoid/tanh/exp/log/sqrt, forward + backward) in **≤ 4096 lines** of
   C+Vulkan (measured over the new compiled core sources).
2. Keep numeric parity within existing tolerances (48/48 gate spirit: every
   implemented fn matches PyTorch within documented tolerance).
3. Reduce kernel count for a pointwise chain from N dispatches to 1, cutting
   submit/fence overhead (the documented root cause of the performance miss).
4. Leave conv/sequence/top-level model composition as compositions of the core
   primitives (as exp_038/039 already demonstrated), not separate kernels.

## Predictions

- New core compiles on Windows (MSVC) and Linux (gcc) with Vulkan + CPU fallback.
- Focused parity script vs PyTorch CUDA/CPU passes within existing tolerances.
- A fused chain (e.g. `((a*b)+c).relu()`) runs in one Vulkan dispatch; measured
  wall time per element improves vs the eager legacy path on the same GPU.
- Measured core line count ≤ 4096.
- Existing legacy parity gate remains green (legacy kept as reference build).

## Measurement

Each claim gets a command + JSON/txt evidence in `benchmarks/results/` and a
FINDINGS.md with all negatives. No tolerance relaxed. Version released as 0.2.

## Honesty

"Fewest lines" is a design constraint, not a claim of full PyTorch API coverage.
The rewrite targets the parity-relevant core; breadth (conv variants, optimizers,
data loading) remains out of scope and is stated plainly in FINDINGS. Any
numeric or performance miss is reported, not hidden.
