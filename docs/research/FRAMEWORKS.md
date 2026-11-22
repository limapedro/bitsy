# High-level study: tinygrad, PyTorch, JAX

A research record written before any skeleton code, comparing how the three
frameworks turn a user program into fast accelerator work. Sources read
2026-11-22/23:

- tinygrad: https://docs.tinygrad.org/ (home, quickstart, developer intro /
  layout / speed / UOp), plus source `tinygrad/tensor.py`,
  `tinygrad/renderer/cstyle.py`, `tinygrad/engine/jit.py`.
- PyTorch 2.15: https://docs.pytorch.org/docs/2.15/index.html and
  https://docs.pytorch.org/docs/2.15/community/design.html.
- JAX: https://docs.jax.dev/en/latest/ (home, JAX 101) and the docs tree
  (JAX 201/301/401/501/601: jit, jaxpr, Pallas, internals).

Everything below is our synthesis; quoted phrases are attributed. This
research informs the skeleton in `ROADMAP.md`, `SKELETON.md` and
`FEATURE_PARITY_MATRIX.md`.

## 1. The shared problem

All three solve the same problem: a user writes math on arrays; the framework
must (a) get the *numbers* right (PyTorch is the numeric oracle) and (b) run as
fast as possible on the accelerator. They differ radically in *where* they
spend complexity and *when* they decide what to compute.

## 2. tinygrad — "reduce complexity, maximum performance, few lines"

- **Philosophy**: smallest readable stack; performance comes from structure,
  not from thousands of kernels. Home page: *"All operations in tinygrad are
  lazy, meaning they won't do anything until you realize"*; *"tinygrad requires
  @TinyJit to be fast … replay the kernels used in the decorated function."*
  README: *"Your accelerator of choice only needs to support a total of ~25 low
  level ops"* and *"No code golf … the true goal is reducing complexity."*
- **Four layers** (developer intro): a PyTorch-like **frontend** (Tensor → one
  `UOp`), a **scheduler** (splits the graph into `CALL`s — *one CALL = one GPU
  kernel*), a **lowering** engine (graph rewrites → linear UOps → render →
  compile), and an **execution** engine (dispatch or JIT replay).
- **One IR**: `UOp(op, src, arg)`, hash-consed so pointer equality = structural
  equality. Ops are a short list (~25 math primitives, no Relu/Softmax/LayerNorm
  — those are composed). Views (reshape/permute/expand/pad) are free UOps, not
  data movement.
- **Fusion = shared loop nest**: the scheduler assigns `RANGE` axes; a value
  inherits its single consumer's ranges, so an elementwise chain becomes ONE
  kernel. Optimization is *axis retyping* (`SPLIT` into UPCAST/LOCAL/GLOBAL/WARP
  axes); tensor cores are just another axis type (`WMMA`).
- **Speed model** (developer/speed): memory is the bottleneck; arithmetic
  intensity dictates tiling (~100 reuses from HBM, ~4 from L1 on a 4090); BEAM
  search compiles many equivalent kernels and times them; execution speed
  comes from *bypassing the driver* by prebuilding the command queue (faster
  than CUDA, often than CUDA graphs).
- **Takeaway for us**: lazy graph + fusion + captured replay is the cheapest
  route to both few lines and high speed; resist adding a kernel per operation.

## 3. PyTorch — "usability first, reasonable performance"

- **Philosophy** (design.html): *"Usability over Performance"*, *"Simple Over
  Easy"* (*"explicit is better than implicit"*), *"Python First"*. It
  deliberately avoids restriction-first regimes (no mandatory static shapes /
  graph mode). Performance is achieved by enormous C++ dispatch engineering,
  not by hiding cost from the user.
- **Execution model**: eager by default; each op is dispatched through a C++
  stack to ATen kernels. `torch.compile` (TorchDynamo → AOTAutograd → Inductor)
  captures graphs, fuses pointwise ops into Triton kernels, and can use CUDA
  graphs to cut launch overhead — the "compile path" converges toward
  tinygrad/JAX ideas while keeping eager semantics.
- **Autograd**: dynamic graph recorded as ops execute (`requires_grad`,
  `grad_fn`), reverse-mode on `.backward()`. Explicit, debuggable, flexible.
- **Numeric contract**: broadcasting semantics, stride/view semantics, and
  fp32 accumulation order are all defined; cross-backend exact bit equality is
  *not* guaranteed, so numeric parity is judged with tolerances (EXACT for
  simple elementwise, rtol/atol ~1e-5 for matmul/transcendentals).
- **Takeaway for us**: PyTorch is the oracle, not the architecture to copy.
  Match its *semantics and numerics*; use tinygrad/JAX *compilation* ideas for
  speed. Our parity suite must encode PyTorch's tolerance reality honestly.

## 4. JAX — "composable transformations of pure functions"

- **Philosophy**: *"high performance array computing … accelerator-oriented …
  program transformations."* NumPy-style API (`jax.numpy`) plus composable
  transforms: `jit` (compile), `grad` (autodiff), `vmap` (batching), and
  sharding/parallelization. Same code runs on CPU/GPU/TPU.
- **Tracing model**: transforms work by tracing a Python function into a
  staged IR (**jaxpr**) — a small, explicit, typed language of *primitives*.
  `jit` traces once, lowers to XLA (StableHLO → LLVM/TPU IR), compiles, and
  caches by shapes/dtypes; replay avoids Python entirely. `grad` transforms the
  jaxpr (JVP/VJP), `vmap` batches by transforming primitives. The docs:
  *"JAX 601 — the jaxpr language, primitives, and Autodidax, which builds JAX's
  core from scratch."*
- **Purity & explicitness**: functions are pure (no hidden RNG state — explicit
  PRNG keys), state is threaded through (`pytrees` for nested structures).
  This purity is *why* transformations compose and why compilation is possible.
- **Custom kernels**: Pallas (JAX 401) writes GPU/TPU kernels in a Python-like
  model; FFI calls external code. XLA does the fusion and layout optimization
  that tinygrad does by hand.
- **Precision** (JAX 101 default dtypes / numerical precision docs): fp32 is
  the default; 64-bit requires an explicit flag — analogous to our "fp32 is the
  main dtype" decision.
- **Takeaway for us**: the jaxpr idea — a tiny explicit typed IR that every
  transformation rewrites — is the cleanest mental model for our node/graph.
  Purity (no hidden mutation) makes capture/replay and grad composition safe.

## 5. Comparison at a glance

| Axis | tinygrad | PyTorch | JAX |
|---|---|---|---|
| Default mode | lazy + JIT | eager (+ opt-in compile) | tracing + jit |
| Core IR | UOp (~25 ops) | ATen ops + FX/Inductor | jaxpr primitives |
| Autodiff | reverse on UOp graph | dynamic `grad_fn` | transform (`grad`/JVP/VJP) |
| Fusion | scheduler, one CALL=kernel | Inductor/Triton | XLA |
| Speed trick | prebuilt command queue, driver bypass | C++ dispatch + CUDA graphs | XLA + compiled jit |
| Lines (conceptual core) | very small (few thousand py) | huge (C++ ATen) | medium (Python + XLA dep) |
| View semantics | free UOps | stride views | shape/stride in jaxpr |
| Numeric oracle role | ≈ PyTorch | **the oracle** | ≈ PyTorch (fp32) |

## 6. Decisions we adopt for this project

1. **PyTorch = numeric oracle.** Every op is validated against real PyTorch
   (CUDA on this machine, fp32) with documented tolerances; misses are recorded
   honestly. Parity gates are the definition of "how good the library is".
2. **tinygrad = architecture.** Lazy graph → fusion scheduler → parameterized
   kernels → captured command buffer replay. Few C lines, one op-program kernel,
   one GEMM, one reduce.
3. **JAX = IR hygiene.** A small explicit typed node IR; pure functional ops so
   autodiff and JIT compose; fp32 default.
4. **fp32 main dtype** everywhere; other dtypes are follow-on work.
5. **Speed target = PyTorch on this GPU.** Track two numbers continuously:
   (a) numeric max-abs-diff per op/step, (b) speed ratio torch/bitsy for each
   op, each model step, and the full model. The progress image tracks both.
6. **Line budget**: C+Vulkan core ≤ 4096 lines (stretch 2048), enforced by a
   counting tool, mirroring tinygrad's "fewest lines" constraint.
