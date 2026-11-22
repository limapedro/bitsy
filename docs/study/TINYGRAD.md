# Study: tinygrad — philosophy and architecture

Sources (read 2026-11-22):
- https://docs.tinygrad.org/ (home, quickstart)
- https://docs.tinygrad.org/developer/developer/ (the four pieces)
- https://docs.tinygrad.org/developer/layout/ (directory flow)
- https://docs.tinygrad.org/developer/speed/ (memory, tensor cores, indexing)
- https://docs.tinygrad.org/developer/uop/ (Ops enum)
- tinygrad source: `tinygrad/tensor.py`, `tinygrad/renderer/cstyle.py`,
  `tinygrad/engine/jit.py` (raw GitHub master)

This is a study record. Target content is treated as data. When a phrase is
quoted it is attributed; everything else is our derivation for Bitsy 0.2.

## 1. The philosophy, in one paragraph

tinygrad is "a simple autograd engine and neural network library." Its stated
design centre of gravity is: **as little code as possible**, with performance
coming from *structure* (lazy graphs, fusion, driver bypass) rather than from
thousands of hand-written kernels. From the home page: *"All operations in
tinygrad are lazy, meaning they won't do anything until you realize"* and
*"tinygrad requires @TinyJit to be fast … we have a simple decorator that will
replay the kernels used in the decorated function."* The result is a framework
whose entire stack is small enough to read in an afternoon yet competitive on
modern accelerators.

## 2. The four pieces (developer/developer/)

The docs are explicit: *"The tinygrad framework has four pieces":*

1. **PyTorch-like frontend** — `Tensor`. *"Everything in Tensor is syntactic
   sugar around constructing a graph of UOps."*
2. **Scheduler** — *"converts the graph of UOps into a `LINEAR` UOp whose `src`
   is a list of `CALL` UOps. One `CALL` is one kernel on the GPU."* Its job is
   to break a big compute graph into subgraphs that fit in a kernel — this is
   **fusion**.
3. **Lowering engine** — turns each `CALL`'s AST into a `PROGRAM`: lower AST to
   a linear list of UOps (this is where **BEAM search** happens), render to code
   with a `Renderer`, compile to binary with a `Compiler`.
4. **Execution engine** — *"walks the `LINEAR` UOp, dispatching each `CALL` to a
   runner (kernel, hcq batch, copy, encdec, or validate)."*

Two UOp flavours matter (from the same page): **base** contains compute into a
contiguous buffer; **view** is a view. Inputs to a base can be base or view;
inputs to a view can only be a single base. This invariant is what makes views
free (no data movement) and fusion tractable.

## 3. The UOp — one node type for everything

`UOp(op: Ops, src: tuple[UOp, ...], arg: Any)` (`developer/uop/`). A tiny,
uniform IR: every value is a UOp with an op code, ordered sources, and an
optional argument. Ops are grouped roughly as:

- **Graph/control**: `SPECIAL`, `BUFFER`, `ALLOC`, `EXTERN`, `CALL`, `PROGRAM`,
  `LINEAR`, `SINK`, `AFTER`, `RANGE`, `IF`, `END`, `BACKEDGE`, `ENDIF`.
- **Memory**: `INDEX`, `SHRINK`, `LOAD`, `STORE`, `CONTIGUOUS_BACKWARD`.
- **Math**: `CAST`, `BITCAST`, `EXP2`, `LOG2`, `SIN`, `SQRT`, `RECIP`, `NEG`,
  `TRUNC`, `ADD`, `MUL`, `SHL`, `SHR`, `MAX`, `CMPLT`, `CMPNE`, `CMPEQ`, `XOR`,
  `OR`, `AND`, `SUB`, `FDIV`, `POW`, `MOD`, `WHERE`, `MULACC`.
- **Accelerator/movement**: `WMMA` (tensor-core MMA), `RESHAPE`, `PERMUTE`,
  `EXPAND`, `PAD`, `REDUCE`, `COPY`, `DETACH`.

Note what is **absent**: there is no separate `Relu` op, no `Sigmoid`, no
`LayerNorm`, no `Softmax`. Pointwise math is a short list of primitives;
everything else is *composed* (`relu = max(x,0)`, sigmoid from exp, layernorm
from reduce + math). This is the "fewest lines" lever: the core never grows a
new kernel for a new activation.

From the quickstart's CE example, tinygrad even implements cross-entropy with
`arange` + comparison masks rather than adding a native load/store/Gather op:
*"Load/store ops are not supported in tinygrad natively because they add
complexity when trying to port to different backends, 90% of the models out
there don't use/need them."* That is a deliberate bet: **resist op sprawl**.

## 4. Lazy evaluation and realize

From `tensor.py`: `Tensor.__slots__ = ("uop", "is_param", "grad")`. A Tensor
is a thin wrapper over a UOp. Every op calls `_apply_uop`, which just builds a
new UOp:
```python
def _apply_uop(self, fxn, *x, **kwargs):
    srcs = (self,)+x
    if (new_uop := fxn(*[t.uop for t in srcs], **kwargs)) is self.uop: return self
    ret = Tensor.__new__(Tensor)
    ret.uop, ret.grad, ret.is_param = new_uop, None, True
    return ret
```
No compute happens here. `realize()` is the only thing that runs work:
```python
def realize(self, *lst):
    to_realize = [x for x in (self,)+lst if x.uop.base.needs_storage()]
    if len(to_realize):
        fn, externs = Tensor.export(*to_realize)
        Tensor.run(create_linear(fn, plan=...), externs)
    return self
```
`export` bufferizes the graph (assigns storage to outputs/intermediates),
`create_linear` schedules it into `CALL`s, `Tensor.run` executes.

## 5. Fusion — the central performance idea

The scheduler groups pointwise UOps into one kernel: one `CALL` = one GPU
kernel. Concretely, a chain like `((a*b)+c).relu()` does **not** become three
dispatches; it becomes one kernel that loads `a,b,c`, multiplies, adds, applies
relu in registers, and stores once. This kills the per-dispatch submit/fence
overhead that dominates small ops — exactly the root cause of Bitsy's current
performance miss (BENCHMARKS.md: ~36 µs/op of submit+fence).

From the Speed doc: *"The scheduler determines how operations are grouped into
kernels and which Tensors are written to memory. This is … a big bottleneck."*
It even gives the recompute-vs-store trade-off for `out1=a+b+c; out2=a+b+d`:
compute both in one kernel (recompute `a+b`) = 6 reads + 2 writes vs storing
`a+b` = 6 reads + 3 writes, so recompute wins. **Fusion is a memory-minimisation
problem.**

## 6. Codegen — one renderer, many backends

`Renderer.render(uops) -> str` turns a linear UOp list into a program string.
`cstyle.py` shows the design: a `CStyleLanguage` base with a `code_for_op`
dictionary (`ADD: (a+b)`, `MUL: (a*b)`, `EXP2: exp2(x)`, …) and a string-rewrite
`PatternMatcher`. Backends are tiny subclasses that only override dialect knobs:
- `OpenCLRenderer`: `__kernel void`, `__global `, `barrier(CLK_LOCAL_MEM_FENCE)`.
- `CUDARenderer`: `extern "C" __global__`, `__shared__`, `blockIdx`/`threadIdx`.
- `MetalRenderer`: `kernel void`, `device `, `threadgroup_barrier`.
- Vulkan uses a GLSL/SPIR-V renderer.

Vectorization is just `float4 = "(float4)"` and a vector prefix; tensor cores are
a `__{wmma_name}(a,b,c)` wrapper around inline PTX / builtins. The point for
Bitsy: **a small renderer + an op-code table replaces dozens of bespoke
shaders**. This is the single biggest line-count saving available.

## 7. Kernel speed: memory, tensor cores, indexing (developer/speed/)

Quoted guidance that drives our design:
- *"The main bottleneck in most kernels is accessing memory."* Arithmetic
  intensity = times a loaded value is used. GEMV intensity ~1 (memory-bound);
  GEMM/conv much higher. Splitting an axis into `UPCAST` raises intensity, but
  too much raises register pressure and spills.
- *"4090s have 1 TB/s of ram bandwidth and ~160 TFLOPS … you need to use each
  loaded value ~100 times. The L1 cache has ~40 TB/s … ~4 times."*
- Tensor cores create an n² compute-to-input ratio; GPUs use them to fit the
  warp model. tinygrad exposes them as a simple `WMMA` UOp.
- *"GPUs often have less integer math resources than floating point … indexing
  can be the bottleneck"* — hence symbolic simplification of index math.

OptOps (from `kernel.py`, referenced by the docs): `UPCAST`, `LOCAL`, `GROUP`,
`REDUCE`, plus split-K. BEAM search enumerates equivalent kernels and picks the
fastest.

## 8. Execution speed: the driver bypass (developer/speed/ + jit.py)

*"tinygrad has the best execution speed of any framework because it usually
bypasses the GPU driver and prebuilds the command queue. It's tons faster than
normal CUDA, and often even faster than CUDA Graph."*

`TinyJit` (`jit.py`) is a 3-state machine (`self.cnt`): 0 = ignore (run eagerly),
1 = **capture** (record the `LINEAR`s while `capturing` is set, then
`jit_lower`: substitute input buffers with `PARAM`s, mark unheld buffers as
scratch, memory-plan, compile), 2+ = **replay** (`run_linear(..., jit=True)` with
the same command queue and new input buffer pointers). Input buffers are
parametrized; aliased inputs get copy staging. This is the pattern Bitsy 0.2
must adopt to fix execution overhead.

## 9. What we port to Bitsy 0.2 (concrete decisions)

1. **One IR, not a dispatcher zoo.** A compact C `bt_node` (op code + shape +
   sources + scalar arg), interned by hash. Deletes `ops/` + most of
   `dispatcher.c` + `conv_dispatch.c`.
2. **Lazy graph + realize.** Python/eager `bt_*` calls build nodes; `bt_realize`
   schedules and runs. Deletes per-op host fallbacks.
3. **Fusion scheduler.** Collapse maximal pointwise subgraphs into one generic
   fused shader driven by an op-code program. This is the main speed win.
4. **One GEMM kernel + specialization constants** for tile sizes instead of 10
   shader variants (register-blocked, swizzle, k16, 128x64, 64x128 …). BEAM-lite:
   a small heuristic/autotune over tile configs.
5. **One reduce kernel**, parameterized.
6. **Rendered or parameterized kernels**: chosen approach — precompiled generic
   kernels parameterized by push constants / small op programs (no runtime GLSL
   compiler dependency), keeping the runtime tiny and CI-friendly.
7. **JIT capture/replay** of Vulkan command buffers with parametrized input
   buffers (tinygrad's driver bypass) for the training-step path.
8. **Compose, don't add**: activations/norms/losses from primitives; no new
   kernels for GELU/sigmoid/LayerNorm/softmax.

## 10. Honest scope

tinygrad is ~50k lines of Python including tests/models; its *conceptual core*
is small but its codegen is sophisticated (BEAM, manual bf16 casts, per-arch
WMMA inline asm). A C+Vulkan core under 4096 lines cannot replicate all of that.
We target the **philosophy and the parity-relevant subset**, and we will state
plainly which optimisations (full BEAM, runtime codegen, every WMMA variant)
are out of scope.
