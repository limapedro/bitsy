# Study: PyTorch 2.15 — design philosophy and semantics

Sources (read 2026-11-22):
- https://docs.pytorch.org/docs/2.15/index.html
- https://docs.pytorch.org/docs/2.15/community/design.html (Design Philosophy)
- Section links captured from the 2.15 reference index (autograd, Tensor views,
  broadcasting semantics, numerical accuracy, CUDA semantics, torch.compile,
  torch.optim, nn / nn.functional, C++ / accelerator notes).

This is a study record; it informs what "numeric parity with PyTorch" means for
Bitsy 0.2. Oracle in this repo: PyTorch 2.11.0+cu126 (the installed version);
the 2.15 docs are the current published semantics and are stable/reference for
the APIs we use.

## 1. Design philosophy (community/design.html)

Three principles, quoted:

1. **Usability over Performance.** *"PyTorch's primary goal is usability. A
   secondary goal is to have reasonable performance."* It deliberately avoids
   *"restriction-first regimes (for example, static shapes, graph-mode only)."*
   This is the opposite of tinygrad's laziness-by-default and explains why eager
   PyTorch is fast despite Python: enormous engineering in C++ dispatch.
2. **Simple Over Easy.** Borrowing Zen of Python: *"Explicit is better than
   implicit; Simple is better than complex."* Example: every tensor has an
   explicit device; cross-device ops error at the point of invocation rather
   than silently placing. *"Focusing on simple lower-level building blocks."*
3. **Python First with best-in-class language interoperability.** *"PyTorch is
   not a Python binding into a monolithic C++ framework. It is built to be
   deeply integrated into Python."* Python overhead was attacked by rewriting
   autograd in C++, then operator definitions, then TorchDynamo/torch.compile.

**Derivation for Bitsy:** we cannot out-engineer PyTorch's dispatch. tinygrad's
lesson is to avoid the overhead by *not dispatching per op* (lazy + fusion +
JIT). Our numeric target is PyTorch; our performance *strategy* is tinygrad's.

## 2. What "numeric parity" actually requires

The 2.15 reference highlights the exact semantics our parity gate encodes:

- **Broadcasting semantics** (`notes/broadcasting.html`): a dimension is
  compatible when equal or 1; the smaller tensor is virtually expanded. Our
  `bitsy_broadcast_shapes` + broadcast_grad mirror this; mismatches must raise.
- **Tensor Views** (`tensor_view.html`): `view`/`reshape` share storage; a view
  with non-unit stride (e.g. transpose) must be read with stride-aware indexing.
  exp_037/GPT-2 found exactly this: the register-blocked GEMM ignored strides and
  produced a zero gradient. Parity means stride correctness, not just shape.
- **Numerical accuracy** (`notes/numerical_accuracy.html`): fp32 matmul on GPU
  accumulates in a different order than CPU/cuBLAS, so *exact bit equality is
  not a valid cross-backend target*. Our `tolerances.py` encodes this: EXACT for
  add/sub/mul/div/relu, rtol/atol 1e-5 for transcendentals and matmul.
- **CUDA semantics** (`notes/cuda.html`): device tensors, async execution,
  `torch.cuda.synchronize()` before timing. Our benchmarks mirror this with
  `bt.flush()` + `bt.wait()` and median-of-N timing.
- **Autograd mechanics** (`notes/autograd.html`): dynamic graph, `requires_grad`,
  `grad`, backward formulas. Our Python `Tensor._op` graph mirrors this; the C
  core does not yet own it (M4 goal).

## 3. API surface we treat as the parity contract

From the 2.15 index, the representative APIs relevant to our implemented subset:

- `torch.Tensor`: creation (`from_numpy`), views (`reshape`, `transpose`), math
  (`add/sub/mul/div/relu/sigmoid/tanh/exp/log/sqrt/gelu`), reductions.
- `torch.nn` / `torch.nn.functional`: `Linear`, `LayerNorm`, `Embedding`,
  `RNN`/`LSTM` cells, `gelu`, `softmax`, `log_softmax`, `cross_entropy`,
  `mse_loss`.
- `torch.autograd`: `backward`, gradients.
- `torch.optim`: `SGD`, `AdamW`.
- `torch.compile` / CUDA graphs: the performance analogue of tinygrad's JIT.

Tolerances are the contract boundary: we do **not** claim full API coverage;
we claim the implemented functions match on tested shapes. This matches the
scoping in `experiments/exp_037/HYPOTHESIS.md`.

## 4. Concrete parity facts already established in this repo

From `tests/parity/reports/latest.json` and exp_037–039 (all committed):

| op / model | tolerance class | observed max diff |
|---|---|---|
| add/sub/mul/div/relu | EXACT | 0 (fp32 bit-exact) |
| sigmoid/tanh/exp/log/sqrt/gelu | APPROX 1e-5/1e-6 | ≤1.2e-7 |
| matmul | rtol/atol 1e-5 | ≤4.8e-7 (grad), 1.16e-6 (GPT-2 logits) |
| LayerNorm | APPROX | ≤5.4e-7 |
| AdamW step | APPROX | 1.5e-8 (W), 9.3e-10 (b) |
| RNN h / LSTM h,c | 1e-5 | 5.9e-6 / 2.1e-6, 3.1e-6 |
| GPT-2 loss | 1e-3 | 0.0 |

Divergence we must preserve or document: Bitsy `sqrt` clamps negatives at 0
while `torch.sqrt` returns NaN (gate uses positive inputs; exp_037 FINDINGS).

## 5. Performance reference points (PyTorch side)

- PyTorch eager dispatch is fast in C++; tinygrad beats it by bypassing the
  driver entirely. Our legacy eager ratios are 0.04–0.71 (models), 0.41–0.53
  (elementwise), captured step ~0.46 (BENCHMARKS.md).
- CUDA graphs (`torch.cuda.CUDAGraph`) are PyTorch's capture/replay, analogous
  to tinygrad's `TinyJit`. Our Vulkan command-buffer replay is the equivalent.

## 6. Decisions carried into the 0.2 rewrite

1. Keep the EXACT/APPROX tolerance split; never relax to hide a bug.
2. Preserve stride/view correctness in the new lazy graph (transpose must be a
   view op with strides, materialised only when a kernel consumes it).
3. Keep autograd semantics: `_op`-style backward, grad accumulation, unbroadcast.
4. Preserve the parity gate as the acceptance test for the rewrite.
5. Benchmark with correct sync, warmup, median + spread, GPU clock — matching
   PyTorch's own CUDA-semantics guidance.
