# Python library — complete function map (v0.2 roadmap)

The Python package mirrors PyTorch's numeric surface with tinygrad's lazy,
functional design. It is a thin layer over the C ABI in
`C_VULKAN_CORE.md`: every Python op constructs a lazy C node; `realize()` is the
only execution trigger. This is a planning doc — no code on this branch.

Design rules (from `../research/FRAMEWORKS.md`):
- **Functional**: stateless ops are functions; classes only hold parameters.
  No `nn.Module` base class — `get_parameters(obj)` walks plain objects
  (tinygrad). `__call__` replaces `forward`.
- **Lazy + JIT**: ops build a graph; `realize()` runs it; `jit()` captures and
  replays the command queue (§JIT).
- **fp32 default**, dataclass configs, no argparse (frozen dataclass = JIT-stable).
- **Parity contract**: every function validated against PyTorch within
  `../FEATURE_PARITY_MATRIX.md` tolerances.

### Kernel vocabulary (the C+Vulkan target for every op below)

| Token | Primitive kernel | Notes |
|---|---|---|
| K0 | none / metadata | reshape, transpose, expand, squeeze — stride edits, no dispatch |
| K1 | `ew_unary(op)` | one thread per element (EXP/LOG/SQRT/RELU/GELU/SIGMOID…) |
| K2 | `ew_binary(op)` | broadcast-aware ADD/MUL/DIV/POW/WHERE/CMP/MIN/MAX |
| K3 | `reduce(op, axis)` | workgroup ADD/MAX/…; argmax/argmin use indexed MAX/MIN |
| K4 | `gemm(A,B,C,flags,bias)` | tiled matmul (+ fused bias); gemv for 1-D |
| K5 | `conv` / im2col+K4 | convolution; pooling is a small variant |
| K6 | `softmax_fwd/bwd` | fused max→exp→sum→div; log-softmax variant |
| K7 | move/copy/gather/scatter | strided copy, concat, one-hot, padding |
| K8 | `rng(seed,counter,shape)` | counter-based RNG (Threefry/Philox) |
| K9 | `scan(op,axis)` | prefix scan for cumsum/cumprod |
| K10 | `optim_update(p,g,state,cfg)` | fused SGD/AdamW step |
| FUSE | fused op-program | several primitives in one shader (GELU = tanh+mul fused) |

Nearly every op below lowers to **≤2 dispatches**, matching tinygrad's primitive
vocabulary (ALU + REDUCE + GEMM/WMMA + 5 movement ops).

### Numeric-parity pitfalls to encode (from research)

- **GELU flavor**: tinygrad defaults to `approximate='tanh'`, PyTorch to exact
  erf — force the same on both (our v0.2 uses exact erf).
- **CE axis/ignore_index**: tinygrad class axis −1, PyTorch dim 1; ignore −1 vs
  −100; align before comparing.
- **Eps values**: LayerNorm 1e-5, RMSNorm 1e-6, AdamW 1e-8, BatchNorm 1e-5.
- **Reduction order**: Vulkan workgroup reductions differ from cuDNN/MKL —
  assert with tolerances (fp32 atol 1e-5 / rtol 1e-3), never exact equality.
- **Stable math**: softmax/log_softmax subtract max; BCE-logits uses
  `max(x,0)−xy+log(1+e^−|x|)`; log-sigmoid/softplus guard large magnitudes.
- **Optimizer semantics**: tinygrad AdamW = LAMB(adam=True) matching PyTorch
  AdamW with eps outside sqrt; tinygrad SGD differs from PyTorch SGD on first
  step (momentum buffer seeding) — mirror one framework, not a blend.
- **RNG**: algorithms differ across frameworks — don't promise bitwise streams;
  inject fixed tensors for parity tests.

---

## 1. Tensor — creation & conversion

| Function | Purpose | C primitive(s) |
|---|---|---|
| `Tensor(data, device=, dtype=f32, requires_grad=)` | wrap/import array | `bt_from_host` |
| `Tensor.zeros/ones(full)_like(*shape)` | constant tensors | `bt_zeros` + fill |
| `Tensor.from_numpy(arr)` / `.numpy()` | interop | `bt_buffer_upload/download` |
| `.device`, `.shape`, `.dtype` | metadata | `bt_shape_*` |
| `.realize()` | schedule + execute graph | `bt_realize` |
| `.detach()`, `.clone()`, `.to(device)` | graph/device control | view + copy |

## 2. Tensor — movement / views (free, no copy)

| Function | Purpose | C primitive |
|---|---|---|
| `.reshape(*shape)`, `.view(*shape)` | reinterpret shape | `bt_reshape` (view) |
| `.transpose()`, `.T` | 2D transpose | `bt_transpose` (view) |
| `.expand()`, `.squeeze()/unsqueeze()` | broadcast/axes views | view strides |
| `.flatten()` | flatten leading dims | `bt_reshape` |

## 3. Tensor — elementwise math (all fused, all with backward except noted)

| Function | Purpose | C primitive | Backward |
|---|---|---|---|
| `+ - * /`, `neg` | arithmetic (broadcast) | fused ADD/SUB/MUL/DIV | yes |
| `exp, log, sqrt` | transcendental | fused EXP/LOG/SQRT | yes |
| `tanh, sigmoid, relu, gelu` | activations | fused TANH/SIGM/RELU/GELU | yes |
| `maximum/minimum, where` | select | fused op-program | yes |
| comparison `== != < >` | masks | fused op-program | no |

## 4. Tensor — reductions & matmul (cut kernels)

| Function | C primitive | Backward |
|---|---|---|
| `.sum(axis,keepdim)`, `.mean()` | `bt_reduce_sum` (+ div) | broadcast g |
| `.max()/argmax()` | reduce (max) | argmax inference only |
| `.matmul()`, `@` | `bt_matmul` → gemm | g·Bᵀ, Aᵀ·g |
| `.dot()`, `.outer()` | gemm shapes | yes |

## 5. Tensor — numeric parity cases covered

Every row of `FEATURE_PARITY_MATRIX.md` maps here: add/sub/mul/div (EXACT),
unary math (1.2e-7), sqrt/gelu (1.2e-7), matmul fwd/grad (4.8e-7),
softmax/log_softmax (3e-8), reduce/transpose (1e-6), CE/SGD (1e-6),
model losses (≤1e-3).

## 6. autograd

| API | Purpose | Notes |
|---|---|---|
| `requires_grad`, `.grad`, `.backward(gradient=None)` | reverse-mode | C graph owns backward; Python just forwards |
| `no_grad()` context | disable graph building | sets a flag on node construction |
| `gradcheck(fn, x)` | finite-difference check | external test helper |

Backward rules are implemented once in C (mirroring the verified autograd.py
formulas): quotient rule for div, `1−out²` tanh, `out(1−out)` sigmoid, `gt0`
relu, `erf+phi` gelu, transpose/matmul/reduce as in `C_VULKAN_CORE.md`.

**Minimal backward set to implement first** (covers >95% of transformer/CNN
training graphs): `ADD, MUL, SUB, DIV, POW, EXP, LOG, SQRT, WHERE, MAX/MIN
(ties), REDUCE{ADD,MAX,MUL}, RESHAPE/EXPAND/PAD/PERMUTE, COPY, GEMM (two
GEMMs), CONV (two convs), SOFTMAX/LOG_SOFTMAX, LAYERNORM, CROSS_ENTROPY,
EMBEDDING scatter-add`. **Inference-only (no grad)**: creation, RNG, comparisons,
`argmax/argmin/nonzero/masked_select`, `one_hot`, `triu/tril`, optimizer state
mutations, `detach`/`no_grad`/JIT wrappers, parameter discovery.

## 7. `nn` modules (classes hold Parameters only)

| Module | Composition of primitives | Parity evidence |
|---|---|---|
| `Linear(in,out,bias)` | `x @ W + b` | W grad 6e-8 |
| `Conv2d` (phase later) | im2col/gemm or direct | CIFAR logits 1.2e-5 |
| `LayerNorm(n,eps)` | reduce mean/var + fused affine | ≤5.4e-7 |
| `Embedding(n,d)` | one-hot matmul; dense weight grad | forward 0 |
| `Dropout` (eval identity) | passthrough | eval parity |
| `ReLU/Tanh/Sigmoid/GELU` | unary ops | as ops |
| `RNNCell/RNN`, `LSTMCell/LSTM` | gate matmuls + sigmoid/tanh; step selection | h 5.9e-6, c 3.1e-6 |
| attention block | per-head Q/K/V/O projections + masked softmax (composed) | GPT-2 logits 1.16e-6 |
| `Sequential(*layers)` | compose `__call__` | MLP |

## 8. `functional` / losses (stateless functions)

| Function | C composition | Parity evidence |
|---|---|---|
| `softmax, log_softmax` | exp/reduce/log fused | 3e-8 |
| `cross_entropy(logits, labels)` | log_softmax + gather/reduce | GPT-2 loss 0.0 |
| `mse_loss(pred,target)` | reduce(mean((p−t)²)) | MLP |

## 9. `optim` (stateful, over parameter tensors)

| Optimizer | Update | Parity evidence |
|---|---|---|
| `SGD(params, lr)` | `p -= lr*g` | MNIST identical |
| `AdamW(params, lr, betas, eps, wd)` | bias-corrected moments + decoupled decay | W 1.5e-8, b 9.3e-10 |
| `.zero_grad()`, `.step()` | standard | — |

## 10. JIT / compile

| API | Purpose | C primitive |
|---|---|---|
| `jit(fn)` (decorator) | capture ops on first calls, replay command queue after | `bt_capture_begin/end`, `bt_replay` |
| `get_parameters(obj)` | recursively collect Parameter tensors from plain objects | Python only |

JIT is where speed comes from: replay is one `vkQueueSubmit2`, no per-op Python
or driver work (see `VULKAN_PERFORMANCE.md` §3).

## 11. Package layout

```
python/bitsy/
  __init__.py        # public exports: Tensor, nn, optim, functional, jit, ...
  tensor.py          # Tensor class + ops, autograd hooks
  _ctypes.py / _ffi  # C ABI binding (thin)
  nn.py              # modules (plain classes), get_parameters
  optim.py           # SGD, AdamW
  functional.py      # activations, losses
```

## 12. Implementation order (ties to `ROADMAP.md` phase 2)

1. `_ffi` + Tensor creation/views + realize/to_host — parity on add.
2. Elementwise + unary + backward — 48-case op gate logic (Python side).
3. matmul/reduce + rest of backward — GEMM/reduction parity.
4. nn (Linear, activations, LN, Embedding, RNN/LSTM) + losses — model parity.
5. AdamW + training-step parity — MNIST/CIFAR/GPT-2.
6. `jit()` — replay benchmarks.

Sources: tinygrad `Tensor` API and nn (https://docs.tinygrad.org/tensor/,
https://docs.tinygrad.org/nn/), PyTorch 2.15 `torch.nn`/`torch.nn.functional`/
`torch.optim` (https://docs.pytorch.org/docs/2.15/), JAX 101 transformations
(https://docs.jax.dev/en/latest/101/).
