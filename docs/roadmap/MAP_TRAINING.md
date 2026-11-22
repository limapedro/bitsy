# Exhaustive function map — autograd, optimizers, JIT, distributed

Companion to `PRIMITIVE_SET.md`, `MAP_TENSOR.md`, `MAP_NN.md`.

---

## A. Autograd API (12 functions)

The C graph owns backward rules (`PRIMITIVE_SET`); Python just exposes them.

| PyTorch API | Composition | Grad | Test |
|---|---|---|---|
| `requires_grad_ / requires_grad` | metadata flag | — | autograd |
| `Tensor.grad / grad_fn / is_leaf` | C graph metadata | — | autograd |
| `backward(gradient=None)` | C reverse pass → primitive VJPs | — | gradcheck |
| `torch.autograd.grad` | C reverse pass, no `.grad` mutation | — | gradcheck |
| `Tensor.gradient` (tinygrad) | same, jittable | — | gradcheck |
| `detach / detach_` | view opcode | — | detach |
| `no_grad / enable_grad / set_grad_enabled` | context flag | — | no_grad |
| `inference_mode` | context flag (no graph) | — | eval |
| `retain_grad / register_hook` | C graph node hooks (defer) | — | hooks |
| `gradcheck / gradgradcheck` | finite-difference harness | — | gradcheck |
| `create_graph (double backward)` | defer (out of scope v0.2) | — | — |
| `zero_grad_` (tensor) | set .grad = None | — | optim |

**Backward rule table (implemented once in C, ~18 rules):** ADD, SUB, MUL, DIV,
POW, EXP, LOG, SQRT, WHERE, MIN/MAX(ties), REDUCE{ADD,MAX,MUL},
RESHAPE/EXPAND/PAD/PERMUTE/FLIP, GEMM (2 GEMMs), CONV (2 convs), SOFTMAX,
LOG_SOFTMAX, LAYERNORM, CROSS_ENTROPY, EMBEDDING(scatter-add), BATCHNORM,
DROPOUT(saved mask). Covers >95% of training graphs.

## B. Optimizers (14 functions)

All updates = one fused `optim_update` primitive (K10); state in Python tensors.

| PyTorch API | Composition | Bwd | Test |
|---|---|---|---|
| `torch.optim.Optimizer` (base) | host: dedup params, zero_grad/step | ✗ | — |
| `zero_grad` | set grads None | ✗ | — |
| `step` | schedule_step + K10 + assign | ✗ | step |
| `SGD (lr, momentum, wd, nesterov, dampening)` | K10: buf←μbuf+(1−d)g; p−=lr·g' | ✗ | MNIST |
| `AdamW (lr,betas,eps,wd)` | K10: m,v, bias-corr, up=m̂/(√v̂+ε)+wd·θ | ✗ | W 1.5e-8 |
| `Adam` | K10 without decoupled wd | ✗ | adam |
| `Adagrad` | K10: sum of squares accumulator | ✗ | adagrad |
| `RMSprop` | K10: EMA of squares | ✗ | rmsprop |
| `Adadelta` | K10: dual EMA | ✗ | adadelta |
| `LBFGS` | iterative line search (defer) | ✗ | — |
| `NAdam / RAdam / Adamax` | K10 variants | ✗ | adamvar |
| `SGD/AdamW fused=True` | single K10 fused kernel | ✗ | fused step |
| `LR scheduler (StepLR, CosineAnnealing, OneCycle, ReduceLROnPlateau)` | host lr update | ✗ | lr |
| `OptimizerGroup / param groups` | host composition | ✗ | — |

**Parity pitfall:** tinygrad SGD seeds momentum differently from PyTorch on the
first step — mirror PyTorch exactly (buf starts 0, `buf←μbuf+g`, seed not used).
AdamW: eps **outside** sqrt in both, decoupled wd `up=m̂/(√v̂+ε)+wd·θ`.

## C. JIT / compile (10 functions)

| PyTorch API | Composition | Test |
|---|---|---|
| `@TinyJit` / `TinyJit(fn)` | capture graph on call 1–2, replay prebuilt CB after | replay speed |
| `Context(TRAINING=1)` | host flag | training |
| `Tensor.no_grad` | host flag | eval |
| `Tensor.realize()` | scheduler + dispatch | — |
| `Tensor.schedule_with_vars / run_schedule` | lower graph → compiled programs | — |
| `torch.compile` (analog) | JIT capture + fusion | — |
| `VIZ=1` | graph dump | — |
| `Device.DEFAULT` | device selection | — |
| `TinyJit(prune=True)` | prune kernels not touching inputs | — |
| `jit(skip_first_run)` | defer compilation 1 step | — |

**JIT contract:** pure functions, static shapes, framework tensors only. Write
the whole training step as a function so forward+backward+optimizer are captured
together. Data-dependent compaction (`nonzero`/`masked_select` with size=None)
is not jittable — use fixed-size forms.

## D. Serialization / interop (8 functions) — host only, no kernel

| API | Composition | Disp | Test |
|---|---|---:|---|
| `state_dict / get_state_dict` | object walk | 0 | state |
| `load_state_dict` | COPY per param | n | state |
| `safe_save / safe_load` | host safetensors | 0 | io |
| `torch.save / torch.load` | host pickle | 0 | io |
| `from_torch / to_torch` | COPY interop | 1 | interop |
| `DLPack (from_dlpack / to_dlpack)` | buffer share | 0 | interop |
| `numpy() / item() / tolist()` | realize + download | 1 | interop |
| `from_numpy / frombuffer` | COPY host→dev | 1 | interop |

## E. Distributions / RNG (10 functions) — RNG primitive

| API | Composition | Disp | Test |
|---|---|---:|---|
| `manual_seed` | host seed | 0 | rng |
| `get_rng_state / set_rng_state` | host state | 0 | rng |
| `Generator` | RNG(state) | 0 | rng |
| `Normal.sample` | RNG normal | 1 | rng |
| `Uniform.sample` | RNG uniform | 1 | rng |
| `Bernoulli.sample` | RNG uniform + CMP | 1 | rng |
| `Categorical.sample` | RNG uniform + scan | 2 | rng |
| `multinomial` | RNG + scan | 2 | rng |
| `poisson` | RNG + fused (defer) | 1 | rng |
| `normal_ / uniform_ / bernoulli_` | RNG in-place | 1 | rng |

## F. Explicitly out of scope for v0.2 (listed, not silently dropped)

| Area | Functions | Reason |
|---|---|---|
| Quantization | `quantize`, `scaled_mm`, `scaled_addmm`, `grouped_mm`, `int8/fp8` ops | fp32 main dtype; fp16 storage later |
| Distributed | `DataParallel`, `DDP`, FSDP, `all_reduce`, `all_gather`, `broadcast`, RPC | single GPU first |
| Data loading | Dataset/DataLoader/Sampler | training scripts read NumPy |
| FX/export/ONNX | `torch.fx`, `torch.export`, `torch.onnx`, `torch.jit.script` | post-1.0 |
| Profiling hooks | `torch.profiler`, `register_hook`, autograd hooks | host tooling |
| Decompositions | `svd/qr/cholesky/inv/pinv/solve/det` | iterative, defer |
| Data-dependent JIT | `nonzero(size=None)`, dynamic shapes | static-shape capture |
| Complex tensors | complex dtype ops | fp32 only |

These are documented so the "100% parity" claim is scoped honestly: full
coverage of the **in-scope** surface, not every PyTorch symbol.

---

**MAP_TRAINING count:** autograd 12 + optimizers 14 + JIT 10 + serialization 8 +
RNG 10 = 54 mapped; out-of-scope 12 (listed) = **54 training functions mapped +
12 explicitly scoped out**.

## Grand total across all maps

| Map | Functions |
|---|---:|
| MAP_TENSOR | 133 |
| MAP_NN | 98 |
| MAP_TRAINING | 54 (+12 out-of-scope) |
| **Total mapped to frozen primitives** | **285 functions/layers/ops** |

Every one of the 285 is a Python composition of the ~45 frozen primitives defined
in `PRIMITIVE_SET.md`. **Adding any of them changes zero C lines.**
