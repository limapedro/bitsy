# Feature inventory — a PyTorch-grade framework

This is the **public API surface** needed for a professional-grade deep learning
framework, using PyTorch's names because PyTorch is Bitsy's numeric and
performance oracle. It covers representative functions, not every overload or
dtype variant (PyTorch's full count is in the thousands once those are included).

Status legend: **done** · **partial** · **missing**.
Per-function numeric and performance evidence lives in [PROGRESS.md](PROGRESS.md).

---

## 1. Tensor creation and conversion

| Representative API | Status |
|---|---|
| `tensor`, `as_tensor`, `from_numpy`, `frombuffer`, `from_torch` | partial (`from_numpy` only) |
| `zeros`, `zeros_like`, `ones`, `ones_like`, `empty`, `empty_like`, `full`, `full_like` | partial (C API: empty/zeros; `full` missing) |
| `arange`, `range`, `linspace`, `logspace`, `eye`, `identity` | missing |
| `rand`, `randn`, `randint`, `randperm`, `normal`, `uniform` | missing (RNG parity not implemented) |
| `tril`, `triu`, `diag`, `diagflat`, `meshgrid`, `indices`, `tile`, `repeat_interleave` | missing |
| `as_strided`, `numpy`, `tolist`, `item`, `cpu`, `vulkan`, `to`, `clone`, `contiguous` | partial (`numpy`, `cpu`/`vulkan` device strings, reshape view) |

## 2. Tensor structure, views, indexing

| Representative API | Status |
|---|---|
| `shape`, `stride`, `numel`, `ndim`, `element_size`, `is_contiguous` | partial (shape/strides/numel/contiguity in C) |
| `view`, `reshape`, `transpose`, `t` | done (contiguous `view` + 2D strided transpose; `Tensor.transpose` differentiable since exp_037) |
| `squeeze`, `unsqueeze`, `flatten` | missing |
| `permute`, `expand`, `expand_as`, `repeat`, `narrow`, `unbind`, `unfold`, `roll`, `flip` | missing |
| `cat`, `stack`, `hstack`, `vstack`, `dstack`, `column_stack`, `moveaxis`, `swapaxes` | missing |
| `index_select`, `gather`, `scatter`, `scatter_`, `index_add`, `index_copy`, `index_fill`, `put`, `take` | missing |
| `masked_select`, `masked_scatter`, `masked_fill`, `where`, `nonzero`, `argwhere` | missing |
| `__getitem__` / `__setitem__` (basic, advanced, boolean) | missing |

## 3. Elementwise math

| Representative API | Status |
|---|---|
| `add`, `sub`, `mul`, `div` | done (cpu_ref + Vulkan, broadcast) |
| `relu`, `sigmoid`, `tanh`, `exp`, `log` | done (cpu_ref + Vulkan) |
| `true_divide`, `floor_divide`, `mod`, `remainder`, `fmod`, `pow`, `rpow`, `neg`, `abs`, `sign` | partial (PR #2: `pow`/`neg`/`abs`/`sign` done; division variants, mod/fmod missing) |
| `sqrt` | done (exp_037: cpu_ref + Vulkan, backward `1/(2*sqrt(x))`; clamps negatives at 0, PyTorch returns NaN — documented in exp_037 FINDINGS) |
| `rsqrt`, `square`, `reciprocal`, `exp2`, `expm1`, `log2`, `log10`, `log1p`, `logaddexp` | partial (PR #9: `rsqrt`/`square`/`reciprocal`/`exp2`/`log2` done; `expm1`/`log10`/`log1p`/`logaddexp` missing) |
| `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh` | missing |
| `erf`, `erfc`, `erfinv`, `lgamma`, `gamma`, `digamma` | missing |
| `clamp`, `clip`, `round`, `floor`, `ceil`, `trunc`, `frac`, `nan_to_num`, `nan_to_num_` | partial (PR #9: `clamp`/`clip` done; rounding family missing) |
| `hypot`, `copysign`, `nextafter`, `lerp`, `addcmul`, `addcdiv` | partial (PR #9: `lerp` done; rest missing) |

## 4. Comparisons, logical, bitwise

| Representative API | Status |
|---|---|
| `eq`, `ne`, `gt`, `ge`, `lt`, `le`, `equal`, `greater`, `less` | done (PR #2 C primitives; PR #9 Python `Tensor` methods, 44-case parity) |
| `isclose`, `allclose`, `isnan`, `isinf`, `isfinite` | missing (only used inside the parity harness) |
| `logical_and`, `logical_or`, `logical_not`, `logical_xor`, `all`, `any` | partial (PR #9: four logical helpers done; `all`/`any` missing) |
| `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`, `left_shift`, `right_shift` | missing |

## 5. Reductions

| Representative API | Status |
|---|---|
| `sum`, `argmax` | partial (Python helper via matmul; NumPy argmax in eval) |
| `mean`, `median`, `prod`, `min`, `max`, `argmin`, `std`, `var`, `norm`, `logsumexp` | partial (PR #9: `std`/`var`/`norm`/`logsumexp` done; `median`/`prod`/`argmin` missing) |
| `topk`, `kthvalue`, `sort`, `argsort`, `cumsum`, `cumprod`, `cummax`, `cummin`, `diff` | missing |
| `unique`, `bincount`, `histc`, `bucketize`, `nonzero`, `count_nonzero` | missing |

## 6. Linear algebra (`torch.linalg` and classics)

| Representative API | Status |
|---|---|
| `matmul`, `mm`, `transpose` | done (2D, fp32; naive + tiled Vulkan GEMM) |
| `bmm`, `addmm`, `dot`, `vdot`, `outer`, `inner`, `einsum`, `tensordot` | missing |
| `solve`, `inv`, `pinv`, `lstsq`, `det`, `slogdet`, `matrix_rank`, `matrix_power` | missing |
| `norm`, `matrix_norm`, `vector_norm`, `lu`, `qr`, `svd`, `cholesky`, `eigh`, `eig` | missing |
| `cross`, `trace`, `kron`, `triangular_solve`, `householder` | missing |

## 7. RNG and distributions

| Representative API | Status |
|---|---|
| `manual_seed`, `seed`, `get_rng_state`, `set_rng_state`, `Generator` | missing |
| `rand`, `randn`, `randint`, `randperm`, `normal_`, `uniform_`, `bernoulli` | missing |
| `multinomial`, `poisson`, `exponential`, `log_normal`, `geometric` | missing |
| `torch.distributions.{Normal,Uniform,Bernoulli,Categorical,...}` | missing |

## 8. Autograd

| Representative API | Status |
|---|---|
| `requires_grad`, `grad`, `grad_fn`, `is_leaf`, `detach`, `backward` | partial (interim Python graph) |
| `no_grad`, `enable_grad`, `set_grad_enabled`, `inference_mode` | missing |
| `retain_grad`, `register_hook`, `autograd.grad`, `autograd.functional.*` | missing |
| `create_graph` (double backward), `gradcheck`, `gradgradcheck` | missing (external gradcheck script only) |

## 9. `nn.Module` infrastructure

| Representative API | Status |
|---|---|
| `Module`, `parameters`, `children`, `modules`, `__call__` | partial |
| `state_dict`, `load_state_dict`, `train`, `eval`, `to`, `apply`, `zero_grad` | missing |
| `register_buffer`, `register_parameter`, forward/backward hooks | missing |

## 10. nn layers

| Representative API | Status |
|---|---|
| `Linear`, `Conv2d`, `Sequential`, `Flatten` | done (minimal) |
| `ReLU`, `Tanh`, `Sigmoid` | done |
| `Conv1d`, `Conv3d`, `ConvTranspose*`, `Bilinear`, `LazyLinear` | missing |
| `MaxPool*`, `AvgPool*`, `Adaptive*Pool*`, `MaxUnpool*`, `LPPool*` | missing |
| `LayerNorm` | done (exp_037: last-dim, eps=1e-5, fwd/grads match PyTorch ~1e-6) |
| `BatchNorm*`, `GroupNorm`, `InstanceNorm*`, `RMSNorm` | missing |
| `RNN`, `LSTM`, cells | done (exp_038: unidirectional, copy-in weights; separate gate Parameters, packed layout not emulated) |
| `GRU`, `PackedSequence` | missing |
| `MultiheadAttention`, `Transformer*` | partial (exp_039: per-head projections + masked attention composed in an example GPT-2, not a reusable nn module) |
| `Embedding` | done (exp_037: one-hot matmul lookup, dense weight grad) |
| `Dropout` | partial (eval/identity only; no training mask) |
| `EmbeddingBag`, `Upsample`, `PixelShuffle`, `ModuleList/Dict` | missing |

## 11. Activations and losses (`nn.functional`)

| Representative API | Status |
|---|---|
| `relu`, `sigmoid`, `tanh`, `softmax` (via `log_softmax` internally) | done (exp_037: softmax dx 1.7e-7) |
| fused `add + relu` pointwise kernel | done |
| `gelu`, `log_softmax` | done (exp_037: exact-erf gelu fwd/bwd 1.2e-7; `log_softmax` now handles >256 cols) |
| `silu`, `mish`, `leaky_relu`, `elu`, `selu`, `softplus`, `glu` | partial (PR #9: six activations done incl. hardswish/hardtanh; `glu` missing) |
| `mse_loss`, `cross_entropy` | done |
| `nll_loss`, `l1_loss`, `smooth_l1_loss`/`huber_loss`, `bce_loss`, `bce_with_logits_loss` | done (PR #9: all five, pure Python compositions, fwd+grad parity) |
| `kl_div`, `poisson_nll_loss`, `hinge_loss`, `triplet_margin_loss`, `ctc_loss` | missing |

## 12. Optimizers and schedulers

| Representative API | Status |
|---|---|
| `SGD` | done (plain SGD) |
| `AdamW` | done (exp_037: betas/eps/decoupled weight decay, bias correction; one-step match 1.5e-8) |
| SGD momentum/Nesterov/weight decay, `Adam`, `RMSprop`, `Adagrad`, `Adadelta`, `LBFGS`, `NAdam` | missing |
| `lr_scheduler.*` (`StepLR`, `CosineAnnealingLR`, `OneCycleLR`, `ReduceLROnPlateau`, ...) | missing |
| `clip_grad_norm_`, `clip_grad_value_`, fused update kernels | missing |

## 13. Data loading

| Representative API | Status |
|---|---|
| `Dataset`, `DataLoader`, `Sampler`, `random_split`, `Subset`, `collate_fn`, `num_workers` | missing |
| Vision/text datasets and transforms (or interop with them) | missing (training scripts read NumPy arrays) |

## 14. Serialization and interop

| Representative API | Status |
|---|---|
| `save`, `load`, `state_dict` files, safetensors, PyTorch checkpoint import/export | missing |
| `torch.jit.*`, `torch.onnx.export`, `torch.fx`, `torch.compile` | missing |
| DLPack, buffer protocol, `from_torch`/`to_torch` | missing |

## 15. Device, streams, performance

| Representative API | Status |
|---|---|
| `device`, `cpu`, `vulkan`, `torch.cuda.synchronize` equivalent | partial (device strings, `bt_vulkan_flush()`, fence waits) |
| `Stream`, `Event`, async scheduler | missing |
| timeline semaphore submission (`bt_vulkan_flush` / `bt_vulkan_wait`) | partial |
| `autocast`, `GradScaler`, AMP, mixed precision | missing |
| `torch.compile`, `torch.profiler`, `torch.utils.benchmark` | missing |
| GPU timestamp queries | done (`bt_vulkan_device_time_ms()`) |
| deterministic algorithms, `set_num_threads`, matmul precision flags | missing |

## 16. Distributed

`torch.distributed`, process groups, `init_process_group`, `DistributedDataParallel`,
`DataParallel`, `FSDP`, `all_reduce`, `all_gather`, `broadcast`, `reduce_scatter`,
`send`/`recv`, RPC, sharded checkpointing. **Status: missing / out of scope until the core is solid.**

## 17. FFT, sparse, quantized, special

`torch.fft.*`, `torch.sparse.*`, complex tensors, `torch.ao.quantization`,
`torch.quantization`, QAT, int8/uint8 compute kernels. **Status: missing / out of scope.**

---

## Where to look next

- [PROGRESS.md](PROGRESS.md) — per-function implementation status, numeric
  differences versus PyTorch, and measured performance ratios.
- [BENCHMARKS.md](../BENCHMARKS.md) — full benchmark tables and honesty rules.
- [ARCHITECTURE_LOG.md](../ARCHITECTURE_LOG.md) — why decisions were made.
- [ROADMAP.md](../ROADMAP.md) — milestone sequence.
