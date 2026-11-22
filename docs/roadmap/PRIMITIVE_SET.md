# The frozen primitive set — why the core never grows (v0.2)

This is the architectural contract that makes "map every function, add zero C
lines" possible. The entire C+Vulkan core is a **fixed vocabulary** of low-level
primitives. PyTorch's hundreds of functions and layers are all *compositions* of
these primitives, written in the lean Python library. Adding a new activation,
layer, or loss is a Python change only — **the C line count is constant**.

> Rule (enforced by `tools/count_lines.py`): a PR that adds a new C kernel or a
> new C dispatch function is rejected unless it replaces an existing primitive.
> New functionality goes in `python/bitsy/` as a composition.

This document defines every primitive, its exact opcode/semantics, and its
kernel. `MAP_TENSOR.md`, `MAP_NN.md` and `MAP_TRAINING.md` map every PyTorch
function onto these primitives.

---

## 1. The complete primitive inventory (~45 opcodes, 6 shaders)

### 1.1 Fused ALU op-program (`fused.comp`) — one shader, all elementwise math

Every pointwise operation, unary or binary, is an opcode interpreted by ONE
compute shader that walks an op-program per element. Adding a new elementwise
function = adding an opcode to this table (rare) or, usually, composing existing
opcodes (free).

| Op | Mnemonic | Semantics | Grad |
|---|---|---|---|
| 0 | `LOAD` | read input slot at broadcast-adjusted offset | — |
| 1 | `STORE` | write result | — |
| 2 | `CONST` | push scalar immediate | — |
| 3 | `ADD` | a + b | ✓ |
| 4 | `SUB` | a − b | ✓ |
| 5 | `MUL` | a · b | ✓ |
| 6 | `DIV` | a / b | ✓ |
| 7 | `NEG` | −a | ✓ |
| 8 | `POW` | a^b | ✓ |
| 9 | `MIN` | min(a,b) | tie |
| 10 | `MAX` | max(a,b) | tie |
| 11 | `CMPLT` / `CMPLE` / `CMPNE` / `CMPEQ` | compare → 0/1 | ✗ |
| 12 | `WHERE` | c ? a : b | branches only |
| 13 | `EXP` | exp(a) | ✓ |
| 14 | `LOG` | log(a) | ✓ |
| 15 | `SQRT` | sqrt(max(a,0)) | ✓ |
| 16 | `RECIP` | 1/a | ✓ |
| 17 | `SIN`/`COS`/`TAN` | trig | ✓ |
| 18 | `ABS` | |a| | tie |
| 19 | `SIGN` | sign(a) | ✗ |
| 20 | `FMA` | a·b + c (fused) | ✓ |
| 21 | `CAST` / `BITCAST` | dtype conversion | identity |

### 1.2 Materializing kernels (one each)

| Kernel | Parameters | Covers |
|---|---|---|
| `gemm` | M,N,K, transposeA/B, alpha,beta, bias, accumulate | matmul, linear, bmm, dot, bilinear (via 2 GEMMs), attention scores/values |
| `reduce` | op ∈ {ADD, MUL, MAX, MIN, ANY, ALL}, axis, keepdim, indexed | sum/prod/max/min/mean/var/std/argmax/argmin/logsumexp |
| `scan` | op ∈ {ADD, MUL, MAX, MIN}, axis, reverse | cumsum/cumprod/cummax/cummin/logcumsumexp |
| `index` | mode ∈ {gather, scatter-add, scatter-mul, onehot, masked-select} | indexing, embedding, scatter, one_hot, topk-ish |
| `rng` | kind ∈ {uniform, normal, randint}, seed, counter, shape | all creation/random/dropout masks/implicit init |

### 1.3 Free movement opcodes (no kernel — stride/descriptor edits, K0)

`reshape`, `permute`, `transpose`, `expand`/`broadcast`, `pad`, `shrink`, `flip`,
`squeeze`/`unsqueeze`, `flatten`, `contiguous`. These never launch a shader;
they are folded into the index math of the consumer kernel.

### 1.4 Graph/meta opcodes (no kernel)

`ALLOC`, `COPY` (device copy), `CONTIGUOUS`, `ASSIGN`/`AFTER` (mutation edge for
optimizer), `scalar`/`const`.

**Total: ~45 opcodes across 6 shader kernels.** Every number below is fixed for v0.2.

---

## 2. The composition rules (how Python maps functions → primitives)

Every PyTorch function is exactly one of four composition shapes:

1. **Direct primitive** — one opcode, e.g. `torch.relu` → `MAX(x,0)`.
2. **Fused expression** — several ALU opcodes in one `fused.comp` program,
   e.g. `silu(x) = x * sigmoid(x)` → `MUL(x, DIV(CONST(1), ADD(CONST(1), EXP(NEG(x)))))` → **1 dispatch**.
3. **View chain** — movement opcodes only, e.g. `torch.transpose` → `permute`.
4. **Short kernel sequence** — 2–4 primitives, e.g. attention = GEMM→softmax→GEMM.

The scheduler fuses rule-2 chains automatically; the Python author never writes
a shader. This is why the Python library stays lean (each function is a few
lines) and fast (fusion keeps dispatch count low).

---

## 3. Budget proof

| Layer | Lines (fixed) | Grows when adding a new function? |
|---|---:|---|
| `core/` device/arena/shape/error | ~400 | no |
| `graph/` node + lazy ops + views | ~500 | no (new op = opcode entry, not new path) |
| `sched/` fusion + buffer planner | ~350 | no |
| `codegen/` 6 shaders + encode | ~550 | no |
| `exec/` Vulkan device + capture/replay | ~700 | no |
| **C+Vulkan core total** | **~2500** | **constant** (cap 4096) |
| `python/bitsy/` | lean, ~1500 | yes — but Python, not counted in C budget |

Adding e.g. `torch.mish` = one line of Python composition
`x * tanh(softplus(x))`; core unchanged. This is the entire point of the design.

---

## 4. Coverage claim

The three map documents enumerate **every** function in PyTorch's tensor,
`nn.functional`, `nn`, autograd and optim surfaces that is in scope for numeric
parity, each tagged with its primitive composition, dispatch count, backward
rule, and parity test. Functions out of scope (quantization, distributed,
data-dependent compaction under JIT) are listed explicitly as such — not
silently omitted. Counts are printed at the end of each map so coverage is
auditable.
