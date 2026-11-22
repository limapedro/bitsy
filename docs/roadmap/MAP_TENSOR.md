# Exhaustive function map — Tensor API → primitives

Companion to `PRIMITIVE_SET.md`. Every row is a PyTorch tensor function and the
**exact primitive composition** in the Python library. "Disp" = Vulkan dispatches
after fusion (0 = view/meta, 1 = single fused kernel). "Bwd" = has backward rule
(✓ yes, ✗ no, S subgradient). "Test" = parity case / oracle.

Sources: PyTorch 2.15 `torch` + `Tensor` API (https://docs.pytorch.org/docs/2.15/),
tinygrad Tensor API (https://docs.tinygrad.org/tensor/). Counts are printed at
the end.

---

## A. Tensor creation (24 functions)

| PyTorch API | Composition (primitives) | Disp | Bwd | Test |
|---|---|---:|:---:|---|
| `torch.tensor / as_tensor / from_numpy` | COPY host→dev | 1 | ✗ | from_numpy 0 |
| `torch.empty / empty_like` | ALLOC | 0 | ✗ | shape |
| `torch.zeros / zeros_like` | ALLOC + CONST(0) fill | 1 | ✗ | zeros |
| `torch.ones / ones_like` | ALLOC + CONST(1) fill | 1 | ✗ | ones |
| `torch.full / full_like` | ALLOC + CONST(v) fill | 1 | ✗ | full |
| `torch.arange` | RNG-like index→FMA (`start+i·step`) | 1 | ✗ | arange |
| `torch.linspace` | arange + FMA scale/shift | 1 | ✗ | linspace |
| `torch.logspace` | linspace + POW(10,·) | 1 | ✗ | logspace |
| `torch.eye / identity` | arange CMP broadcast | 1 | ✗ | eye |
| `torch.rand / rand_like` | RNG uniform | 1 | ✗ | distribution |
| `torch.randn / randn_like` | RNG normal (Box–Muller) | 1 | ✗ | distribution |
| `torch.randint / randint_like` | RNG randint | 1 | ✗ | distribution |
| `torch.randperm` | RNG randint + argsort(scan) | 2 | ✗ | distribution |
| `torch.normal` | RNG normal(mean,std) | 1 | ✗ | distribution |
| `torch.uniform_ / normal_` | RNG uniform/normal in-place | 1 | ✗ | init |
| `torch.empty_strided` | ALLOC(strided) | 0 | ✗ | shape |
| `torch.tril / triu` | arange CMP + WHERE mask | 1 | ✗ | triu |
| `torch.diag / diagflat / diag_embed` | arange CMP + WHERE | 1 | ✗ | diag |
| `torch.meshgrid` | arange + EXPAND(2 axes) | 0 | ✗ | meshgrid |
| `torch.indices` | arange + EXPAND | 0 | ✗ | shape |
| `torch.tile / repeat` | view EXPAND + COPY | 1 | ✓ (sum repeats) | repeat |
| `torch.repeat_interleave` | EXPAND + reshape | 1 | ✓ | repeat |
| `torch.frombuffer / from_blob` | COPY external ptr | 0 | ✗ | interop |
| `torch.as_strided` | view(stride) | 0 | ✗ | view |

## B. Movement / views / shape (22 functions)

| PyTorch API | Composition | Disp | Bwd | Test |
|---|---|---:|:---:|---|
| `reshape / view` | reshape | 0 | ✓ | view |
| `Tensor.view` | reshape | 0 | ✓ | view |
| `permute / transpose / T / mT` | permute | 0 | ✓ | transpose |
| `transpose` (dim0,dim1) | permute | 0 | ✓ | transpose |
| `expand / expand_as / broadcast_to` | expand | 0 | ✓ (reduce) | broadcast |
| `squeeze / unsqueeze` | reshape | 0 | ✓ | squeeze |
| `flatten / unflatten` | reshape | 0 | ✓ | flatten |
| `contiguous` | COPY (only if strided) | 0/1 | ✓ | stride |
| `flip` | flip (negative stride) | 0 | ✓ | flip |
| `narrow / chunk / split / split_with_sizes` | shrink | 0 | ✓ | split |
| `pad / F.pad` | pad view (+ CONST fill) | 0/1 | ✓ | pad |
| `cat / concat` | COPY into preallocated slots | 1 | ✓ | cat |
| `stack / hstack / vstack / dstack / column_stack` | unsqueeze + cat | 1 | ✓ | stack |
| `moveaxis / swapaxes / swappedaxes` | permute | 0 | ✓ | permute |
| `roll` | roll (index shift) | 1 | ✓ | roll |
| `unbind / dstack` | split + squeeze | 0 | ✓ | unbind |
| `unfold / F.unfold` | unfold view | 0 | ✓ | unfold |
| `Tensor.unfold` | unfold view | 0 | ✓ | unfold |
| `F.fold` | scatter-add into output | 1 | ✓ | fold |
| `pixel_shuffle / pixel_unshuffle` | permute + reshape | 0 | ✓ | vision |
| `rot90` | flip + permute | 0 | ✓ | rot90 |
| `resize_ / resize_as_` | reshape (+ pad/trunc) | 0/1 | ✗ | shape |

## C. Indexing, slicing, gathering (17 functions)

| PyTorch API | Composition | Disp | Bwd | Test |
|---|---|---:|:---:|---|
| `__getitem__` (int/slice/ellipsis) | shrink view | 0 | ✓ (scatter) | getitem |
| `__setitem__` | scatter/assign | 1 | ✓ | setitem |
| `index_select / take` | index gather | 1 | ✓ | gather |
| `gather` | index gather | 1 | ✓ (scatter-add) | gather |
| `scatter / scatter_ / scatter_add_` | index scatter mode | 1 | ✓ | scatter |
| `index_add / index_copy / index_fill / index_put` | index scatter modes | 1 | ✓ | scatter |
| `take_along_dim / put_along_dim` | index gather/scatter | 1 | ✓ | gather |
| `where (3-arg) / masked_select` | WHERE + index compact | 1 | ✓ / ✗ | where |
| `masked_fill / masked_scatter` | WHERE | 1 | ✓ (branches) | masked |
| `nonzero` | index compact | 1 | ✗ | nonzero |
| `argwhere` | nonzero | 1 | ✗ | nonzero |
| `select / narrow` | shrink | 0 | ✓ | select |
| `diagonal / diagonal_scatter` | shrink + permute | 0 | ✓ | diag |
| `topk / argsort / sort` | scan/index + sort | 2 | ✗ | sort |
| `kthvalue` | sort + index | 2 | ✗ | sort |
| `unique / unique_consecutive` | sort + scan | 2 | ✗ | unique |
| `bucketize / searchsorted` | scan/binary | 1 | ✗ | bucketize |

## D. Pointwise unary math (20 functions)

All are a single fused program (1 dispatch) unless noted; formulas from PyTorch
docs. Backward per `PRIMITIVE_SET` op table.

| PyTorch API | Composition (fused op-program) | Disp | Bwd | Test |
|---|---|---:|:---:|---|
| `abs / absolute` | ABS | 1 | S | abs |
| `neg / negative` | NEG | 1 | ✓ | neg |
| `square` | MUL(x,x) | 1 | ✓ | square |
| `reciprocal` | RECIP | 1 | ✓ | recip |
| `exp / exp2 / expm1` | EXP / EXP scaled / EXP−1 | 1 | ✓ | exp |
| `log / log2 / log10 / log1p` | LOG scaled / LOG(1+x) | 1 | ✓ | log |
| `sqrt / rsqrt` | SQRT / RECIP(SQRT) | 1 | ✓ | sqrt |
| `pow (unary)` | POW | 1 | ✓ | pow |
| `sin / cos / tan` | SIN/COS/TAN | 1 | ✓ | trig |
| `asin / acos / atan` | fused poly | 1 | ✓ | trig |
| `sinh / cosh / tanh` | EXP combos / TANH | 1 | ✓ | trig |
| `asinh / acosh / atanh` | LOG+SQRT combos | 1 | ✓ | trig |
| `erf / erfc / erfinv` | fused approx | 1 | ✓ | erf |
| `lgamma / gamma / digamma` | fused approx (defer) | 1 | ✓ | special |
| `floor / ceil / round / trunc / frac` | fused round ops | 1 | S | rounding |
| `sign / signbit` | SIGN | 1 | ✗ | sign |
| `clamp / clip` | MIN/MAX | 1 | S | clamp |
| `nan_to_num` | WHERE(isnan,...) | 1 | ✗ | special |
| `deg2rad / rad2deg` | MUL const | 1 | ✓ | scale |
| `sinc` | SIN(x)/x fused | 1 | ✓ | trig |

## E. Pointwise binary / comparison (15 functions)

| PyTorch API | Composition | Disp | Bwd | Test |
|---|---|---:|:---:|---|
| `add / sub / mul / div / floor_divide` | ADD/SUB/MUL/DIV/FDIV-floor | 1 | ✓ | binary |
| `mod / remainder / fmod` | MOD fused | 1 | ✓ | mod |
| `pow (binary)` | POW | 1 | ✓ | pow |
| `maximum / minimum` | MAX/MIN | 1 | S | minmax |
| `where (cond)` | WHERE | 1 | branches | where |
| `atan2` | fused ATAN2 | 1 | ✓ | trig |
| `hypot` | SQRT(x²+y²) | 1 | ✓ | hypot |
| `copysign` | bit ops | 1 | ✓ | sign |
| `lerp` | x + w(y−x) fused | 1 | ✓ | lerp |
| `logaddexp / logaddexp2` | max + LOG(EXP) fused | 1 | ✓ | logaddexp |
| `eq / ne / lt / le / gt / ge` | CMP opcodes | 1 | ✗ | compare |
| `isclose / allclose` | ABS+CMP + REDUCE | 1/2 | ✗ | compare |
| `isfinite / isinf / isnan` | fused predicate | 1 | ✗ | special |
| `logical_and/or/not/xor` | CMP/WHERE | 1 | ✗ | logical |
| `bitwise_and/or/xor/not`, shifts | bit opcodes | 1 | ✗ | bit |

## F. Reductions (16 functions)

| PyTorch API | Composition | Disp | Bwd | Test |
|---|---|---:|:---:|---|
| `sum / sum(dim)` | REDUCE ADD | 1 | ✓ (broadcast) | sum |
| `prod` | REDUCE MUL | 1 | ✓ (zero-aware) | prod |
| `mean` | REDUCE ADD + DIV(n) | 1 | ✓ | mean |
| `max / min (dim)` | REDUCE MAX/MIN (+index) | 1 | S (ties) | max |
| `amax / amin` | REDUCE MAX/MIN | 1 | S | max |
| `argmax / argmin` | REDUCE indexed MAX/MIN | 1 | ✗ | argmax |
| `var / std (correction)` | 2×REDUCE ADD + FMA | 2 | ✓ | var |
| `var_mean / std_mean` | var + mean | 2 | ✓ | var |
| `any / all` | REDUCE ANY/ALL | 1 | ✗ | logical |
| `logsumexp` | REDUCE MAX + REDUCE ADD + LOG | 2 | ✓ | logsumexp |
| `cumsum / cumprod` | SCAN ADD/MUL | 1 | ✓ | scan |
| `cummax / cummin` | SCAN MAX/MIN (+index) | 1 | ✗ | scan |
| `logcumsumexp` | SCAN + LOG | 1 | ✓ | scan |
| `count_nonzero` | REDUCE ADD of (x≠0) | 1 | ✗ | count |
| `norm / vector_norm / matrix_norm` | POW + REDUCE + POW | 2 | ✓ | norm |
| `dist` | SUB + norm | 2 | ✓ | dist |

## G. Linear algebra & matmul (12 functions)

| PyTorch API | Composition | Disp | Bwd | Test |
|---|---|---:|:---:|---|
| `matmul / @ / bmm` | GEMM | 1 | ✓ (2 GEMM) | matmul |
| `mm / mv` | GEMM / gemv | 1 | ✓ | matmul |
| `dot / vdot / inner` | MUL + REDUCE / gemv | 1 | ✓ | dot |
| `outer / ger` | EXPAND + MUL | 1 | ✓ | outer |
| `addmm / addmv / addr` | GEMM(accumulate,bias) | 1 | ✓ | addmm |
| `einsum` | expand + GEMM/REDUCE | varies | ✓ | einsum |
| `tensordot` | reshape + GEMM | 1 | ✓ | tensordot |
| `bilinear` | 2×GEMM | 2 | ✓ | bilinear |
| `trace` | diagonal + REDUCE | 1 | ✓ | trace |
| `kron` | EXPAND + MUL | 1 | ✓ | kron |
| `cross` | fused cross product | 1 | ✓ | cross |
| `svd / qr / cholesky / inv / pinv / det / solve` | iterative GEMM family (defer, out of scope v0.2) | — | ✗ | — |

## H. Distance & other tensor ops (7 functions)

| PyTorch API | Composition | Disp | Bwd | Test |
|---|---|---:|:---:|---|
| `pairwise_distance` | SUB + norm | 2 | ✓ | dist |
| `cosine_similarity` | normalize + MUL+REDUCE | 2 | ✓ | dist |
| `pdist / cdist` | SUB + norm (broadcast) | 2 | ✓ | dist |
| `sort / argsort (values)` | sort primitive | 1 | ✗ | sort |
| `topk` | sort + shrink | 1 | ✗ | topk |
| `histc / bincount` | index scatter | 1 | ✗ | hist |
| `renorm` | norm + WHERE scale | 2 | ✓ | norm |

---

**MAP_TENSOR count:** creation 24 + movement 22 + indexing 17 + unary 20 +
binary 15 + reductions 16 + linalg 12 + other 7 = **133 tensor functions mapped**.
