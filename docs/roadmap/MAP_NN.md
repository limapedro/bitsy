# Exhaustive function map — nn.functional + nn layers → primitives

Companion to `PRIMITIVE_SET.md`. Every PyTorch `nn.functional` function and `nn`
module from the 2.15 reference (https://docs.pytorch.org/docs/2.15/nn.functional.html,
https://docs.pytorch.org/docs/2.15/nn.html) mapped to its Python composition of
frozen primitives. **Disp** = dispatches after fusion. Layers are plain Python
classes (`__call__`), adding zero C lines.

---

## A. Non-linear activations (29 functions)

All are 1 fused dispatch. `nn.X` module = thin stateful wrapper around the
functional form (or identity if stateless).

| F. API | Formula (fused op-program) | nn module | Bwd | Test |
|---|---|---|---|---|
| `relu / relu_` | MAX(x,0) | ReLU / ReLU6 | ✓/S | relu |
| `relu6` | MIN(MAX(x,0),6) | ReLU6 | S | relu6 |
| `elu / elu_` | x>0?x:α(EXP(x)−1) | ELU | ✓ | elu |
| `selu` | scale·(max(0,x)+min(0,α(EXP(x)−1))) | SELU | ✓ | selu |
| `celu` | max(0,x)+min(0,α(EXP(x/α)−1)) | CELU | ✓ | celu |
| `leaky_relu / _` | max(0,x)+slope·min(0,x) | LeakyReLU | ✓ | leaky |
| `prelu` | max(0,x)+w·min(0,x) (w param) | PReLU | ✓ | prelu |
| `rrelu / _` | randomized slope | RReLU | ✓ | rrelu |
| `hardtanh / _` | clamp(x,min,max) | Hardtanh | S | hardtanh |
| `hardswish` | x·clamp(x+3,0,6)/6 | Hardswish | ✓ | hardswish |
| `hardsigmoid` | clamp(x+3,0,6)/6 | Hardsigmoid | ✓ | hardsig |
| `threshold / _` | x>th?x:value | Threshold | ✗ | threshold |
| `glu` | split→x₁·sigmoid(x₂) | GLU | ✓ | glu |
| `gelu` | x·Φ(x) (erf exact) | GELU | ✓ | gelu 1.2e-7 |
| `logsigmoid` | log(sigmoid(x)) stable | — | ✓ | logsig |
| `hardshrink` | \|x\|>λ?x:0 | Hardshrink | ✓ | shrink |
| `tanhshrink` | x−tanh(x) | Tanhshrink | ✓ | shrink |
| `softshrink` | soft threshold | Softshrink | ✓ | shrink |
| `softsign` | x/(1+\|x\|) | — | ✓ | softsign |
| `softplus` | (1/β)log(1+EXP(βx)) | Softplus | ✓ | softplus |
| `softmin` | softmax(−x) | Softmin | ✓ | softmax |
| `softmax` | EXP(x−max)/SUM | Softmax | ✓ | softmax 3e-8 |
| `log_softmax` | (x−max)−LOG(SUM) | — | ✓ | logsoftmax |
| `gumbel_softmax` | RNG gumbel + softmax | — | ✗ | gumbel |
| `tanh` | tanh | Tanh | ✓ | tanh |
| `sigmoid` | sigmoid | Sigmoid | ✓ | sigmoid |
| `silu` | x·sigmoid(x) | SiLU | ✓ | silu |
| `mish` | x·tanh(softplus(x)) | Mish | ✓ | mish |
| `normalize` | x/max(norm(x),eps) | — | ✓ | normalize |

## B. Normalization functions (8 functions)

| F. API | Composition | nn module | Bwd | Test |
|---|---|---|---|---|
| `layer_norm` | REDUCE mean, REDUCE var, fused affine γ·(x−μ)/σ+β | LayerNorm | ✓ | LN 5.4e-7 |
| `rms_norm` | REDUCE sumsq → x/√(mean+eps)·γ | RMSNorm | ✓ | rmsnorm |
| `batch_norm` | REDUCE mean/var over batch + affine + running stats | BatchNorm1d/2d/3d | ✓ | batchnorm |
| `group_norm` | reshape→batch_norm over groups | GroupNorm | ✓ | groupnorm |
| `instance_norm` | reshape→layer_norm per (N,C) | InstanceNorm1d/2d/3d | ✓ | instancenorm |
| `local_response_norm` | REDUCE squares over channel window + affine | — | ✓ | lrn |
| `sync_batch_norm` | batch_norm across devices (dist, out of scope) | SyncBatchNorm | — | — |
| `LayerNorm2d` (tinygrad) | layer_norm axis=1 | — | ✓ | LN |

## C. Convolution functions (7 functions)

| F. API | Composition | nn module | Bwd | Test |
|---|---|---|---|---|
| `conv1d` | unfold(view) + GEMM (+bias) | Conv1d | ✓ | conv1d |
| `conv2d` | unfold(view) + GEMM (+bias) | Conv2d | ✓ | CIFAR 1.2e-5 |
| `conv3d` | unfold(view) + GEMM (+bias) | Conv3d | ✓ | conv3d |
| `conv_transpose1d/2d/3d` | scatter-im2col + GEMM | ConvTranspose* | ✓ | convT |
| `unfold` (F) | unfold view | — | ✓ | unfold |
| `fold` (F) | scatter-add | — | ✓ | fold |
| `LazyConv*` | shape inference + conv | LazyConv* | ✓ | lazy |

## D. Pooling functions (7 functions)

| F. API | Composition | nn module | Bwd | Test |
|---|---|---|---|---|
| `avg_pool1d/2d/3d` | REDUCE ADD over window ÷ count | AvgPool* | ✓ | avgpool |
| `max_pool1d/2d/3d` | REDUCE MAX over window (+index) | MaxPool* | ✓ (scatter) | maxpool |
| `max_unpool1d/2d/3d` | index scatter by saved indices | MaxUnpool* | ✗ | unpool |
| `lp_pool1d/2d/3d` | POW + REDUCE ADD + POW(1/p) | LPPool* | ✓ | lppool |
| `adaptive_max_pool1d/2d/3d` | max_pool with computed kernel | AdaptiveMaxPool* | ✓ | adapool |
| `adaptive_avg_pool1d/2d/3d` | avg_pool with computed kernel | AdaptiveAvgPool* | ✓ | adapool |
| `fractional_max_pool2d/3d` | RNG boundaries + max_pool | FractionalMaxPool* | ✗ | fracpool |

## E. Attention mechanisms (6 functions)

| F. API | Composition | nn module | Bwd | Test |
|---|---|---|---|---|
| `scaled_dot_product_attention` | GEMM(Q,Kᵀ)·scale → +mask → softmax → GEMM(·,V) | — | ✓ (recompute) | GPT-2 1.16e-6 |
| `attention_bias` (causal etc.) | CONST mask (arange CMP) | — | ✗ | causal mask |
| `MultiheadAttention` | 4×Linear + reshape + SDPA + Linear | MultiheadAttention | ✓ | MHA |
| `TransformerEncoderLayer` | LN + MHA + LN + MLP (FFN) | TransformerEncoderLayer | ✓ | transformer |
| `TransformerDecoderLayer` | LN + MHA×2 + LN + MLP | TransformerDecoderLayer | ✓ | transformer |
| `Transformer / Encoder / Decoder` | stack of layers + norm | Transformer* | ✓ | transformer |

## F. Linear functions (2 functions)

| F. API | Composition | nn module | Bwd | Test |
|---|---|---|---|---|
| `linear` | x·Wᵀ + b (GEMM + bias) | Linear / LazyLinear | ✓ | W grad 6e-8 |
| `bilinear` | 2×GEMM | Bilinear | ✓ | bilinear |

## G. Dropout functions (4 functions)

| F. API | Composition | nn module | Bwd | Test |
|---|---|---|---|---|
| `dropout` | RNG mask + MUL (inverted, TRAINING only) | Dropout | ✓ (mask) | eval identity |
| `alpha_dropout` | dropout + affine transform | AlphaDropout | ✓ | alphadrop |
| `feature_alpha_dropout` | channel mask + alpha | FeatureAlphaDropout | ✓ | fadrop |
| `dropout1d/2d/3d` | channel RNG mask + MUL | Dropout1d/2d/3d | ✓ | dropout2d |

## H. Sparse / embedding (3 functions)

| F. API | Composition | nn module | Bwd | Test |
|---|---|---|---|---|
| `embedding` | index gather (one-hot matmul) | Embedding | ✓ (scatter-add W) | forward 0 |
| `embedding_bag` | gather + REDUCE (sum/mean/max) | EmbeddingBag | ✓ | embbag |
| `one_hot` | arange CMP + WHERE | — | ✗ | onehot |

## I. Loss functions (21 functions)

All = core primitives + REDUCE(reduction=mean/sum/none).

| F. API | Composition | nn module | Bwd | Test |
|---|---|---|---|---|
| `cross_entropy` | log_softmax + gather + REDUCE | CrossEntropyLoss | ✓ | GPT-2 loss 0.0 |
| `nll_loss` | gather log-prob + REDUCE | NLLLoss | ✓ | nll |
| `mse_loss` | REDUCE((p−t)²) | MSELoss | ✓ | MLP |
| `l1_loss` | REDUCE(\|p−t\|) | L1Loss | S | l1 |
| `huber_loss / smooth_l1_loss` | piecewise fused + REDUCE | HuberLoss / SmoothL1Loss | ✓ | huber |
| `binary_cross_entropy` | fused log/where + REDUCE | BCELoss | ✓ | bce |
| `binary_cross_entropy_with_logits` | stable BCE-logits fused + REDUCE | BCEWithLogitsLoss | ✓ | bcelogits |
| `poisson_nll_loss` | x−t·log(x)+lgamma fused | PoissonNLLLoss | ✓ | poisson |
| `kl_div` | t(log t−x) fused + REDUCE | KLDivLoss | ✓ | kl |
| `gaussian_nll_loss` | fused log(σ²)+... | GaussianNLLLoss | ✓ | gnll |
| `ctc_loss` | dynamic programming (defer, out of scope) | CTCLoss | ✓ | — |
| `cosine_embedding_loss` | 1−cos + margin | CosineEmbeddingLoss | ✓ | cosloss |
| `hinge_embedding_loss` | max(0,margin−y·t) | HingeEmbeddingLoss | ✓ | hinge |
| `margin_ranking_loss` | max(0,−t(y₁−y₂)+margin) | MarginRankingLoss | ✓ | margin |
| `multi_margin_loss` | fused margin | MultiMarginLoss | ✓ | mmargin |
| `multilabel_margin_loss` | fused | MultiLabelMarginLoss | ✓ | mlmargin |
| `multilabel_soft_margin_loss` | sigmoid+log fused | MultiLabelSoftMarginLoss | ✓ | mlsoft |
| `soft_margin_loss` | log(1+EXP(−y·t)) | SoftMarginLoss | ✓ | softmargin |
| `triplet_margin_loss` | max(0,d(a,p)−d(a,n)+margin) | TripletMarginLoss | ✓ | triplet |
| `triplet_margin_with_distance_loss` | same w/ distance fn | TripletMarginWithDistanceLoss | ✓ | triplet |
| `linear_cross_entropy` | GEMM + CE fused | — | ✓ | linearCE |

## J. Vision functions (6 functions)

| F. API | Composition | nn module | Bwd | Test |
|---|---|---|---|---|
| `interpolate` (nearest/bilinear/bicubic) | index gather + WHERE / lerp | — | ✓ | interp |
| `upsample / nearest / bilinear` | interpolate | Upsample | ✓ | upsample |
| `pad` | pad view | ZeroPad2d etc. | ✓ | pad |
| `grid_sample` | gather w/ computed indices (bilinear) | — | ✓ | grid |
| `affine_grid` | matmul (theta·grid) | — | ✓ | affine |
| `pixel_shuffle / unshuffle` | permute + reshape | PixelShuffle/Unshuffle | ✓ | vision |

## K. nn container / utility layers (5 functions)

| nn API | Composition | Notes |
|---|---|---|
| `Sequential / ModuleList / ModuleDict` | iterate `__call__` | plain Python containers |
| `Parameter / ParameterList / ParameterDict` | Tensor(requires_grad) | parameter holders |
| `Identity / Flatten / Unflatten` | identity / reshape | no kernel |
| `Lambda` | call fn | Python |
| `get_parameters / state_dict / load_state_dict` | object walk | host-side, no kernel |

---

**MAP_NN count:** activations 29 + norm 8 + conv 7 + pool 7 + attention 6 +
linear 2 + dropout 4 + embedding 3 + loss 21 + vision 6 + container 5 =
**98 nn functions/layers mapped**.
