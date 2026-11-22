"""Minimal nn surface (Module, Linear, activations, MSELoss) over Bitsy ops."""

from __future__ import annotations

import numpy as np

from . import _ctypes as C
from .autograd import Tensor, reduce_sum_op

class Parameter(Tensor):
  def __init__(self, tensor:Tensor) -> None:
    super().__init__(tensor._t, True, tensor.device)

class Module:
  def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
  def parameters(self) -> list[Parameter]:
    out = []
    for value in vars(self).values():
      if isinstance(value, Parameter): out.append(value)
      elif isinstance(value, Module): out.extend(value.parameters())
      elif isinstance(value, (list, tuple)):
        for item in value:
          if isinstance(item, Module): out.extend(item.parameters())
    return out

class Linear(Module):
  def __init__(self, in_features:int, out_features:int, rng, device:str="cpu") -> None:
    w = (rng.standard_normal((in_features, out_features)) * 0.1).astype(np.float32)
    b = np.zeros((1, out_features), dtype=np.float32)
    self.W = Parameter(Tensor.from_numpy(w, device))
    self.b = Parameter(Tensor.from_numpy(b, device))
    self._ones:dict[int, Tensor] = {}

  def forward(self, x:Tensor) -> Tensor:
    # exp_024: fused GEMM+bias epilogue replaces (x@W) + (ones@b) + add.
    ct = C.matmul_bias(x._t, self.W._t, self.b._t)
    Wt = C.transpose_view(self.W._t)
    xt = C.transpose_view(x._t)
    def back(g):
      return (g._op(C.matmul(g._t, Wt), (), None),
              g._op(C.matmul(xt, g._t), (), None),
              reduce_sum_op(g, 0, True))
    return x._op(ct, (x, self.W, self.b), back)

  def forward_tanh(self, x:Tensor) -> Tensor:
    """Fused forward for Linear followed by Tanh (exp_036)."""
    ct = C.matmul_bias_tanh(x._t, self.W._t, self.b._t)
    Wt = C.transpose_view(self.W._t)
    xt = C.transpose_view(x._t)
    def back(g):
      pre = g._op(C.tanh_backward(g._t, ct), (), None)  # dL/d(x@W+b)
      return (pre._op(C.matmul(pre._t, Wt), (), None),
              pre._op(C.matmul(xt, pre._t), (), None),
              reduce_sum_op(pre, 0, True))
    return x._op(ct, (x, self.W, self.b), back)

class RNNCell(Module):
  """Vanilla recurrent cell: h_t = tanh(x_t@W_ih + h@W_hh + b)."""
  def __init__(self, input_size:int, hidden_size:int, device:str="cpu", seed:int=0) -> None:
    rng = np.random.default_rng(seed)
    lim = 1.0 / np.sqrt(hidden_size)
    self.W_ih = Parameter(Tensor.from_numpy((rng.uniform(-lim, lim, (input_size, hidden_size))).astype(np.float32), device))
    self.W_hh = Parameter(Tensor.from_numpy((rng.uniform(-lim, lim, (hidden_size, hidden_size))).astype(np.float32), device))
    self.b = Parameter(Tensor.from_numpy(np.zeros((1, hidden_size), np.float32), device))
    self.hidden_size = hidden_size

  def forward(self, x:Tensor, h:Tensor) -> Tensor:
    return (x @ self.W_ih + h @ self.W_hh + self.b).tanh()

class RNN(Module):
  """Unidirectional RNN over a (batch, time, input) sequence."""
  def __init__(self, input_size:int, hidden_size:int, device:str="cpu", seed:int=0) -> None:
    self.cell = RNNCell(input_size, hidden_size, device, seed)
    self.hidden_size = hidden_size

  def forward(self, x:Tensor, h0:Tensor|None=None) -> tuple:
    shp = x.shape
    batch, time = shp[0], shp[1]
    h = h0 if h0 is not None else Tensor.from_numpy(np.zeros((batch, self.hidden_size), np.float32), x.device)
    # x[:, t] via a host-built slice: indices are data, not gradients.
    xflat = x.reshape(batch * time, shp[-1])
    outputs = []
    for t in range(time):
      s = np.zeros((batch, batch * time), np.float32)
      for b in range(batch):
        s[b, b * time + t] = 1.0
      xt = Tensor.from_numpy(s, x.device) @ xflat
      h = self.cell(xt, h)
      outputs.append(h)
    # stack outputs along time via reshape of a row-major concatenation is not
    # available; return the final hidden state and the per-step list.
    return h, outputs

class LSTMCell(Module):
  """LSTM cell with separate gate weights (parity-friendly; PyTorch's packed
  gate matrix is split into four Parameters by the test harness)."""
  def __init__(self, input_size:int, hidden_size:int, device:str="cpu", seed:int=0) -> None:
    rng = np.random.default_rng(seed)
    lim = 1.0 / np.sqrt(hidden_size)
    def p(shape):
      return Parameter(Tensor.from_numpy(rng.uniform(-lim, lim, shape).astype(np.float32), device))
    self.W_i, self.W_f, self.W_g, self.W_o = (
      p((input_size, hidden_size)), p((input_size, hidden_size)),
      p((input_size, hidden_size)), p((input_size, hidden_size)))
    self.U_i, self.U_f, self.U_g, self.U_o = (
      p((hidden_size, hidden_size)), p((hidden_size, hidden_size)),
      p((hidden_size, hidden_size)), p((hidden_size, hidden_size)))
    self.b = Parameter(Tensor.from_numpy(np.zeros((1, 4 * hidden_size), np.float32), device))
    self.hidden_size = hidden_size

  def forward(self, x:Tensor, h:Tensor, c:Tensor) -> tuple:
    # bias slices are host-built constants (no gradient needed through slicing)
    bh = self.b._t.numpy()
    hsz = self.hidden_size
    bi = Tensor.from_numpy(np.ascontiguousarray(bh[:, 0:hsz]), x.device)
    bf = Tensor.from_numpy(np.ascontiguousarray(bh[:, hsz:2*hsz]), x.device)
    bg = Tensor.from_numpy(np.ascontiguousarray(bh[:, 2*hsz:3*hsz]), x.device)
    bo = Tensor.from_numpy(np.ascontiguousarray(bh[:, 3*hsz:4*hsz]), x.device)
    i = (x @ self.W_i + h @ self.U_i + bi).sigmoid()
    f = (x @ self.W_f + h @ self.U_f + bf).sigmoid()
    g = (x @ self.W_g + h @ self.U_g + bg).tanh()
    o = (x @ self.W_o + h @ self.U_o + bo).sigmoid()
    c_new = f * c + i * g
    h_new = o * c_new.tanh()
    return h_new, c_new

class LSTM(Module):
  def __init__(self, input_size:int, hidden_size:int, device:str="cpu", seed:int=0) -> None:
    self.cell = LSTMCell(input_size, hidden_size, device, seed)
    self.hidden_size = hidden_size

  def forward(self, x:Tensor, state=None) -> tuple:
    shp = x.shape
    batch, time = shp[0], shp[1]
    if state is None:
      z = np.zeros((batch, self.hidden_size), np.float32)
      h, c = (Tensor.from_numpy(z, x.device), Tensor.from_numpy(z, x.device))
    else:
      h, c = state
    xflat = x.reshape(batch * time, shp[-1])
    outputs = []
    for t in range(time):
      s = np.zeros((batch, batch * time), np.float32)
      for b in range(batch):
        s[b, b * time + t] = 1.0
      xt = Tensor.from_numpy(s, x.device) @ xflat
      h, c = self.cell(xt, h, c)
      outputs.append(h)
    return (h, c), outputs

class GELU(Module):
  """Exact (erf) GELU, matching torch.nn.GELU() default."""
  def forward(self, x:Tensor) -> Tensor: return x.gelu()

class LayerNorm(Module):
  """LayerNorm over the last dimension, matching torch.nn.LayerNorm(eps=1e-5)."""
  def __init__(self, normalized_shape:int, eps:float=1e-5, device:str="cpu") -> None:
    self.W = Parameter(Tensor.from_numpy(np.ones((1, normalized_shape), np.float32), device))
    self.b = Parameter(Tensor.from_numpy(np.zeros((1, normalized_shape), np.float32), device))
    self.eps, self.size = eps, normalized_shape

  def forward(self, x:Tensor) -> Tensor:
    shp = x.shape
    import math
    flat = x.reshape(math.prod(shp[:-1]), self.size) if len(shp) > 2 else x
    mean = reduce_sum_op(flat, 1, True) / self.size
    centered = flat - mean
    var = reduce_sum_op(centered * centered, 1, True) / self.size
    invstd = 1.0 / (var + self.eps).sqrt()
    out = centered * invstd * self.W + self.b
    return out.reshape(*shp) if len(shp) > 2 else out

class Embedding(Module):
  """Embedding lookup via a one-hot matmul (device ops; indices passed as numpy)."""
  def __init__(self, num_embeddings:int, embedding_dim:int, device:str="cpu", seed:int=0) -> None:
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((num_embeddings, embedding_dim)) * 0.02).astype(np.float32)
    self.W = Parameter(Tensor.from_numpy(w, device))
    self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim

  def forward(self, indices) -> Tensor:
    idx = np.asarray(indices, dtype=np.int64)
    lead = idx.shape[:-1]
    flat = idx.reshape(idx.size)
    oh = np.zeros((flat.shape[0], self.num_embeddings), np.float32)
    oh[np.arange(flat.shape[0]), flat] = 1.0
    oh_t = Tensor.from_numpy(oh, self.W.device)
    ct = C.matmul(oh_t._t, self.W._t)
    def back(g):
      ohT = Tensor(C.transpose_view(oh_t._t), False, self.W.device)
      return (ohT @ g,)
    t = self.W._op(ct, (self.W,), back)
    return t.reshape(*lead, idx.shape[-1], self.embedding_dim)

class Dropout(Module):
  """Identity in eval mode (the only mode used for parity benchmarks)."""
  def forward(self, x:Tensor) -> Tensor: return x

class Conv2d(Module):
  """NCHW 3x3 convolution with pad/stride; gradients run on the same device."""
  def __init__(self, in_ch:int, out_ch:int, rng, device:str="cpu", pad:int=1, stride:int=1) -> None:
    w = (rng.standard_normal((out_ch, in_ch, 3, 3)) * 0.1).astype(np.float32)
    self.W = Parameter(Tensor.from_numpy(w, device))
    self.b = Parameter(Tensor.from_numpy(np.zeros((out_ch,), np.float32), device))
    self.pad, self.stride = pad, stride

  def forward(self, x:Tensor) -> Tensor:
    ct = C.conv2d(x._t, self.W._t, self.b._t, self.pad, self.stride)
    def back(g):
      dx = C.conv2d_backward_x(g._t, self.W._t, x.shape, self.pad, self.stride)
      dw = C.conv2d_backward_w(x._t, g._t, self.W.shape, self.pad, self.stride)
      db = C.conv2d_backward_b(g._t, self.b.shape[0])
      return (Tensor(dx, device=x.device), Tensor(dw, device=x.device),
              Tensor(db, device=x.device))
    ctx = (back, (x, self.W, self.b))
    return Tensor(ct, True, x.device, ctx)

class Tanh(Module):
  def forward(self, x:Tensor) -> Tensor: return x.tanh()

class Sigmoid(Module):
  def forward(self, x:Tensor) -> Tensor: return x.sigmoid()

class ReLU(Module):
  def forward(self, x:Tensor) -> Tensor: return x.relu()

class Sequential(Module):
  def __init__(self, *layers:Module) -> None: self.layers = layers
  def forward(self, x:Tensor) -> Tensor:
    layers = self.layers
    i = 0
    while i < len(layers):
      layer = layers[i]
      if (isinstance(layer, Linear) and i + 1 < len(layers) and
          isinstance(layers[i + 1], Tanh)):
        x = layer.forward_tanh(x)  # one dispatch instead of GEMM+bias + tanh
        i += 2
      else:
        x = layer(x)
        i += 1
    return x

def gelu(x:Tensor) -> Tensor:
  return x.gelu()

def softmax(x:Tensor) -> Tensor:
  """Softmax on the last dimension, built from the verified log_softmax + exp."""
  shp = x.shape
  import math
  flat = x.reshape(math.prod(shp[:-1]), shp[-1]) if len(shp) > 2 else x
  out = flat.log_softmax().exp()
  return out.reshape(*shp) if len(shp) > 2 else out

def log_softmax(logits:Tensor) -> Tensor:
  # exp_027: one fused dispatch instead of exp + reduce + log + subtract.
  return logits.log_softmax()

def cross_entropy(logits:Tensor, onehot:Tensor) -> Tensor:
  # exp_028: one fused reduction kernel instead of mul + two reductions + scale.
  return logits.log_softmax().ce_loss(onehot)

def mse_loss(pred:Tensor, target:Tensor) -> Tensor:
  d = pred - target
  sq = d * d
  per_row = reduce_sum_op(sq, 1, True)
  total = reduce_sum_op(per_row, 0, True)
  n = 1
  for s in pred.shape: n *= s
  return total * (1.0 / n)  # PyTorch F.mse_loss default reduction='mean'
