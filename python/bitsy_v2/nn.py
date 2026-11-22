"""nn layers — pure Python compositions over compact C primitives.

Models are plain classes with __call__ (tinygrad philosophy). No C lines added.
"""
from __future__ import annotations

import numpy as np
from .tensor import Tensor


class Parameter(Tensor):
  def __init__(self, rt, arr, name=""):
    a = np.ascontiguousarray(arr, dtype=np.float32)
    super().__init__(rt, rt._raw_tensor(a, True), a.shape, True)
    self.name = name


class Module:
  def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
  def parameters(self):
    out = []
    for v in vars(self).values():
      if isinstance(v, Parameter): out.append(v)
      elif isinstance(v, Module): out.extend(v.parameters())
      elif isinstance(v, (list, tuple)):
        for it in v:
          if isinstance(it, Module): out.extend(it.parameters())
    return out


class Linear(Module):
  def __init__(self, rt, in_f, out_f, seed=0):
    rng = np.random.default_rng(seed)
    self.W = Parameter(rt, rng.standard_normal((out_f, in_f)) * np.sqrt(2.0/in_f), "W")
    self.b = Parameter(rt, np.zeros((1, out_f)), "b")

  def forward(self, x): return x.linear(self.W, self.b)


class ReLU(Module):
  def forward(self, x): return x.relu()


class GELU(Module):
  def forward(self, x): return x.gelu()


class Tanh(Module):
  def forward(self, x): return x.tanh()


class Sigmoid(Module):
  def forward(self, x): return x.sigmoid()


class Sequential(Module):
  def __init__(self, *layers): self.layers = layers
  def forward(self, x):
    for l in self.layers: x = l(x)
    return x


class LayerNorm(Module):
  def __init__(self, rt, dim, eps=1e-5):
    self.W = Parameter(rt, np.ones((1, dim)), "lnW")
    self.b = Parameter(rt, np.zeros((1, dim)), "lnb")
    self.eps, self.dim = eps, dim

  def forward(self, x):
    if len(x.shape) >= 1 and x.shape[-1] <= 256:
      # PR #25: one fused dispatch (was 4)
      h = x.rt.layernorm(x.h, self.W.h, self.b.h, self.eps)
      return type(x)(x.rt, h, x.shape,
                     x.requires_grad or self.W.requires_grad or self.b.requires_grad)
    mean = x.mean(axis=-1, keepdim=True)
    d = x - mean
    var = (d * d).mean(axis=-1, keepdim=True)
    inv = 1.0 / (var + self.eps).sqrt()
    return d * inv * self.W + self.b


class Embedding(Module):
  def __init__(self, rt, num, dim, seed=0):
    rng = np.random.default_rng(seed)
    self.W = Parameter(rt, rng.standard_normal((num, dim)) * 0.02, "embW")
    self.num, self.dim = num, dim

  def forward(self, indices):
    # one-hot lookup via gemm: indices are host data (numpy)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    oh = np.zeros((len(idx), self.num), np.float32)
    oh[np.arange(len(idx)), idx] = 1.0
    oh_t = Tensor.from_numpy(self.W.rt, oh)
    # W is (num,dim); oh @ W -> (len, dim); gradient flows to W via autograd
    from .tensor import Tensor as T
    return T(self.W.rt, self.W.rt.gemm(oh_t.h, self.W.h), (len(idx), self.dim), True)


class RNNCell(Module):
  """h_t = tanh(x @ W_ih + h @ W_hh + b), using separate matmuls."""
  def __init__(self, rt, in_f, hidden, seed=0):
    rng = np.random.default_rng(seed)
    self.W_ih = Parameter(rt, rng.standard_normal((in_f, hidden)) * 0.02, "Wih")
    self.W_hh = Parameter(rt, rng.standard_normal((hidden, hidden)) * 0.02, "Whh")
    self.b = Parameter(rt, np.zeros((1, hidden)), "rb")

  def forward(self, x, h):
    return (x.matmul(self.W_ih) + h.matmul(self.W_hh) + self.b).tanh()


class LSTMCell(Module):
  """One LSTM step with separate per-gate matrices (no concat/slice primitive).

  i = sigmoid(x@W_i + h@U_i); f = sigmoid(...); g = tanh(...); o = sigmoid(...)
  c_t = f*c + i*g ; h_t = o*tanh(c_t)
  """
  def __init__(self, rt, in_f, hidden, seed=0):
    rng = np.random.default_rng(seed)
    s = 0.02
    self.Wi = Parameter(rt, rng.standard_normal((in_f, hidden)) * s)
    self.Ui = Parameter(rt, rng.standard_normal((hidden, hidden)) * s)
    self.Wf = Parameter(rt, rng.standard_normal((in_f, hidden)) * s)
    self.Uf = Parameter(rt, rng.standard_normal((hidden, hidden)) * s)
    self.Wg = Parameter(rt, rng.standard_normal((in_f, hidden)) * s)
    self.Ug = Parameter(rt, rng.standard_normal((hidden, hidden)) * s)
    self.Wo = Parameter(rt, rng.standard_normal((in_f, hidden)) * s)
    self.Uo = Parameter(rt, rng.standard_normal((hidden, hidden)) * s)
    self.hidden = hidden

  def forward(self, x, h, c):
    i = (x.matmul(self.Wi) + h.matmul(self.Ui)).sigmoid()
    f = (x.matmul(self.Wf) + h.matmul(self.Uf)).sigmoid()
    g = (x.matmul(self.Wg) + h.matmul(self.Ug)).tanh()
    o = (x.matmul(self.Wo) + h.matmul(self.Uo)).sigmoid()
    c2 = f * c + i * g
    h2 = o * c2.tanh()
    return h2, c2


class RNN(Module):
  """Unrolled RNN over a list of per-step tensors."""
  def __init__(self, rt, in_f, hidden, seed=0):
    self.cell = RNNCell(rt, in_f, hidden, seed)

  def forward(self, xs, h0=None):
    h = h0 if h0 is not None else Tensor.zeros(self.cell.W_ih.rt, (xs[0].shape[0], self.cell.W_ih.shape[1]))
    outs = []
    for x in xs:
      h = self.cell(x, h)
      outs.append(h)
    return outs, h


class LSTM(Module):
  def __init__(self, rt, in_f, hidden, seed=0):
    self.cell = LSTMCell(rt, in_f, hidden, seed)

  def forward(self, xs, state=None):
    rt = self.cell.Wi.rt
    b = xs[0].shape[0]; H = self.cell.hidden
    h, c = state if state is not None else (Tensor.zeros(rt, (b,H)), Tensor.zeros(rt, (b,H)))
    outs = []
    for x in xs:
      h, c = self.cell(x, h, c)
      outs.append(h)
    return outs, (h, c)
